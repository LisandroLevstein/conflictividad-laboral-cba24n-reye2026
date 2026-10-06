# -*- coding: utf-8 -*-
"""Entrena BETO con ajuste y predice evaluación sin leer sus etiquetas."""
from __future__ import annotations

import argparse
import math
import time

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup

from common import (
    ARTIFACTS,
    DATA_VALIDATION,
    ensure_dirs,
    hardware_settings,
    load_config,
    load_corpus,
    read_json,
    seed_everything,
    write_json,
)
from training import TextDataset, make_model, predict


def main(seed: int) -> None:
    ensure_dirs()
    if not torch.cuda.is_available():
        raise RuntimeError("Se requiere CUDA")
    cfg = load_config()
    hw = hardware_settings(cfg)
    threshold_file = ARTIFACTS / f"seed_{seed}" / "threshold.json"
    if not threshold_file.exists():
        raise FileNotFoundError("Falta threshold.json; ejecute primero 02_cv_ajuste.py")
    threshold = float(read_json(threshold_file)["threshold"])
    seed_everything(seed)

    # Únicos archivos con etiquetas leídos en este script: los 250 de ajuste.
    ajuste = pd.read_csv(DATA_VALIDATION / "muestra_ajuste_etiquetada.csv")
    ajuste["etiqueta"] = ajuste["etiqueta"].str.strip().str.upper()
    ajuste = ajuste[ajuste["etiqueta"].isin(["SI", "NO"])].copy()
    ajuste["y"] = (ajuste["etiqueta"] == "SI").astype(int)

    # La evaluación se identifica exclusivamente mediante el archivo de diseño.
    evaluacion = pd.read_csv(DATA_VALIDATION / "muestra_evaluacion_diseno.csv")
    corpus = load_corpus()
    train_df = ajuste.merge(corpus, on="id_doc", validate="one_to_one")
    eval_df = evaluacion.merge(corpus, on="id_doc", validate="one_to_one")
    train_df = train_df.sort_values("id_doc").reset_index(drop=True)
    eval_df = eval_df.sort_values("id_doc").reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name"], revision=cfg["model_revision"], use_fast=True
    )
    tokenizer.truncation_side = "right"
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    train_ds = TextDataset(train_df["texto_analisis"], train_df["y"], tokenizer, hw["max_length"])
    eval_ds = TextDataset(eval_df["texto_analisis"], None, tokenizer, hw["max_length"])
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=int(cfg["train_batch_size"]), shuffle=True,
        collate_fn=collator, generator=generator, num_workers=0,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=int(cfg["eval_batch_size"]), shuffle=False,
        collate_fn=collator, num_workers=0,
    )

    device = torch.device("cuda")
    model = make_model(cfg, device)
    optimizer = AdamW(model.parameters(), lr=float(cfg["learning_rate"]),
                      weight_decay=float(cfg["weight_decay"]))
    accum = int(cfg["gradient_accumulation_steps"])
    steps_per_epoch = math.ceil(len(train_loader) / accum)
    total_steps = steps_per_epoch * int(cfg["epochs"])
    warmup_steps = int(round(total_steps * float(cfg["warmup_ratio"])))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(hw["fp16"]))
    losses = []
    started = time.time()

    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(int(cfg["epochs"])):
        epoch_loss = 0.0
        for step, batch in enumerate(tqdm(train_loader, desc=f"final época {epoch+1}")):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=bool(hw["fp16"]), dtype=torch.float16):
                loss = model(**batch).loss / accum
            scaler.scale(loss).backward()
            epoch_loss += float(loss.detach().cpu()) * accum
            is_update = (step + 1) % accum == 0 or (step + 1) == len(train_loader)
            if is_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["max_grad_norm"]))
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        losses.append(epoch_loss / max(len(train_loader), 1))

    probs = predict(model, eval_loader, device, bool(hw["fp16"]))
    out = eval_df[["id_doc", "votos", "N_h_eval"]].copy()
    out["prob_si"] = probs
    out["pred_beto"] = (out["prob_si"] >= threshold).astype(int)
    out_path = ARTIFACTS / f"seed_{seed}" / "eval_predictions_blind.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    write_json(ARTIFACTS / f"seed_{seed}" / "final_training_log.json", {
        "seed": seed,
        "threshold": threshold,
        "n_train": len(train_df),
        "n_eval": len(eval_df),
        "epoch_losses": losses,
        "elapsed_seconds": round(time.time() - started, 2),
        "max_vram_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        "hardware": hw,
        "evaluation_labels_read": False,
    })
    print("Predicción ciega terminada:", out_path)
    print("Positivos predichos en muestra (sin ponderar):", int(out["pred_beto"].sum()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    cfg_ = load_config()
    main(args.seed if args.seed is not None else int(cfg_["primary_seed"]))
