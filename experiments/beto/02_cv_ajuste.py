# -*- coding: utf-8 -*-
"""Genera probabilidades out-of-fold y fija el umbral con la muestra de ajuste."""
from __future__ import annotations

import argparse
import gc
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from common import (
    ARTIFACTS,
    DATA_VALIDATION,
    ensure_dirs,
    hardware_settings,
    load_config,
    load_corpus,
    original_design_weights,
    seed_everything,
    threshold_grid,
    weighted_metrics,
    write_json,
)


from training import TextDataset, make_model, predict

def train_fold(train_df, val_df, tokenizer, cfg, hw, seed, fold):
    seed_everything(seed + fold)
    device = torch.device("cuda")
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    train_ds = TextDataset(train_df["texto_analisis"], train_df["y"], tokenizer, hw["max_length"])
    val_ds = TextDataset(val_df["texto_analisis"], val_df["y"], tokenizer, hw["max_length"])
    generator = torch.Generator().manual_seed(seed + fold)
    train_loader = DataLoader(
        train_ds, batch_size=int(cfg["train_batch_size"]), shuffle=True,
        collate_fn=collator, generator=generator, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(cfg["eval_batch_size"]), shuffle=False,
        collate_fn=collator, num_workers=0,
    )
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
        for step, batch in enumerate(tqdm(train_loader, desc=f"fold {fold} época {epoch+1}", leave=False)):
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

    probs = predict(model, val_loader, device, bool(hw["fp16"]))
    log = {
        "fold": fold,
        "seed": seed,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "epoch_losses": losses,
        "elapsed_seconds": round(time.time() - started, 2),
        "max_vram_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
    }
    del model, optimizer, scheduler, scaler, train_loader, val_loader, train_ds, val_ds
    gc.collect()
    torch.cuda.empty_cache()
    return probs, log


def main(seed: int) -> None:
    ensure_dirs()
    if not torch.cuda.is_available():
        raise RuntimeError("Se requiere CUDA")
    cfg = load_config()
    hw = hardware_settings(cfg)
    seed_everything(seed)

    ajuste_all = pd.read_csv(DATA_VALIDATION / "muestra_ajuste_etiquetada.csv")
    ajuste_all["etiqueta"] = ajuste_all["etiqueta"].str.strip().str.upper()
    ajuste_all["w"] = original_design_weights(ajuste_all, "votos", "N_h")
    corpus = load_corpus()
    df = ajuste_all[ajuste_all["etiqueta"].isin(["SI", "NO"])].copy()
    df["y"] = (df["etiqueta"] == "SI").astype(int)
    df = df.merge(corpus, on="id_doc", how="left", validate="one_to_one")
    df = df.sort_values("id_doc").reset_index(drop=True)
    strat = df["votos"].astype(str) + "_" + df["etiqueta"]

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name"], revision=cfg["model_revision"], use_fast=True
    )
    tokenizer.truncation_side = "right"
    skf = StratifiedKFold(n_splits=int(cfg["n_folds"]), shuffle=True, random_state=seed)
    out_dir = ARTIFACTS / f"seed_{seed}" / "folds"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_logs = []
    oof_parts = []

    for fold, (tr, va) in enumerate(skf.split(df, strat), start=1):
        fold_path = out_dir / f"fold_{fold}_pred.csv"
        log_path = out_dir / f"fold_{fold}_log.json"
        expected_ids = set(df.iloc[va]["id_doc"])
        if fold_path.exists() and log_path.exists():
            old = pd.read_csv(fold_path)
            if set(old["id_doc"]) == expected_ids:
                print(f"Fold {fold}: reutilizando salida existente")
                oof_parts.append(old)
                from common import read_json
                all_logs.append(read_json(log_path))
                continue
        print(f"Fold {fold}/{cfg['n_folds']}: entrenamiento")
        probs, log = train_fold(df.iloc[tr], df.iloc[va], tokenizer, cfg, hw, seed, fold)
        part = df.iloc[va][["id_doc", "votos", "N_h", "w", "etiqueta", "y"]].copy()
        part["fold"] = fold
        part["prob_si"] = probs
        part.to_csv(fold_path, index=False)
        write_json(log_path, log)
        oof_parts.append(part)
        all_logs.append(log)

    oof = pd.concat(oof_parts, ignore_index=True).sort_values("id_doc")
    if len(oof) != len(df) or oof["id_doc"].duplicated().any():
        raise RuntimeError("Las predicciones OOF no cubren exactamente los 238 casos decidibles")
    oof_path = ARTIFACTS / f"seed_{seed}" / "oof_predictions.csv"
    oof.to_csv(oof_path, index=False)

    rows = []
    y = oof["y"].to_numpy(float)
    w = oof["w"].to_numpy(float)
    domain = np.ones(len(oof), dtype=float)
    for threshold in threshold_grid(cfg):
        pred = (oof["prob_si"].to_numpy() >= threshold).astype(float)
        precision, recall, f1 = weighted_metrics(y, w, pred, domain)
        rows.append({"threshold": threshold, "precision": precision, "recall": recall, "f1": f1})
    grid = pd.DataFrame(rows)
    best_f1 = grid["f1"].max()
    tied = grid[np.isclose(grid["f1"], best_f1, rtol=0, atol=1e-12)].copy()
    tied["distance_05"] = (tied["threshold"] - 0.5).abs()
    best = tied.sort_values(["distance_05", "threshold"]).iloc[0]
    threshold_path = ARTIFACTS / f"seed_{seed}" / "threshold.json"
    threshold_record = {
        "seed": seed,
        "threshold": float(best["threshold"]),
        "precision_oof_weighted": float(best["precision"]),
        "recall_oof_weighted": float(best["recall"]),
        "f1_oof_weighted": float(best["f1"]),
        "tie_break": "umbral más cercano a 0.5; luego el menor",
        "weights": "N_h/n_h con n_h original antes de excluir DUDOSO",
        "n_decidable": int(len(oof)),
        "hardware": hw,
        "training_logs": all_logs,
    }
    grid.to_csv(ARTIFACTS / f"seed_{seed}" / "threshold_grid.csv", index=False)
    write_json(threshold_path, threshold_record)
    print("CV terminada")
    print("Umbral fijado:", threshold_record)
    print("OOF:", oof_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    cfg_ = load_config()
    main(args.seed if args.seed is not None else int(cfg_["primary_seed"]))
