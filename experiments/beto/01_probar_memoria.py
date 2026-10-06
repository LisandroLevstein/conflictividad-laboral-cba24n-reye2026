# -*- coding: utf-8 -*-
"""Prueba un paso real de entrenamiento y fija longitud por restricción de VRAM."""
from __future__ import annotations

import gc
import traceback

import pandas as pd
import torch
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import (
    ARTIFACTS,
    DATA_VALIDATION,
    ensure_dirs,
    load_config,
    load_corpus,
    seed_everything,
    write_json,
)


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def attempt(text: str, max_length: int, fp16: bool, cfg: dict) -> dict:
    cleanup()
    seed_everything(int(cfg["primary_seed"]))
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name"], revision=cfg["model_revision"], use_fast=True
    )
    tokenizer.truncation_side = "right"
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"],
        revision=cfg["model_revision"],
        num_labels=2,
        use_safetensors=True,
    ).to("cuda")
    if cfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model.train()
    optimizer = AdamW(model.parameters(), lr=float(cfg["learning_rate"]))
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    encoded = {k: v.to("cuda") for k, v in encoded.items()}
    labels = torch.tensor([1], dtype=torch.long, device="cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=fp16)
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=fp16, dtype=torch.float16):
        loss = model(**encoded, labels=labels).loss
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["max_grad_norm"]))
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()
    out = {
        "success": True,
        "max_length": int(max_length),
        "fp16": bool(fp16),
        "observed_tokens": int(encoded["input_ids"].shape[1]),
        "loss": float(loss.detach().cpu()),
        "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
    }
    del model, optimizer, encoded, labels, scaler, tokenizer
    cleanup()
    return out


def main() -> None:
    ensure_dirs()
    if not torch.cuda.is_available():
        raise RuntimeError("Esta prueba requiere CUDA")
    cfg = load_config()
    ajuste = pd.read_csv(DATA_VALIDATION / "muestra_ajuste_etiquetada.csv")
    corpus = load_corpus()
    df = ajuste[["id_doc"]].merge(corpus, on="id_doc", validate="one_to_one")

    # Elegimos el texto de ajuste con más caracteres: no se accede a evaluación.
    text = df.loc[df["texto_analisis"].astype(str).str.len().idxmax(), "texto_analisis"]
    attempts = []
    candidates = [(int(x), bool(cfg["fp16"])) for x in cfg["candidate_max_lengths"]]
    if bool(cfg["fp16"]):
        candidates.append((int(cfg["candidate_max_lengths"][-1]), False))

    selected = None
    for max_length, fp16 in candidates:
        print(f"Probando max_length={max_length}, fp16={fp16}...")
        try:
            result = attempt(str(text), max_length, fp16, cfg)
            attempts.append(result)
            selected = result
            print("  correcta; VRAM pico:", result["peak_vram_mib"], "MiB")
            break
        except torch.cuda.OutOfMemoryError as exc:
            attempts.append({
                "success": False,
                "max_length": max_length,
                "fp16": fp16,
                "error": "CUDA out of memory",
            })
            print("  no entra en VRAM")
            cleanup()
        except Exception as exc:
            attempts.append({
                "success": False,
                "max_length": max_length,
                "fp16": fp16,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            })
            print("  falló:", repr(exc))
            cleanup()

    if selected is None:
        write_json(ARTIFACTS / "hardware_attempts.json", {"attempts": attempts})
        raise RuntimeError("Ninguna configuración de memoria resultó viable")

    hardware = {
        "max_length": selected["max_length"],
        "fp16": selected["fp16"],
        "peak_vram_mib_probe": selected["peak_vram_mib"],
        "probe_observed_tokens": selected["observed_tokens"],
        "selection_rule": "primera configuración viable en orden 512, 384, 256; fp16 primero",
        "attempts": attempts,
    }
    write_json(ARTIFACTS / "hardware.json", hardware)
    print("Configuración fijada por hardware:", hardware)
    print("Registro:", ARTIFACTS / "hardware.json")


if __name__ == "__main__":
    main()
