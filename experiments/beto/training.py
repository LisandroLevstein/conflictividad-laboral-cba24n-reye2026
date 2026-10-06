# -*- coding: utf-8 -*-
"""Componentes mínimos de entrenamiento compartidos por CV y modelo final."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.enc = tokenizer(
            list(texts), truncation=True, max_length=max_length, padding=False
        )
        self.labels = list(labels) if labels is not None else None

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        if self.labels is not None:
            item["labels"] = int(self.labels[i])
        return item


def make_model(cfg, device):
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"],
        revision=cfg["model_revision"],
        num_labels=2,
        use_safetensors=True,
    ).to(device)
    if cfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    return model


def predict(model, loader, device, fp16):
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            batch.pop("labels", None)
            with torch.cuda.amp.autocast(enabled=fp16, dtype=torch.float16):
                logits = model(**batch).logits
            probs.extend(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
    return np.asarray(probs)
