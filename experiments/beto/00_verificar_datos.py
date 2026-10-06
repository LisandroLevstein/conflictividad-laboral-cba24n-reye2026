# -*- coding: utf-8 -*-
"""Audita insumos y registra el entorno sin entrenar modelos."""
from __future__ import annotations

import pandas as pd

from common import (
    ARTIFACTS,
    DATA_VALIDATION,
    ensure_dirs,
    environment_report,
    load_config,
    load_corpus,
    original_design_weights,
    validate_unique,
    write_json,
)


def main() -> None:
    ensure_dirs()
    cfg = load_config()
    corpus = load_corpus()
    ajuste = pd.read_csv(DATA_VALIDATION / "muestra_ajuste_etiquetada.csv")
    eval_dis = pd.read_csv(DATA_VALIDATION / "muestra_evaluacion_diseno.csv")

    validate_unique(corpus, "corpus")
    validate_unique(ajuste, "ajuste")
    validate_unique(eval_dis, "evaluacion_diseno")

    required_aj = {"id_doc", "etiqueta", "votos", "N_h"}
    required_ev = {"id_doc", "votos", "N_h_eval"}
    if not required_aj.issubset(ajuste.columns):
        raise ValueError(f"Faltan columnas en ajuste: {required_aj - set(ajuste.columns)}")
    if not required_ev.issubset(eval_dis.columns):
        raise ValueError(f"Faltan columnas en evaluación: {required_ev - set(eval_dis.columns)}")

    ajuste["etiqueta"] = ajuste["etiqueta"].astype(str).str.strip().str.upper()
    if not set(ajuste["etiqueta"]).issubset({"SI", "NO", "DUDOSO"}):
        raise ValueError("Etiquetas inesperadas en ajuste")
    if set(ajuste["id_doc"]) & set(eval_dis["id_doc"]):
        raise ValueError("Hay solapamiento entre ajuste y evaluación")

    aj = ajuste.merge(corpus, on="id_doc", how="left", validate="one_to_one")
    ev = eval_dis.merge(corpus, on="id_doc", how="left", validate="one_to_one")
    for name, df in (("ajuste", aj), ("evaluación", ev)):
        missing = df["texto_analisis"].isna() | df["texto_analisis"].astype(str).str.strip().eq("")
        if missing.any():
            raise ValueError(f"{name}: {int(missing.sum())} textos faltantes o vacíos")

    ajuste["w"] = original_design_weights(ajuste, "votos", "N_h")
    decidibles = ajuste[ajuste["etiqueta"].isin(["SI", "NO"])].copy()
    joint = decidibles["votos"].astype(str) + "_" + decidibles["etiqueta"]
    min_joint = int(joint.value_counts().min())
    if min_joint < int(cfg["n_folds"]):
        raise ValueError(
            f"La estratificación conjunta votos×etiqueta tiene una celda de {min_joint}, "
            f"menor que n_folds={cfg['n_folds']}"
        )

    report = {
        "config": cfg,
        "environment": environment_report(),
        "counts": {
            "corpus": int(len(corpus)),
            "ajuste_total": int(len(ajuste)),
            "ajuste_decidible": int(len(decidibles)),
            "evaluacion": int(len(eval_dis)),
            "ajuste_etiquetas": ajuste["etiqueta"].value_counts().to_dict(),
            "ajuste_por_estrato_etiqueta": (
                ajuste.groupby(["votos", "etiqueta"]).size().unstack(fill_value=0).to_dict()
            ),
        },
    }
    write_json(ARTIFACTS / "precheck.json", report)

    print("Verificación de insumos: CORRECTA")
    print(f"Corpus: {len(corpus):,}")
    print(f"Ajuste: {len(ajuste)} ({len(decidibles)} decidibles)")
    print(f"Evaluación ciega para entrenamiento: {len(eval_dis)}")
    print("Etiquetas de ajuste:", ajuste["etiqueta"].value_counts().to_dict())
    print("GPU:", report["environment"].get("gpu", "no disponible"))
    print("Registro:", ARTIFACTS / "precheck.json")


if __name__ == "__main__":
    main()
