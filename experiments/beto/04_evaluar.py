# -*- coding: utf-8 -*-
"""Evalúa BETO sobre las 1.400 etiquetas y compara F1 de forma pareada."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import (
    ARTIFACTS,
    DATA_PROCESSED,
    DATA_VALIDATION,
    RESULTS,
    ensure_dirs,
    load_config,
    original_design_weights,
    read_json,
    weighted_metrics,
    write_json,
)


def main(seed: int) -> None:
    ensure_dirs()
    cfg = load_config()
    pred_path = ARTIFACTS / f"seed_{seed}" / "eval_predictions_blind.csv"
    threshold_path = ARTIFACTS / f"seed_{seed}" / "threshold.json"
    if not pred_path.exists() or not threshold_path.exists():
        raise FileNotFoundError("Faltan predicciones ciegas o threshold.json")

    beto = pd.read_csv(pred_path)
    labels = pd.read_csv(DATA_VALIDATION / "muestra_evaluacion_etiquetada.csv")
    design = pd.read_csv(DATA_VALIDATION / "muestra_evaluacion_diseno.csv")
    base = pd.read_parquet(DATA_PROCESSED / "predicciones.parquet", columns=[
        "id_doc", "m_lda", "m_nmf", "m_kmeans", "m_prototipo"
    ])
    labels["etiqueta"] = labels["etiqueta"].str.strip().str.upper()
    df = labels[["id_doc", "etiqueta"]].merge(
        design[["id_doc", "votos", "N_h_eval"]], on="id_doc", validate="one_to_one"
    ).merge(beto[["id_doc", "prob_si", "pred_beto"]], on="id_doc", validate="one_to_one"
    ).merge(base, on="id_doc", validate="one_to_one")
    if len(df) != 1400:
        raise RuntimeError(f"Se esperaban 1.400 casos, se obtuvieron {len(df)}")

    # El n_h debe calcularse sobre las 1.400 originales, antes de excluir DUDOSO.
    df["w"] = original_design_weights(df, "votos", "N_h_eval")
    df["y"] = (df["etiqueta"] == "SI").astype(float)
    df["domain"] = df["etiqueta"].isin(["SI", "NO"]).astype(float)
    df["ens_v2"] = (
        df[["m_lda", "m_nmf", "m_kmeans", "m_prototipo"]].sum(axis=1) >= 2
    ).astype(int)

    y = df["y"].to_numpy(float)
    w = df["w"].to_numpy(float)
    domain = df["domain"].to_numpy(float)
    c_beto = df["pred_beto"].to_numpy(float)
    c_ens = df["ens_v2"].to_numpy(float)
    classifiers = {"beto": c_beto, "ensamble_v2": c_ens}
    rng = np.random.default_rng(int(cfg["bootstrap_seed"]))
    groups = [np.where(df["votos"].to_numpy() == h)[0] for h in range(5)]
    replicas = [
        np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        for _ in range(int(cfg["bootstrap_replicates"]))
    ]

    rows = []
    f1_boot = {}
    for name, c in classifiers.items():
        precision, recall, f1 = weighted_metrics(y, w, c, domain)
        bs = np.asarray([
            weighted_metrics(y[ix], w[ix], c[ix], domain[ix])[2] for ix in replicas
        ])
        f1_boot[name] = bs
        rows.append({
            "seed": seed,
            "clasificador": name,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "f1_lo": round(float(np.percentile(bs, 2.5)), 4),
            "f1_hi": round(float(np.percentile(bs, 97.5)), 4),
            "n_muestra": len(df),
            "n_decidible": int(domain.sum()),
        })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS / f"metricas_seed_{seed}.csv", index=False)

    diff_point = weighted_metrics(y, w, c_beto, domain)[2] - weighted_metrics(y, w, c_ens, domain)[2]
    diffs = f1_boot["beto"] - f1_boot["ensamble_v2"]
    comparison = pd.DataFrame([{
        "seed": seed,
        "comparacion": "F1(BETO) - F1(ensamble >=2)",
        "diferencia_puntual": round(float(diff_point), 4),
        "ic95_lo": round(float(np.percentile(diffs, 2.5)), 4),
        "ic95_hi": round(float(np.percentile(diffs, 97.5)), 4),
        "prop_dif_positiva": round(float((diffs > 0).mean()), 4),
        "B": int(cfg["bootstrap_replicates"]),
    }])
    comparison.to_csv(RESULTS / f"comparacion_pareada_seed_{seed}.csv", index=False)

    sensitivity = []
    for scenario, y_s, dom_s in [
        ("principal (DUDOSO excluidos)", y, domain),
        ("DUDOSO como NO", y, np.ones_like(domain)),
        ("DUDOSO como SI", np.where(df["etiqueta"].eq("DUDOSO"), 1.0, y), np.ones_like(domain)),
    ]:
        for name, c in classifiers.items():
            p, r, f = weighted_metrics(y_s, w, c, dom_s)
            sensitivity.append({"seed": seed, "escenario": scenario, "clasificador": name,
                                "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)})
    pd.DataFrame(sensitivity).to_csv(RESULTS / f"sensibilidad_dudoso_seed_{seed}.csv", index=False)

    summary = {
        "seed": seed,
        "threshold": read_json(threshold_path),
        "metrics_file": f"results/metricas_seed_{seed}.csv",
        "paired_comparison_file": f"results/comparacion_pareada_seed_{seed}.csv",
    }
    write_json(RESULTS / f"resumen_seed_{seed}.json", summary)
    print("\nMétricas principales")
    print(metrics.to_string(index=False))
    print("\nComparación pareada")
    print(comparison.to_string(index=False))
    print("\nResultados guardados en:", RESULTS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    cfg_ = load_config()
    main(args.seed if args.seed is not None else int(cfg_["primary_seed"]))
