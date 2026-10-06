# -*- coding: utf-8 -*-
"""Consolida las métricas finales de la semilla primaria y las sensibilidades."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


MODULE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = MODULE_DIR / "results"
CONFIG_PATH = MODULE_DIR / "config.json"


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    primary_seed = int(config["primary_seed"])
    expected_seeds = [
        primary_seed,
        *[int(seed) for seed in config["sensitivity_seeds"]],
    ]

    rows = []

    for seed in expected_seeds:
        summary_path = RESULTS_DIR / f"resumen_seed_{seed}.json"

        if not summary_path.exists():
            raise FileNotFoundError(f"Falta el archivo {summary_path}")

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        threshold = summary["threshold"]

        metrics_path = RESULTS_DIR / Path(summary["metrics_file"]).name
        comparison_path = (
            RESULTS_DIR / Path(summary["paired_comparison_file"]).name
        )

        metrics = pd.read_csv(metrics_path)
        comparison = pd.read_csv(comparison_path)

        beto_rows = metrics.loc[metrics["clasificador"].eq("beto")]
        ensamble_rows = metrics.loc[
            metrics["clasificador"].eq("ensamble_v2")
        ]

        if len(beto_rows) != 1 or len(ensamble_rows) != 1:
            raise RuntimeError(
                f"Métricas incompletas o duplicadas para la semilla {seed}"
            )

        if len(comparison) != 1:
            raise RuntimeError(
                f"Comparación pareada inválida para la semilla {seed}"
            )

        beto = beto_rows.iloc[0]
        ensamble = ensamble_rows.iloc[0]
        paired = comparison.iloc[0]

        rows.append({
            "seed": seed,
            "corrida": "primaria" if seed == primary_seed else "sensibilidad",
            "umbral": threshold["threshold"],
            "f1_oof_ajuste": threshold["f1_oof_weighted"],
            "precision_eval": beto["precision"],
            "recall_eval": beto["recall"],
            "f1_beto": beto["f1"],
            "f1_beto_lo": beto["f1_lo"],
            "f1_beto_hi": beto["f1_hi"],
            "f1_ensamble": ensamble["f1"],
            "delta_f1": paired["diferencia_puntual"],
            "delta_f1_lo": paired["ic95_lo"],
            "delta_f1_hi": paired["ic95_hi"],
            "prop_delta_positiva": paired["prop_dif_positiva"],
            "ic95_delta_excluye_cero": bool(paired["ic95_lo"] > 0),
        })

    table = pd.DataFrame(rows)
    output_path = RESULTS_DIR / "resumen_semillas.csv"
    table.to_csv(output_path, index=False)

    print(table.to_string(index=False))
    print()
    print(
        "F1 BETO medio descriptivo:",
        round(float(table["f1_beto"].mean()), 4),
    )
    print(
        "Rango F1 BETO:",
        f"[{table['f1_beto'].min():.4f}; "
        f"{table['f1_beto'].max():.4f}]",
    )
    print(
        "Delta F1 medio descriptivo:",
        round(float(table["delta_f1"].mean()), 4),
    )
    print("Resultado guardado en:", output_path)


if __name__ == "__main__":
    main()
