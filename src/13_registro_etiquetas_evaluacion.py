# -*- coding: utf-8 -*-
"""13 — Registro de la doble codificación de la muestra de EVALUACIÓN.

Las 1.400 notas fueron codificadas ÍNTEGRAMENTE por dos codificadores de
forma independiente —a ciegas de las predicciones, de los estratos y de
la codificación del otro— y un tercero adjudicó los 79 desacuerdos. Este
script lee el registro, calcula el acuerdo y el kappa de Cohen (t17) y
emite la etiqueta final, único insumo de la inferencia (script 09).

La doble codificación es completa y no parcial: el diseño preveía una
submuestra de fiabilidad del 25 %, y la disponibilidad de dos
codificadores para el total permitió elevarla a la muestra entera. En
consecuencia, la fiabilidad no se estima sobre una submuestra sino que se
mide sobre todos los casos que sostienen las estimaciones, y la
adjudicación cubre la totalidad de los desacuerdos.

Salidas: data/validation/muestra_evaluacion_etiquetada.csv
         results/tables/t17_acuerdo.csv
"""
import numpy as np
import pandas as pd

from config import DATA_VALIDATION, TABLAS


def kappa(a, b):
    cats = sorted(set(a) | set(b))
    po = float((a == b).mean())
    pe = sum(float((a == c).mean()) * float((b == c).mean()) for c in cats)
    return po, (po - pe) / (1 - pe)


def main() -> None:
    cod = pd.read_csv(DATA_VALIDATION / "muestra_evaluacion_codificacion.csv")
    base = pd.read_csv(DATA_VALIDATION /
                       "muestra_evaluacion_para_codificar.csv")
    for c in ["etiqueta_cod1", "etiqueta_cod2", "etiqueta"]:
        cod[c] = cod[c].str.strip().str.upper()
    assert set(cod["etiqueta"]).issubset({"SI", "NO", "DUDOSO"})
    m = base.drop(columns=["etiqueta", "justificacion"], errors="ignore").merge(
        cod, on="id_doc")
    assert len(m) == 1400 and m["id_doc"].is_unique
    m.to_csv(DATA_VALIDATION / "muestra_evaluacion_etiquetada.csv", index=False)

    po3, k3 = kappa(m["etiqueta_cod1"], m["etiqueta_cod2"])
    a2 = (m["etiqueta_cod1"] == "SI").map({True: "SI", False: "noSI"})
    b2 = (m["etiqueta_cod2"] == "SI").map({True: "SI", False: "noSI"})
    po2, k2 = kappa(a2, b2)
    t17 = pd.DataFrame([
        {"comparacion": "etiqueta (SI/NO/DUDOSO)", "n": len(m),
         "acuerdo": round(po3, 4), "kappa": round(k3, 4)},
        {"comparacion": "dicotomica (SI vs no-SI)", "n": len(m),
         "acuerdo": round(po2, 4), "kappa": round(k2, 4)}])
    t17.to_csv(TABLAS / "t17_acuerdo.csv", index=False)
    print(t17.to_string(index=False))
    print("desacuerdos adjudicados:", int(m["adjudicado"].sum()))
    print("etiqueta final:", m["etiqueta"].value_counts().to_dict())
    print("\ncruce cod1 x cod2:")
    print(pd.crosstab(m["etiqueta_cod1"], m["etiqueta_cod2"]))


if __name__ == "__main__":
    main()
