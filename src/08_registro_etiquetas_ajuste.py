# -*- coding: utf-8 -*-
"""08 — Registro de la doble codificación de la muestra de AJUSTE (250).

La muestra de ajuste fue codificada por DOS codificadores de forma
independiente; un tercer codificador adjudicó los desacuerdos. Este
script lee el registro (data/validation/muestra_ajuste_codificacion.csv,
con ambas codificaciones y la etiqueta definitiva) y:

  1. reporta el acuerdo inter-codificador y el kappa de Cohen (t21);
  2. fija el punto de operación del ensamble sobre la etiqueta definitiva
     (tabla t16), única decisión de modelo tomada mirando etiquetas;
  3. calcula los p_h suavizados que el script 12 usa para asignar la
     muestra de evaluación por Neyman.

Los p_h se suavizan con la regla de Jeffreys, (SI_h + 0,5)/(n_h + 1),
para evitar el cero estructural de los estratos de baja frecuencia: con
n_h del orden de 40-50, un estrato puede no exhibir ningún positivo en el
ajuste aunque su proporción poblacional sea no nula, y un p_h = 0 anularía
su asignación de Neyman. El suavizado da un piso positivo sin alterar de
forma apreciable los estratos con señal.

Salidas: data/validation/muestra_ajuste_etiquetada.csv,
         results/tables/{t16_ajuste_corte.csv, t21_acuerdo_ajuste.csv}
"""
import numpy as np
import pandas as pd

from config import DATA_PROCESSED, DATA_VALIDATION, TABLAS


def kappa(a, b):
    cats = sorted(set(a) | set(b))
    po = float((a == b).mean())
    pe = sum(float((a == c).mean()) * float((b == c).mean()) for c in cats)
    return po, (po - pe) / (1 - pe)


def main() -> None:
    cod = pd.read_csv(DATA_VALIDATION / "muestra_ajuste_codificacion.csv")
    dis = pd.read_csv(DATA_VALIDATION / "muestra_ajuste_diseno.csv")
    pred = pd.read_parquet(DATA_PROCESSED / "predicciones.parquet")
    for c in ["etiqueta_cod1", "etiqueta_cod2", "etiqueta"]:
        cod[c] = cod[c].str.strip().str.upper()
    assert set(cod["etiqueta"]).issubset({"SI", "NO", "DUDOSO"})

    d = cod.merge(dis[["id_doc", "votos", "N_h"]], on="id_doc").merge(
        pred[["id_doc", "m_lda", "m_nmf", "m_kmeans", "m_prototipo"]],
        on="id_doc")
    d.to_csv(DATA_VALIDATION / "muestra_ajuste_etiquetada.csv", index=False)

    # -------------------------------------------- acuerdo (t21) ----
    po3, k3 = kappa(cod["etiqueta_cod1"], cod["etiqueta_cod2"])
    a2 = (cod["etiqueta_cod1"] == "SI").map({True: "SI", False: "noSI"})
    b2 = (cod["etiqueta_cod2"] == "SI").map({True: "SI", False: "noSI"})
    po2, k2 = kappa(a2, b2)
    t21 = pd.DataFrame([
        {"comparacion": "etiqueta (SI/NO/DUDOSO)", "n": len(cod),
         "acuerdo": round(po3, 3), "kappa": round(k3, 3)},
        {"comparacion": "dicotomica (SI vs no-SI)", "n": len(cod),
         "acuerdo": round(po2, 3), "kappa": round(k2, 3)}])
    t21.to_csv(TABLAS / "t21_acuerdo_ajuste.csv", index=False)
    print(t21.to_string(index=False))
    print("desacuerdos:", int((cod["etiqueta_cod1"] != cod["etiqueta_cod2"]).sum()))
    print("distribucion final:", cod["etiqueta"].value_counts().to_dict())

    # -------------------------------------- decisión del corte (t16)
    n_h = d.groupby("votos")["id_doc"].count()
    d["w"] = d["N_h"] / d["votos"].map(n_h)
    dec = d[d["etiqueta"] != "DUDOSO"].copy()
    y = (dec["etiqueta"] == "SI").astype(float)
    dec["v"] = dec[["m_lda", "m_nmf", "m_kmeans", "m_prototipo"]].sum(axis=1)
    filas = []
    for v in [1, 2, 3, 4]:
        c = (dec["v"] >= v).astype(float)
        vp = float((dec["w"] * y * c).sum())
        p = vp / float((dec["w"] * c).sum())
        r = vp / float((dec["w"] * y).sum())
        filas.append({"corte": f"votos >= {v}", "precision": round(p, 3),
                      "recall": round(r, 3),
                      "f1": round(2 * p * r / (p + r), 3)})
    t16 = pd.DataFrame(filas)
    t16.to_csv(TABLAS / "t16_ajuste_corte.csv", index=False)
    corte = int(t16.loc[t16["f1"].idxmax(), "corte"].split(">=")[1])
    print("\n" + t16.to_string(index=False))
    print(f"corte elegido: >= {corte}")

    # -------------------------------------- p_h suavizados (Jeffreys)
    si_h = dec.assign(si=(dec["etiqueta"] == "SI")).groupby(
        "votos")["si"].sum().reindex(range(5), fill_value=0)
    tot_h = d.groupby("votos")["id_doc"].count().reindex(range(5))
    p_j = (si_h + 0.5) / (tot_h + 1)
    pd.DataFrame({"votos": range(5), "n_h": tot_h.values,
                  "si_h": si_h.values, "p_h_jeffreys": p_j.round(4).values}
                 ).to_csv(TABLAS / "t15b_ph_ajuste.csv", index=False)
    print("p_h (Jeffreys):", {h: round(v, 4) for h, v in p_j.items()})


if __name__ == "__main__":
    main()
