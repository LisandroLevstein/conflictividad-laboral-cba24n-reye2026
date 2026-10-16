# -*- coding: utf-8 -*-
"""12 — Muestra de EVALUACIÓN (n = 1.400, doble codificación al 25 %).

Muestra estratificada por votos sobre la que se estiman todas las cifras
que el estudio reporta. El marco es el corpus completo menos la muestra de
ajuste (250 notas); no hay otras exclusiones. Las etiquetas de evaluación no
intervinieron en decisiones de especificación ni ajuste. Los métodos no
supervisados, sin embargo, se estimaron sobre el corpus completo antes de
seleccionar la muestra; por ello, la evaluación es independiente respecto de
las etiquetas, pero transductiva respecto de los textos.

TAMAÑO. n = 1.400 se deriva de un objetivo de precisión declarado antes
de codificar: que el semiancho del intervalo del 95 % para F1 sea menor
que la diferencia que el diseño busca resolver entre la regla
protocolar y el ensamble (del orden de 0,05).

ASIGNACIÓN. Se calcula en tiempo de ejecución a partir de los p_h de la
muestra de ajuste (script 08, t15b), siguiendo el orden riguroso: primero
se codifica el ajuste, luego se dimensiona la evaluación con esa
información. Dos criterios, por dos problemas distintos:

  * estrato 0 por COBERTURA (n0 = N0_COBERTURA). Su contribución a la
    varianza del total es la mayor (el estrato es el 80 % de la
    población), pero p_0 se estima cerca de cero y lo que hace falta es
    acotarla por arriba: con 630 lecturas y ningún positivo la cota de
    Clopper-Pearson (97,5 %) queda en 0,58 %.
  * estratos 1-4 por NEYMAN con corrección por población finita, sobre el
    presupuesto restante. Se usan los p_h SUAVIZADOS (Jeffreys) del
    ajuste, para que un estrato sin positivos observados en el ajuste no
    reciba asignación nula (t15b).

DOBLE CODIFICACIÓN. Una submuestra aleatoria del 25 %, estratificada,
se codifica por duplicado para estimar la fiabilidad (kappa) y adjudicar;
el resto una vez. Se marca en la columna `doble`.

Flujo de azar propio: default_rng([SEMILLA, 6]).
"""
import numpy as np
import pandas as pd

from config import DATA_PROCESSED, DATA_VALIDATION, SEMILLA, TABLAS

N_TOTAL = 1400
N0_COBERTURA = 630
FRACCION_DOBLE = 0.25


def asignacion_neyman(N_marco: np.ndarray) -> dict:
    """Neyman con fpc para los estratos 1-4, usando los p_h suavizados
    del ajuste. Devuelve el vector de tamaños (0..4)."""
    ph = pd.read_csv(TABLAS / "t15b_ph_ajuste.csv").set_index(
        "votos")["p_h_jeffreys"].to_dict()
    presupuesto = N_TOTAL - N0_COBERTURA
    Ns = N_marco[1:]
    ps = np.array([ph[h] for h in range(1, 5)])
    S = np.sqrt(ps * (1 - ps))

    def var(nh):
        fpc = np.maximum(1 - nh / Ns, 0)
        return np.sum((Ns ** 2) * ps * (1 - ps) * fpc / nh)

    nh = np.maximum(np.floor(Ns * S / (Ns * S).sum() * presupuesto), 40.0)
    for _ in range(800):
        if nh.sum() < presupuesto:
            g = [var(nh) - var(nh + np.eye(4)[i]) for i in range(4)]
            nh[int(np.argmax(g))] += 1
        elif nh.sum() > presupuesto:
            g = [var(nh - np.eye(4)[i]) - var(nh) for i in range(4)]
            nh[int(np.argmin(g))] -= 1
        else:
            break
    return {0: N0_COBERTURA, **{h + 1: int(nh[h]) for h in range(4)}}


def main() -> None:
    rng = np.random.default_rng([SEMILLA, 6])
    pred = pd.read_parquet(DATA_PROCESSED / "predicciones.parquet")
    corpus = pd.read_parquet(DATA_PROCESSED / "corpus_limpio.parquet")
    ajuste = pd.read_csv(DATA_VALIDATION / "muestra_ajuste_diseno.csv")
    fuera = set(ajuste["id_doc"])

    N_h = pred.groupby("votos")["id_doc"].count()
    marco = pred[~pred["id_doc"].isin(fuera)]
    N_marco = marco.groupby("votos")["id_doc"].count().sort_index()
    asig = asignacion_neyman(N_marco.to_numpy().astype(float))
    print("asignación (desde p_h del ajuste):", asig, "total", sum(asig.values()))

    filas = []
    for h, n in asig.items():
        pool = marco.loc[marco["votos"] == h, "id_doc"].to_numpy()
        filas.append(pd.DataFrame({
            "id_doc": rng.choice(pool, size=n, replace=False), "votos": h}))
    dis = pd.concat(filas, ignore_index=True)
    dis["N_h"] = dis["votos"].map(N_h)
    dis["N_h_eval"] = dis["votos"].map(N_marco)
    assert dis["id_doc"].is_unique and len(dis) == N_TOTAL
    assert not (set(dis["id_doc"]) & fuera)

    dobles = []
    for h, n in asig.items():
        ids = dis.loc[dis["votos"] == h, "id_doc"].to_numpy()
        dobles.append(rng.choice(ids, int(round(FRACCION_DOBLE * n)), False))
    dis["doble"] = dis["id_doc"].isin(set(np.concatenate(dobles)))
    dis.to_csv(DATA_VALIDATION / "muestra_evaluacion_diseno.csv", index=False)
    barajar = int(rng.integers(1e9))

    m = corpus[corpus["id_doc"].isin(dis["id_doc"])][
        ["id_doc", "fecha_dt", "titulo", "copete", "texto_analisis"]].copy()
    m = m.merge(dis[["id_doc", "doble"]], on="id_doc")
    m = m.sample(frac=1.0, random_state=barajar).reset_index(drop=True)
    m.insert(0, "orden", np.arange(1, len(m) + 1))
    for c in ["etiqueta", "justificacion"]:
        m[c] = ""
    cols = ["orden", "id_doc", "fecha_dt", "titulo", "copete",
            "texto_analisis", "doble", "etiqueta", "justificacion"]
    m[cols].to_csv(
        DATA_VALIDATION / "muestra_evaluacion_para_codificar.csv", index=False)
    m.loc[m["doble"], cols].to_csv(
        DATA_VALIDATION / "muestra_evaluacion_doble_para_codificar.csv",
        index=False)
    print("n_h:", dis.groupby("votos")["id_doc"].count().to_dict())
    print("doble:", int(dis["doble"].sum()),
          "| fracciones:", {h: round(asig[h] / N_marco[h], 3) for h in asig})


if __name__ == "__main__":
    main()
