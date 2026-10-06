# -*- coding: utf-8 -*-
"""07 — Muestra de AJUSTE (n = 250).

Muestra estratificada por votos del ensamble destinada exclusivamente a
las decisiones que exigen observar etiquetas: la elección del punto de
operación del ensamble (tabla t16) y el dimensionamiento de la muestra
de evaluación (script 12). NO interviene en ninguna estimación
reportada: la evaluación se realiza sobre una muestra disjunta.

MARCO: el corpus completo (26.855 notas), sin exclusiones.

ASIGNACIÓN n_h = (50, 40, 50, 60, 50) para votos 0..4. No es una
asignación de Neyman, y la razón es que esta muestra no estima: su
función es (i) seleccionar entre los cuatro cortes posibles del ensamble
y (ii) exponer al codificador a la frontera conceptual de la clase para
fijar las convenciones de codificación. Ambos objetivos se sirven
enriqueciendo los estratos donde la clase está genuinamente mezclada
—2, 3 y 4—, en lugar de concentrar la lectura en el estrato 0, que es el
80 % de la población pero rinde casi solo negativos. Con esta asignación
se esperan ~70 positivos, contra ~25 de una asignación proporcional a la
varianza de la prevalencia.

Flujo de azar: default_rng([SEMILLA, 5]).

Salidas:
  data/validation/muestra_ajuste_diseno.csv
  data/validation/muestra_ajuste_para_codificar.csv  (barajada, sin
      votos ni predicciones: insumo del etiquetado a ciegas)
"""
import numpy as np
import pandas as pd

from config import DATA_PROCESSED, DATA_VALIDATION, SEMILLA

ASIGNACION = {0: 50, 1: 40, 2: 50, 3: 60, 4: 50}


def main() -> None:
    rng = np.random.default_rng([SEMILLA, 5])
    pred = pd.read_parquet(DATA_PROCESSED / "predicciones.parquet")
    corpus = pd.read_parquet(DATA_PROCESSED / "corpus_limpio.parquet")
    N_h = pred.groupby("votos")["id_doc"].count()

    filas = []
    for h, n in ASIGNACION.items():
        pool = pred.loc[pred["votos"] == h, "id_doc"].to_numpy()
        filas.append(pd.DataFrame({
            "id_doc": rng.choice(pool, size=n, replace=False), "votos": h}))
    barajar = int(rng.integers(1e9))

    dis = pd.concat(filas, ignore_index=True)
    dis["N_h"] = dis["votos"].map(N_h)
    assert dis["id_doc"].is_unique and len(dis) == 250
    dis.to_csv(DATA_VALIDATION / "muestra_ajuste_diseno.csv", index=False)

    m = corpus[corpus["id_doc"].isin(dis["id_doc"])][
        ["id_doc", "fecha_dt", "titulo", "copete", "texto_analisis"]].copy()
    m = m.sample(frac=1.0, random_state=barajar).reset_index(drop=True)
    m.insert(0, "orden", np.arange(1, len(m) + 1))
    for c in ["etiqueta", "justificacion"]:
        m[c] = ""
    m.to_csv(DATA_VALIDATION / "muestra_ajuste_para_codificar.csv",
             index=False)
    print("n_h ajuste:", dis.groupby("votos")["id_doc"].count().to_dict())
    print("N_h:", N_h.to_dict(), "| total corpus:", int(N_h.sum()))


if __name__ == "__main__":
    main()
