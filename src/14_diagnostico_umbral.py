# -*- coding: utf-8 -*-
"""14 — Diagnóstico del umbral de frecuencia mínima (min_df) frente a la
clase rara.

El vectorizador descarta los términos presentes en menos de MIN_DF = 10
documentos. El umbral está definido sobre el CORPUS (10 de 26.855, el
0,04 %), mientras que el poder diagnóstico de un término es relativo a la
CLASE (10 de ~1.344 positivos sería el 0,7 %). Esa asimetría plantea un
riesgo legítimo: un término presente en pocos documentos, todos ellos
positivos, sería muy informativo y quedaría fuera del vocabulario.

Este script cuantifica el costo efectivo de esa decisión sobre este
corpus, con tres medidas:

  (a) qué proporción del VOCABULARIO y de la MASA DE TOKENS se descarta;
  (b) si esa masa descartada pesa más en los documentos POSITIVOS
      etiquetados que en los negativos —que es la dirección que haría
      del umbral un problema—;
  (c) cuántos términos descartados aparecen en dos o más positivos
      etiquetados, es decir, cuántos candidatos a marcador de clase
      perdió efectivamente el vocabulario.

No modifica el pipeline: es un diagnóstico. Bajar el umbral cambiaría el
vocabulario, y con él los modelos, los votos y por lo tanto los estratos
de las muestras ya codificadas; la evaluación quedaría invalidada. La
comparación entre umbrales queda planteada como trabajo futuro (§6.7).

Salida: results/tables/t20_umbral_minimo.csv
"""
import importlib.util

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from config import DATA_PROCESSED, DATA_VALIDATION, TABLAS

MIN_DF = 10
MAX_DF = 0.5


def cargar_analizador():
    spec = importlib.util.spec_from_file_location(
        "vec3", "03_vectorizacion.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    analizador, _ = mod.construir_analizador()
    return lambda t: [x for x in analizador(t) if " " not in x]


def main() -> None:
    uni = cargar_analizador()
    corp = pd.read_parquet(DATA_PROCESSED / "corpus_limpio.parquet")
    X = CountVectorizer(analyzer=uni, min_df=1).fit_transform(
        corp["texto_analisis"].fillna("").tolist())
    N = X.shape[0]
    df = np.asarray((X > 0).sum(0)).ravel()
    tot = np.asarray(X.sum(0)).ravel()
    raro = df < MIN_DF
    ret = (df >= MIN_DF) & (df <= MAX_DF * N)

    # (b) masa descartada en positivos vs negativos etiquetados
    aj = pd.read_csv(DATA_VALIDATION / "muestra_ajuste_etiquetada.csv")
    ev = pd.read_csv(DATA_VALIDATION / "muestra_evaluacion_etiquetada.csv")
    lab = pd.concat([aj[["id_doc", "etiqueta"]], ev[["id_doc", "etiqueta"]]])
    pos = set(lab.loc[lab["etiqueta"] == "SI", "id_doc"])
    neg = set(lab.loc[lab["etiqueta"] == "NO", "id_doc"])
    fila = {d: i for i, d in enumerate(corp["id_doc"])}
    ip = [fila[d] for d in pos if d in fila]
    ineg = [fila[d] for d in neg if d in fila]
    mp = np.asarray(X[ip].sum(0)).ravel()
    mn = np.asarray(X[ineg].sum(0)).ravel()

    # (c) descartados que aparecen en 2+ positivos etiquetados
    dfp = np.asarray((X[ip] > 0).sum(0)).ravel()
    cand = int(((raro) & (dfp >= 2)).sum())

    t = pd.DataFrame([
        {"medida": "vocabulario de unigramas sin umbral (min_df=1)",
         "valor": int(len(df))},
        {"medida": "unigramas descartados por min_df=10", "valor": int(raro.sum())},
        {"medida": "proporcion del vocabulario descartada",
         "valor": round(float(raro.mean()), 4)},
        {"medida": "unigramas retenidos (min_df=10, max_df=0,5)",
         "valor": int(ret.sum())},
        {"medida": "proporcion de la masa de tokens descartada (corpus)",
         "valor": round(float(tot[raro].sum() / tot.sum()), 4)},
        {"medida": "idem, en los positivos etiquetados",
         "valor": round(float(mp[raro].sum() / mp.sum()), 4)},
        {"medida": "idem, en los negativos etiquetados",
         "valor": round(float(mn[raro].sum() / mn.sum()), 4)},
        {"medida": "positivos etiquetados considerados", "valor": len(ip)},
        {"medida": "negativos etiquetados considerados", "valor": len(ineg)},
        {"medida": "descartados presentes en 2+ positivos etiquetados",
         "valor": cand},
    ])
    t.to_csv(TABLAS / "t20_umbral_minimo.csv", index=False)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
