# -*- coding: utf-8 -*-
"""01 — Limpieza y consolidación del corpus Cba24n 2016-2017.

Entrada : data/raw/cba24n_con_url_amigable_{2016,2017}.csv
Salida  : data/processed/corpus_limpio.parquet (+ .csv)
          results/tables/t02_resumen_limpieza.csv

Reglas (documentadas en la tesis, cap. Datos):
  (i)   se concatenan ambos años y se conserva una fila por nodo (`link`),
        porque las filas repetidas son variantes de URL amigable de una
        misma nota, con contenido editorial idéntico;
  (ii)  `texto_analisis` = `cuerpo`; si está vacío se usa `copete` y, en
        última instancia, `titulo` (decisión fijada por el protocolo);
  (iii) se normalizan restos no editoriales (URLs, handles y embebidos de
        Twitter) sin alterar el texto original, que se preserva;
  (iv)  los seis campos `tag1..tag6` se consolidan en una lista `tags`.
"""
import re
import unicodedata

import pandas as pd

from config import ARCHIVOS_CRUDOS, DATA_PROCESSED, TABLAS

RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_TW = re.compile(r"pic\.twitter\S*|@[A-Za-z0-9_]{2,}|#\w+")
RE_ESP = re.compile(r"\s+")


def limpiar_superficie(texto: str) -> str:
    """Remueve restos no editoriales (URLs, handles, hashtags) y normaliza
    espacios. No modifica mayúsculas ni tildes: es una limpieza de
    superficie previa a la normalización lingüística de 03."""
    if not isinstance(texto, str):
        return ""
    t = unicodedata.normalize("NFC", texto)
    t = RE_URL.sub(" ", t)
    t = RE_TW.sub(" ", t)
    t = RE_ESP.sub(" ", t)
    return t.strip()


def main() -> None:
    resumen = []
    frames = []
    for ruta in ARCHIVOS_CRUDOS:
        df = pd.read_csv(ruta, encoding="utf-8-sig")
        resumen.append({"paso": f"filas crudas {ruta.name}", "n": len(df)})
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    resumen.append({"paso": "filas concatenadas", "n": len(df)})

    # -- deduplicación por nodo -------------------------------------------
    # Verificación de la regla: dentro de un mismo `link` no puede haber
    # cuerpos distintos (si los hubiera, se aborta y se revisa a mano).
    nun = df.groupby("link")["cuerpo"].nunique(dropna=False)
    assert (nun <= 1).all(), "Un mismo nodo presenta cuerpos distintos."
    df = (df.sort_values(["link", "id"])
            .drop_duplicates(subset=["link"], keep="first")
            .reset_index(drop=True))
    resumen.append({"paso": "filas únicas por nodo (link)", "n": len(df)})

    # -- fecha --------------------------------------------------------------
    df["fecha_dt"] = pd.to_datetime(df["fecha"], format="%d-%m-%Y %H:%M:%S")
    assert df["fecha_dt"].notna().all()
    df["anio"] = df["fecha_dt"].dt.year
    df["mes"] = df["fecha_dt"].dt.to_period("M").astype(str)

    # -- texto de análisis ---------------------------------------------------
    for c in ["titulo", "copete", "cuerpo"]:
        df[c] = df[c].fillna("").astype(str).str.strip()
    df["cuerpo_limpio"] = df["cuerpo"].map(limpiar_superficie)
    df["copete_limpio"] = df["copete"].map(limpiar_superficie)
    df["titulo_limpio"] = df["titulo"].map(limpiar_superficie)

    df["fuente_texto"] = "cuerpo"
    vacio = df["cuerpo_limpio"].str.len() == 0
    df.loc[vacio, "fuente_texto"] = "copete"
    vacio2 = vacio & (df["copete_limpio"].str.len() == 0)
    df.loc[vacio2, "fuente_texto"] = "titulo"
    df["texto_analisis"] = df["cuerpo_limpio"]
    df.loc[df["fuente_texto"] == "copete", "texto_analisis"] = df["copete_limpio"]
    df.loc[df["fuente_texto"] == "titulo", "texto_analisis"] = df["titulo_limpio"]
    resumen.append({"paso": "docs con cuerpo vacío (usa copete)",
                    "n": int((df["fuente_texto"] == "copete").sum())})
    resumen.append({"paso": "docs con cuerpo y copete vacíos (usa título)",
                    "n": int((df["fuente_texto"] == "titulo").sum())})

    df["n_caracteres"] = df["texto_analisis"].str.len()
    df["n_palabras"] = df["texto_analisis"].str.split().str.len()

    # -- tags ---------------------------------------------------------------
    cols_tag = [f"tag{i}" for i in range(1, 7)]
    df["tags"] = (df[cols_tag]
                  .apply(lambda r: [str(x).strip() for x in r
                                    if pd.notna(x) and str(x).strip() != ""],
                         axis=1))
    df["tags_str"] = df["tags"].map(lambda xs: " | ".join(xs))

    # -- identificador estable ----------------------------------------------
    df["id_doc"] = df["link"].str.extract(r"node/(\d+)").astype(int)
    assert df["id_doc"].is_unique

    columnas = ["id_doc", "fecha_dt", "anio", "mes", "titulo", "copete",
                "cuerpo", "texto_analisis", "fuente_texto", "n_caracteres",
                "n_palabras", "tags", "tags_str", "link", "url_amigable"]
    df = df[columnas].sort_values("fecha_dt").reset_index(drop=True)

    df.to_parquet(DATA_PROCESSED / "corpus_limpio.parquet", index=False)
    df.drop(columns=["tags"]).to_csv(DATA_PROCESSED / "corpus_limpio.csv",
                                     index=False)
    pd.DataFrame(resumen).to_csv(TABLAS / "t02_resumen_limpieza.csv",
                                 index=False)
    print(pd.DataFrame(resumen).to_string(index=False))
    print("\nCorpus limpio:", len(df), "documentos ->",
          DATA_PROCESSED / "corpus_limpio.parquet")


if __name__ == "__main__":
    main()
