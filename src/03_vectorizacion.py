# -*- coding: utf-8 -*-
"""03 — Representación vectorial del corpus.

Pipeline lingüístico:
  1. minúsculas y remoción de diacríticos;
  2. tokenización por expresión regular ([a-z]{2,});
  3. remoción de stopwords del español (NLTK) más una lista propia de
     términos de medio/boilerplate;
  4. radicalización (stemmer Snowball para español);
  5. matriz TF-IDF con unigramas y bigramas, tf sublineal,
     min_df=10, max_df=0.5;
  6. LSA: SVD truncada a 300 dimensiones + normalización L2.

Salidas: data/interim/{X_tfidf.npz, X_lsa.npy, X_counts.npz,
         vocabulario.json, vocabulario_counts.json, svd.joblib,
         mapa_stem.json}
"""
import json
import re
import unicodedata

import joblib
import numpy as np
import pandas as pd
from nltk.stem import SnowballStemmer

from config import DATA_EXTERNAL


def cargar_stopwords_es() -> list:
    """Stopwords del español desde archivo versionado (reproducible sin
    red). Si el archivo no existe, cae a NLTK y lo escribe para futuras
    corridas; así el pipeline no depende de una descarga externa."""
    ruta = DATA_EXTERNAL / "stopwords_es_nltk.txt"
    if ruta.exists():
        return [ln.strip() for ln in ruta.read_text(encoding="utf-8")
                .splitlines() if ln.strip() and not ln.startswith("#")]
    from nltk.corpus import stopwords  # noqa: fallback con red
    sw = sorted(stopwords.words("spanish"))
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text("# Stopwords del español (NLTK). Versionado.\n"
                    + "\n".join(sw) + "\n", encoding="utf-8")
    return sw
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

from config import DATA_INTERIM, DATA_PROCESSED, MAX_DF, MIN_DF, \
    N_COMPONENTES_LSA, SEMILLA

TOKEN = re.compile(r"[a-z]{2,}")
STEMMER = SnowballStemmer("spanish")

# stopwords adicionales: boilerplate del medio, días, meses y verbos de
# atribución periodística que no discriminan contenido.
STOP_EXTRA = {
    "cba", "cba24n", "canal", "srt", "servicios", "radio", "universidad",
    "lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo",
    "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
    "septiembre", "setiembre", "octubre", "noviembre", "diciembre",
    "ayer", "hoy", "manana", "ademas", "tambien", "segun", "tras", "luego",
    "dijo", "afirmo", "aseguro", "indico", "senalo", "explico", "agrego",
    "sostuvo", "expreso", "manifesto", "informo", "confirmo", "anticipo",
    "detallo", "preciso", "apunto", "remarco", "destaco", "advirtio",
    "mientras", "aunque", "embargo", "asimismo", "respecto", "cabe",
    "horas", "hora", "anos", "ano", "dia", "dias", "vez", "veces",
}


def normalizar(texto: str) -> str:
    t = unicodedata.normalize("NFKD", str(texto).lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def construir_analizador():
    sw = {normalizar(w) for w in cargar_stopwords_es()} | STOP_EXTRA
    cache = {}

    def stem(tok: str) -> str:
        if tok not in cache:
            cache[tok] = STEMMER.stem(tok)
        return cache[tok]

    def analizador(texto: str):
        toks = [t for t in TOKEN.findall(normalizar(texto)) if t not in sw]
        stems = [stem(t) for t in toks]
        # unigramas + bigramas contiguos
        return stems + [f"{a} {b}" for a, b in zip(stems, stems[1:])]

    return analizador, stem


def main() -> None:
    df = pd.read_parquet(DATA_PROCESSED / "corpus_limpio.parquet")
    textos = df["texto_analisis"].tolist()
    analizador, stem = construir_analizador()

    # ------------------------------------------------------------- TF-IDF -
    vec = TfidfVectorizer(analyzer=analizador, min_df=MIN_DF, max_df=MAX_DF,
                          sublinear_tf=True, norm="l2")
    X = vec.fit_transform(textos)
    sparse.save_npz(DATA_INTERIM / "X_tfidf.npz", X)
    vocab = vec.get_feature_names_out().tolist()
    json.dump(vocab, open(DATA_INTERIM / "vocabulario.json", "w"))

    # ------------------------------------------- conteos (para LDA y NPMI) -
    vec_c = CountVectorizer(analyzer=analizador, min_df=MIN_DF, max_df=MAX_DF)
    Xc = vec_c.fit_transform(textos)
    sparse.save_npz(DATA_INTERIM / "X_counts.npz", Xc)
    json.dump(vec_c.get_feature_names_out().tolist(),
              open(DATA_INTERIM / "vocabulario_counts.json", "w"))

    # ---------------------------------------------------------------- LSA -
    svd = TruncatedSVD(n_components=N_COMPONENTES_LSA, random_state=SEMILLA)
    X_lsa = svd.fit_transform(X)
    X_lsa = normalize(X_lsa)
    np.save(DATA_INTERIM / "X_lsa.npy", X_lsa.astype(np.float32))
    joblib.dump(svd, DATA_INTERIM / "svd.joblib")

    # mapa término_original -> stem para anclar el lexicón en el vocabulario
    mapa = {}
    for w in ["paro", "paros", "huelga", "huelgas", "movilizacion",
              "movilizaciones", "asamblea", "asambleas", "piquete",
              "piquetes", "acampe", "bloqueo", "toma", "ocupacion",
              "protesta", "protestas", "manifestacion", "reclamo",
              "reclamos", "alerta", "conciliacion", "paritaria",
              "paritarias", "gremio", "gremios", "gremial", "sindicato",
              "sindicatos", "sindical", "trabajadores", "trabajador",
              "obreros", "empleados", "delegados", "docentes", "choferes",
              "estatales", "municipales", "salario", "salarios", "salarial",
              "sueldo", "sueldos", "despido", "despidos", "despedidos",
              "cesantias", "suspensiones", "reincorporacion", "haberes",
              "aumento", "recomposicion", "convenio", "cgt", "cta", "ate",
              "uta", "uepc", "suoem", "surrbac", "uom", "smata", "uocra",
              "bancarios", "judiciales", "aguinaldo", "medida", "fuerza",
              "lucha", "marcha", "movilizaron", "reclamaron", "pararon"]:
        mapa[w] = stem(w)
    json.dump(mapa, open(DATA_INTERIM / "mapa_stem.json", "w"),
              ensure_ascii=False, indent=1)

    print(f"TF-IDF: {X.shape[0]} docs x {X.shape[1]} términos "
          f"(densidad {X.nnz / (X.shape[0]*X.shape[1]):.4%})")
    print(f"Conteos: {Xc.shape} | LSA: {X_lsa.shape}, "
          f"varianza explicada = {svd.explained_variance_ratio_.sum():.3f}")


if __name__ == "__main__":
    main()
