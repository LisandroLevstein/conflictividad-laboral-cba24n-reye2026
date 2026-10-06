# -*- coding: utf-8 -*-
"""04 — Modelado de tópicos (LDA y NMF), selección de K y anclaje.

Uso (permite ejecutar la grilla por partes):
    python 04_topicos.py nmf-grid
    python 04_topicos.py lda-grid --ks 20 30 40
    python 04_topicos.py lda-grid --ks 50 60
    python 04_topicos.py finalizar

Selección de K: coherencia NPMI media de los 10 términos más probables
de cada tópico, con co-ocurrencias a nivel documento sobre el propio
corpus (Bouma, 2009; Röder et al., 2015). LDA se ajusta sobre conteos
de unigramas; NMF sobre TF-IDF (uni+bigramas).

Anclaje protocolar de tópicos: un tópico se marca como "de conflicto
laboral" si entre sus 20 términos principales aparece al menos un radical
de ACCIÓN y al menos un radical de PROTAGONISTA u OBJETO del lexicón.

Clasificación por documento: SÍ si su tópico dominante está anclado.
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import NMF, LatentDirichletAllocation

from config import DATA_INTERIM, GRILLA_K_TOPICOS, RESULTS, SEMILLA, TABLAS

# Radicales (Snowball) de anclaje, derivados del lexicón del protocolo.
# NOTA DE IMPLEMENTACIÓN. El protocolo se operacionaliza en DOS piezas
# distintas, por razones técnicas:
#   (a) data/lexicon/lexicon.json — expresiones regulares sobre el TEXTO;
#       gobierna las reglas protocolares (amplia/estricta), las semillas
#       del prototipo y las señales por documento (script 02);
#   (b) los conjuntos ANCLA_* de abajo — RADICALES sobre el vocabulario
#       radicalizado del modelo; gobiernan el anclaje de tópicos.
# La pieza (b) es necesaria porque un tópico no expone texto corrido sino
# una lista de radicales, sobre la que las regex de (a) no son aplicables.
# Los radicales de (b) son las raíces Snowball de los términos nucleares
# de las familias ACCION (para ANCLA_ACCION) y PROTAGONISTA/OBJETO (para
# ANCLA_ACTOR_OBJETO) de (a), restringidos a los de uso más frecuente.
# No son, por tanto, el lexicón completo: son un subconjunto derivado.
# Toda modificación de (a) exige revisar (b) y viceversa.
ANCLA_ACCION = {"par", "huelg", "moviliz", "asamble", "piquet", "acamp",
                "protest", "manifest", "reclam", "alert", "marcha", "march",
                "bloque", "paralizacion", "concili"}
ANCLA_ACTOR_OBJETO = {"gremi", "gremial", "sindic", "sindical", "trabaj",
                      "obrer", "emplead", "deleg", "paritari", "salarial",
                      "salari", "sueld", "haber", "despid", "cesant",
                      "suspension", "reincorpor", "cgt", "cta", "uta",
                      "uepc", "suoem", "surrbac", "ate", "uom", "smata",
                      "aguinald", "estatal", "municipal", "docent",
                      "chofer", "bancari"}


def cargar():
    Xc = sparse.load_npz(DATA_INTERIM / "X_counts.npz")
    vocab = np.array(json.load(open(DATA_INTERIM / "vocabulario_counts.json")))
    X = sparse.load_npz(DATA_INTERIM / "X_tfidf.npz")
    vocab_tfidf = np.array(json.load(open(DATA_INTERIM / "vocabulario.json")))
    return Xc, vocab, X, vocab_tfidf


def npmi_topicos(componentes: np.ndarray, vocab_modelo: np.ndarray,
                 B: sparse.csc_matrix, vocab_ref: dict, top: int = 10):
    """Coherencia NPMI media por tópico. B: matriz binaria doc-término de
    referencia (csc); vocab_ref: término -> índice de columna en B."""
    n_docs = B.shape[0]
    eps = 1e-12
    coherencias = []
    for fila in componentes:
        idx = np.argsort(fila)[::-1][:top]
        terminos = [t for t in vocab_modelo[idx] if t in vocab_ref]
        cols = [vocab_ref[t] for t in terminos]
        vals = []
        for i in range(len(cols)):
            bi = B[:, cols[i]]
            di = bi.nnz
            for j in range(i + 1, len(cols)):
                bj = B[:, cols[j]]
                dj = bj.nnz
                dij = bi.multiply(bj).nnz
                p_ij = dij / n_docs
                pmi = np.log((p_ij + eps) / ((di / n_docs) * (dj / n_docs)))
                vals.append(pmi / (-np.log(p_ij + eps)))
        coherencias.append(np.mean(vals) if vals else np.nan)
    return np.array(coherencias)


def preparar_referencia(Xc, vocab):
    B = (Xc > 0).astype(np.int8).tocsc()
    return B, {t: i for i, t in enumerate(vocab)}


def top_terminos(componentes, vocab_modelo, n=15):
    return [", ".join(vocab_modelo[np.argsort(f)[::-1][:n]])
            for f in componentes]


def anclado(componentes, vocab_modelo, top=20):
    flags = []
    for fila in componentes:
        stems = set()
        for t in vocab_modelo[np.argsort(fila)[::-1][:top]]:
            stems.update(t.split())  # los bigramas aportan sus dos radicales
        a = bool(stems & ANCLA_ACCION)
        p = bool(stems & ANCLA_ACTOR_OBJETO)
        flags.append(a and p)
    return np.array(flags)


def cmd_nmf_grid():
    Xc, vocab_c, X, vocab = cargar()
    B, vref = preparar_referencia(Xc, vocab_c)
    filas = []
    for k in GRILLA_K_TOPICOS:
        nmf = NMF(n_components=k, init="nndsvda", random_state=SEMILLA,
                  max_iter=400, tol=1e-4)
        W = nmf.fit_transform(X)
        coh = npmi_topicos(nmf.components_, vocab, B, vref)
        filas.append({"modelo": "NMF", "K": k,
                      "coherencia_npmi": float(np.nanmean(coh))})
        joblib.dump({"W": W.astype(np.float32),
                     "H": nmf.components_.astype(np.float32)},
                    DATA_INTERIM / f"nmf_K{k}.joblib")
        print(filas[-1])
    pd.DataFrame(filas).to_csv(DATA_INTERIM / "grid_nmf.csv", index=False)


def cmd_lda_grid(ks):
    Xc, vocab_c, _, _ = cargar()
    uni = np.array([i for i, w in enumerate(vocab_c) if " " not in w])
    Xu = Xc[:, uni].tocsr()
    vocab_u = vocab_c[uni]
    B, vref = preparar_referencia(Xc, vocab_c)
    filas = []
    for k in ks:
        lda = LatentDirichletAllocation(
            n_components=k, learning_method="online", max_iter=8,
            batch_size=4096, random_state=SEMILLA, n_jobs=-1,
            evaluate_every=-1)
        lda.fit(Xu)
        coh = npmi_topicos(lda.components_, vocab_u, B, vref)
        filas.append({"modelo": "LDA", "K": k,
                      "coherencia_npmi": float(np.nanmean(coh))})
        joblib.dump(lda, DATA_INTERIM / f"lda_K{k}.joblib")
        print(filas[-1])
    ruta = DATA_INTERIM / "grid_lda.csv"
    if ruta.exists():  # la grilla se corre en tandas: acumular sin duplicar
        filas = pd.read_csv(ruta).to_dict("records") + filas
    df = pd.DataFrame(filas).drop_duplicates(
        subset=["modelo", "K"], keep="last").sort_values("K")
    df.to_csv(ruta, index=False)


def _seleccionar_k(grilla: pd.DataFrame, cargador, vocab_modelo) -> int:
    """Regla de selección fijada ex ante: maximizar la coherencia NPMI
    sujeto a que el modelo produzca al menos un tópico anclable al
    lexicón del protocolo (requisito operativo de recuperación de la
    clase minoritaria). Si ningún K de la grilla produce tópicos
    anclados, se toma el máximo de coherencia sin restricción y se
    reporta la imposibilidad."""
    filas = []
    for _, r in grilla.iterrows():
        comp = cargador(int(r["K"]))
        n_anc = int(anclado(comp, vocab_modelo).sum())
        filas.append({"K": int(r["K"]), "coh": r["coherencia_npmi"],
                      "n_anclados": n_anc})
    t = pd.DataFrame(filas)
    admisibles = t[t["n_anclados"] > 0]
    base = admisibles if len(admisibles) else t
    if not len(admisibles):
        print("ADVERTENCIA: ningún K produce tópicos anclados.")
    return int(base.loc[base["coh"].idxmax(), "K"])


def cmd_finalizar():
    Xc, vocab_c, X, vocab = cargar()
    uni = np.array([i for i, w in enumerate(vocab_c) if " " not in w])
    Xu = Xc[:, uni].tocsr()
    vocab_u = vocab_c[uni]

    g_lda = pd.read_csv(DATA_INTERIM / "grid_lda.csv")
    g_nmf = pd.read_csv(DATA_INTERIM / "grid_nmf.csv")
    pd.concat([g_lda, g_nmf]).to_csv(TABLAS / "t03_grilla_coherencia.csv",
                                     index=False)
    k_lda = _seleccionar_k(
        g_lda, lambda k: joblib.load(DATA_INTERIM / f"lda_K{k}.joblib").components_,
        vocab_u)
    k_nmf = _seleccionar_k(
        g_nmf, lambda k: joblib.load(DATA_INTERIM / f"nmf_K{k}.joblib")["H"],
        vocab)
    print(f"K* LDA = {k_lda} | K* NMF = {k_nmf}")

    # ------------------------------------------------------------- LDA ---
    lda = joblib.load(DATA_INTERIM / f"lda_K{k_lda}.joblib")
    gamma = lda.transform(Xu)
    flags_lda = anclado(lda.components_, vocab_u)
    dom_lda = gamma.argmax(1)
    pred_lda = flags_lda[dom_lda]

    # ------------------------------------------------------------- NMF ---
    d = joblib.load(DATA_INTERIM / f"nmf_K{k_nmf}.joblib")
    W, H = d["W"], d["H"]
    flags_nmf = anclado(H, vocab)
    dom_nmf = W.argmax(1)
    pred_nmf = flags_nmf[dom_nmf]

    # tablas de tópicos
    for nombre, comp, voc, flags, kk in [
            ("lda", lda.components_, vocab_u, flags_lda, k_lda),
            ("nmf", H, vocab, flags_nmf, k_nmf)]:
        tt = pd.DataFrame({"topico": range(len(flags)),
                           "anclado_conflicto": flags,
                           "terminos_principales": top_terminos(comp, voc)})
        tt.to_csv(TABLAS / f"t04_topicos_{nombre}_K{kk}.csv", index=False)
        print(f"{nombre.upper()} K={kk}: tópicos anclados = {int(flags.sum())}")

    np.save(DATA_INTERIM / "pred_lda.npy", pred_lda)
    np.save(DATA_INTERIM / "pred_nmf.npy", pred_nmf)
    np.save(DATA_INTERIM / "gamma_lda.npy", gamma.astype(np.float32))
    json.dump({"k_lda": k_lda, "k_nmf": k_nmf,
               "topicos_anclados_lda": [int(i) for i in np.where(flags_lda)[0]],
               "topicos_anclados_nmf": [int(i) for i in np.where(flags_nmf)[0]]},
              open(DATA_INTERIM / "seleccion_topicos.json", "w"), indent=1)
    print("Docs SÍ (LDA):", int(pred_lda.sum()),
          "| Docs SÍ (NMF):", int(pred_nmf.sum()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("nmf-grid")
    p = sub.add_parser("lda-grid")
    p.add_argument("--ks", nargs="+", type=int, required=True)
    sub.add_parser("finalizar")
    a = ap.parse_args()
    if a.cmd == "nmf-grid":
        cmd_nmf_grid()
    elif a.cmd == "lda-grid":
        cmd_lda_grid(a.ks)
    else:
        cmd_finalizar()
