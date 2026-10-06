# -*- coding: utf-8 -*-
"""05 — Agrupamiento K-medias sobre el espacio LSA.

  * Grilla de k evaluada con el coeficiente de silueta (distancia
    coseno; submuestra fija de 6.000 documentos por costo computacional).
  * Ajuste final con KMeans (k-means++, n_init=10) en el k seleccionado.
  * Anclaje protocolar: se marca un cluster como "de conflicto laboral"
    si la proporción de sus documentos que satisface la regla
    PROTO_amplia supera el umbral de mayoría (0,5). La regla de mayoría
    evita que unos pocos documentos con vocabulario conflictivo
    contaminen la etiqueta de clusters temáticamente ajenos.
  * Clasificación: SÍ si el documento pertenece a un cluster anclado.

Salidas: data/interim/{pred_kmeans.npy, clusters_kmeans.npy},
         results/tables/t05_clusters.csv, results/figures/f04_silueta.png
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score

from config import (DATA_INTERIM, DATA_PROCESSED, FIGURAS,
                    GRILLA_K_CLUSTERS, SEMILLA, TABLAS,
                    UMBRAL_MAYORIA_CLUSTER)

def main() -> None:
    X = np.load(DATA_INTERIM / "X_lsa.npy")
    sen = pd.read_parquet(DATA_PROCESSED / "senales_lexicon.parquet")
    proto = sen["proto_amplia"].to_numpy()

    rng = np.random.default_rng(SEMILLA)
    idx_sil = rng.choice(len(X), size=6000, replace=False)

    filas = []
    for k in GRILLA_K_CLUSTERS:
        mb = MiniBatchKMeans(n_clusters=k, random_state=SEMILLA,
                             batch_size=4096, n_init=5)
        lab = mb.fit_predict(X)
        sil = silhouette_score(X[idx_sil], lab[idx_sil], metric="cosine")
        filas.append({"k": k, "silueta": float(sil)})
        print(filas[-1])
    g = pd.DataFrame(filas)
    g.to_csv(DATA_INTERIM / "grid_kmeans.csv", index=False)
    k_opt = int(g.loc[g["silueta"].idxmax(), "k"])
    print("k* =", k_opt)

    km = KMeans(n_clusters=k_opt, random_state=SEMILLA, n_init=10)
    clusters = km.fit_predict(X)
    joblib.dump(km, DATA_INTERIM / "kmeans_final.joblib")
    np.save(DATA_INTERIM / "clusters_kmeans.npy", clusters)

    # ---------------------------------------------- caracterización -------
    Xt = sparse.load_npz(DATA_INTERIM / "X_tfidf.npz").tocsr()
    vocab = np.array(json.load(open(DATA_INTERIM / "vocabulario.json")))
    tabla = []
    flags = np.zeros(k_opt, dtype=bool)
    for c in range(k_opt):
        m = clusters == c
        prop = proto[m].mean()
        flags[c] = prop > UMBRAL_MAYORIA_CLUSTER
        centro = np.asarray(Xt[m].mean(axis=0)).ravel()
        top = ", ".join(vocab[np.argsort(centro)[::-1][:12]])
        tabla.append({"cluster": c, "n_docs": int(m.sum()),
                      "prop_proto_amplia": round(float(prop), 3),
                      "anclado_conflicto": bool(flags[c]),
                      "terminos_principales": top})
    tabla = pd.DataFrame(tabla).sort_values("prop_proto_amplia",
                                            ascending=False)
    tabla.to_csv(TABLAS / "t05_clusters.csv", index=False)

    pred = flags[clusters]
    np.save(DATA_INTERIM / "pred_kmeans.npy", pred)
    print("clusters anclados:", int(flags.sum()),
          "| docs SÍ (K-medias):", int(pred.sum()))
    print(tabla.head(8).to_string(index=False, max_colwidth=90))

    # ------------------------------------------------------- figura -------
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(g["k"], g["silueta"], "o-", color="#1f4e79")
    ax.axvline(k_opt, ls="--", color="grey", lw=1)
    ax.set_xlabel("número de clusters $k$")
    ax.set_ylabel("coeficiente de silueta (coseno)")
    ax.set_title("Selección de $k$ para K-medias")
    fig.tight_layout()
    fig.savefig(FIGURAS / "f04_silueta.png", dpi=200)


if __name__ == "__main__":
    main()
