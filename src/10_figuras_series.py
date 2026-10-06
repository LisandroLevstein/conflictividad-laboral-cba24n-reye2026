# -*- coding: utf-8 -*-
"""10 — Figuras y series temporales.

Genera: f01 volumen mensual del corpus; f02 distribución de longitudes;
f03 curvas de coherencia NPMI; f06 mapa t-SNE (muestra de 6.000 docs)
coloreado por la clasificación final; f07 serie mensual de noticias
detectadas (cortes ≥2 y ≥3); f08 acuerdo de Jaccard entre métodos;
t11 meses pico
con notas ilustrativas.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.manifold import TSNE

from config import DATA_INTERIM, DATA_PROCESSED, FIGURAS, SEMILLA, TABLAS

AZUL, ROJO, GRIS = "#1f4e79", "#c00000", "#b0b0b0"


def main() -> None:
    df = pd.read_parquet(DATA_PROCESSED / "corpus_limpio.parquet")
    pred = pd.read_parquet(DATA_PROCESSED / "predicciones.parquet")
    pred["v3"] = pred["votos"] >= 2
    pred["v2"] = pred["votos"] >= 1

    # --------------------------------------------------- f01: volumen -----
    vol = df.groupby("mes")["id_doc"].count()
    fig, ax = plt.subplots(figsize=(8.6, 3.2))
    ax.bar(range(len(vol)), vol.values, color=AZUL)
    ax.set_xticks(range(0, len(vol), 3))
    ax.set_xticklabels(vol.index[::3], rotation=45, ha="right")
    ax.set_ylabel("noticias")
    ax.set_title("Volumen mensual del corpus Cba24n (2016–2017)")
    fig.tight_layout(); fig.savefig(FIGURAS / "f01_volumen_mensual.png", dpi=200)

    # -------------------------------------------------- f02: longitud -----
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.hist(df["n_palabras"].clip(upper=1200), bins=60, color=AZUL)
    ax.axvline(df["n_palabras"].median(), color=ROJO, ls="--",
               label=f"mediana = {int(df['n_palabras'].median())}")
    ax.set_xlabel("palabras por documento (recortado en 1.200)")
    ax.set_ylabel("documentos"); ax.legend()
    ax.set_title("Longitud del texto de análisis")
    fig.tight_layout(); fig.savefig(FIGURAS / "f02_longitud.png", dpi=200)

    # ------------------------------------------------- f03: coherencia ----
    g = pd.read_csv(TABLAS / "t03_grilla_coherencia.csv")
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for mod, col in [("LDA", AZUL), ("NMF", ROJO)]:
        s = g[g["modelo"] == mod].sort_values("K")
        ax.plot(s["K"], s["coherencia_npmi"], "o-", color=col, label=mod)
    ax.set_xlabel("número de tópicos $K$"); ax.set_ylabel("coherencia NPMI")
    ax.set_title("Coherencia por número de tópicos"); ax.legend()
    fig.tight_layout(); fig.savefig(FIGURAS / "f03_coherencia.png", dpi=200)

    # ------------------------------------------------------ f06: mapa -----
    X = np.load(DATA_INTERIM / "X_lsa.npy")
    rng = np.random.default_rng(SEMILLA)
    idx = rng.choice(len(X), 6000, replace=False)
    Z = TSNE(n_components=2, random_state=SEMILLA, perplexity=40,
             init="pca").fit_transform(X[idx])
    pos = pred["v3"].to_numpy()[idx]
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    ax.scatter(Z[~pos, 0], Z[~pos, 1], s=3, c=GRIS, alpha=0.5,
               label="resto del corpus")
    ax.scatter(Z[pos, 0], Z[pos, 1], s=6, c=ROJO, alpha=0.8,
               label="noticia clasificada positiva (ens. ≥2)")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="lower right", markerscale=3)
    ax.set_title("Proyección t-SNE del espacio LSA (muestra de 6.000 notas)")
    fig.tight_layout(); fig.savefig(FIGURAS / "f06_mapa_tsne.png", dpi=200)

    # ------------------------------------------------------ f07: serie ----
    base = df[["id_doc", "mes"]].merge(pred[["id_doc", "v2", "v3"]],
                                       on="id_doc")
    ser = base.groupby("mes")[["v2", "v3"]].sum()
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    ax.plot(range(len(ser)), ser["v2"], "o-", color=GRIS, lw=1.2,
            label="ensamble ≥2 (sensible)")
    ax.plot(range(len(ser)), ser["v3"], "o-", color=ROJO, lw=1.6,
            label="ensamble ≥3 (recomendado)")
    ax.set_xticks(range(0, len(ser), 3))
    ax.set_xticklabels(ser.index[::3], rotation=45, ha="right")
    ax.set_ylabel("noticias detectadas")
    ax.set_title("Noticias que refieren a acciones conflictivas laborales, por mes")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIGURAS / "f07_serie_mensual.png", dpi=200)

    # ------------------------------------------ f11: serie relativa -------
    # Proporción mensual detectadas/publicadas: distingue un pico de
    # visibilidad relativa de un mes de mayor producción periodística.
    pub = df.groupby("mes")["id_doc"].count().rename("pub")
    rel = ser.join(pub)
    rel["prop2"] = 100 * rel["v2"] / rel["pub"]
    rel["prop3"] = 100 * rel["v3"] / rel["pub"]
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    ax.plot(range(len(rel)), rel["prop2"], "o-", color=GRIS, lw=1.2,
            label="ensamble ≥1 (sensibilidad)")
    ax.plot(range(len(rel)), rel["prop3"], "o-", color=ROJO, lw=1.6,
            label="ensamble ≥2 (punto de operación)")
    ax.set_xticks(range(0, len(rel), 3))
    ax.set_xticklabels(rel.index[::3], rotation=45, ha="right")
    ax.set_ylabel("% de las noticias del mes")
    ax.set_title("Proporción mensual de noticias detectadas sobre "
                 "publicadas")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIGURAS / "f11_serie_proporcion.png",
                                    dpi=200)
    rel[["v2", "v3", "pub", "prop2", "prop3"]].round(3).to_csv(
        TABLAS / "t12_serie_mensual.csv")

    # --------------------------------------------------- f08: acuerdo -----
    cols = ["m_lda", "m_nmf", "m_kmeans", "m_prototipo", "proto_amplia"]
    noms = ["LDA", "NMF", "K-medias", "Prototipo", "Regla amplia"]
    M = pred[cols].to_numpy(bool)
    J = np.zeros((len(cols), len(cols)))
    for i in range(len(cols)):
        for j in range(len(cols)):
            inter = (M[:, i] & M[:, j]).sum()
            union = (M[:, i] | M[:, j]).sum()
            J[i, j] = inter / union if union else np.nan
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(J, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(noms, rotation=30,
                                                        ha="right")
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(noms)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{J[i, j]:.2f}", ha="center", va="center",
                    color="black" if J[i, j] < 0.6 else "white", fontsize=8)
    ax.set_title("Acuerdo entre métodos (índice de Jaccard)")
    fig.colorbar(im, shrink=0.85)
    fig.tight_layout(); fig.savefig(FIGURAS / "f08_jaccard.png", dpi=200)
    pd.DataFrame(J, index=noms, columns=noms).round(3).to_csv(
        TABLAS / "t13_jaccard.csv")



    # ---------------------------------------------------- t11: picos ------
    top_meses = ser["v3"].sort_values(ascending=False).head(6)
    filas = []
    aux = df[["id_doc", "mes", "titulo"]].merge(
        pred[["id_doc", "v3", "puntaje_prototipo"]], on="id_doc")
    for mes, n in top_meses.items():
        ej = (aux[(aux["mes"] == mes) & aux["v3"]]
              .nlargest(3, "puntaje_prototipo")["titulo"].tolist())
        filas.append({"mes": mes, "n_detectadas": int(n),
                      "titulos_ilustrativos": " || ".join(ej)})
    pd.DataFrame(filas).to_csv(TABLAS / "t11_meses_pico.csv", index=False)

    print("Serie mensual (≥3):")
    print(ser["v3"].to_string())
    print("\nSubtipos entre positivos (≥3):")
    print("\nMeses pico:")
    for f in filas:
        print(f"  {f['mes']}: {f['n_detectadas']} | {f['titulos_ilustrativos'][:150]}")


if __name__ == "__main__":
    main()
