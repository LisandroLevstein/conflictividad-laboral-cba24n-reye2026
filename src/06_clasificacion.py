# -*- coding: utf-8 -*-
"""06 — Clasificador por prototipo léxico y ensamble final.

Método M4 (prototipo semántico):
  1. conjunto semilla = documentos que satisfacen PROTO_estricta
     (acción + protagonista + objeto), regla de alta precisión que no
     usa etiquetas manuales;
  2. prototipo = centroide L2-normalizado del conjunto semilla en el
     espacio LSA;
  3. puntaje s_i = similitud coseno de cada documento con el prototipo;
  4. umbral: mezcla gaussiana de 2 componentes ajustada sobre {s_i}
     (EM, 10 inicios); se clasifica SÍ el documento cuya probabilidad
     a posteriori de pertenecer a la componente de media mayor supera
     0,5. El punto de corte es, así, endógeno a los datos.

Ensamble: voto de los cuatro métodos (LDA, NMF, K-medias, prototipo).
El punto de operación (columna ens_operacion) es votos >= CORTE_OPERACION,
elegido sobre la muestra de ajuste (script 08); se reporta la sensibilidad
del conteo para los cortes 1..4. La clasificación es binaria: no se deriva
ningún subtipo de acción.

Salida: data/processed/predicciones.parquet
        results/tables/t06_sensibilidad_votos.csv
        results/figures/f05_puntaje_prototipo.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize

from config import (CORTE_OPERACION, DATA_INTERIM, DATA_PROCESSED,
                    FIGURAS, SEMILLA, TABLAS)

def main() -> None:
    X = np.load(DATA_INTERIM / "X_lsa.npy")
    df = pd.read_parquet(DATA_PROCESSED / "corpus_limpio.parquet")
    sen = pd.read_parquet(DATA_PROCESSED / "senales_lexicon.parquet")
    assert (df["id_doc"].values == sen["id_doc"].values).all()

    # ------------------------------------------------- M4: prototipo ------
    semilla = sen["proto_estricta"].to_numpy()
    prototipo = normalize(X[semilla].mean(axis=0, keepdims=True))
    puntaje = (X @ prototipo.T).ravel()

    gmm = GaussianMixture(n_components=2, random_state=SEMILLA, n_init=10)
    gmm.fit(puntaje.reshape(-1, 1))
    comp_alta = int(np.argmax(gmm.means_.ravel()))
    post = gmm.predict_proba(puntaje.reshape(-1, 1))[:, comp_alta]
    pred_proto = post > 0.5
    umbral_equiv = float(puntaje[pred_proto].min()) if pred_proto.any() else np.nan

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.hist(puntaje, bins=120, color="#9dc3e6", edgecolor="none")
    ax.axvline(umbral_equiv, color="#c00000", ls="--", lw=1.2,
               label=f"umbral GMM ≈ {umbral_equiv:.3f}")
    ax.set_xlabel("similitud coseno con el prototipo léxico")
    ax.set_ylabel("documentos")
    ax.set_title("Puntaje de prototipo y umbral endógeno (mezcla gaussiana)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURAS / "f05_puntaje_prototipo.png", dpi=200)

    # ------------------------------------------------------ ensamble ------
    pred = pd.DataFrame({
        "id_doc": df["id_doc"],
        "m_lda": np.load(DATA_INTERIM / "pred_lda.npy"),
        "m_nmf": np.load(DATA_INTERIM / "pred_nmf.npy"),
        "m_kmeans": np.load(DATA_INTERIM / "pred_kmeans.npy"),
        "m_prototipo": pred_proto,
        "puntaje_prototipo": puntaje,
        "cluster_kmeans": np.load(DATA_INTERIM / "clusters_kmeans.npy"),
    })
    pred["votos"] = (pred[["m_lda", "m_nmf", "m_kmeans", "m_prototipo"]]
                     .sum(axis=1).astype(int))
    # Punto de operación de la tesis (columna binaria). La evaluación
    # (script 09) reconstruye además los cortes 1..4 para la sensibilidad.
    pred["ens_operacion"] = pred["votos"] >= CORTE_OPERACION

    sens = []
    for v in [1, 2, 3, 4]:
        sens.append({"votos_minimos": v,
                     "n_positivos": int((pred["votos"] >= v).sum()),
                     "prop_corpus": round(float((pred["votos"] >= v).mean()), 4)})
    pd.DataFrame(sens).to_csv(TABLAS / "t06_sensibilidad_votos.csv",
                              index=False)

    salida = df[["id_doc", "fecha_dt", "anio", "mes", "titulo",
                 "fuente_texto", "link"]].merge(pred, on="id_doc")
    salida = salida.merge(
        sen[["id_doc", "proto_amplia", "proto_estricta"]], on="id_doc")
    salida.to_parquet(DATA_PROCESSED / "predicciones.parquet", index=False)

    print("semilla (PROTO_estricta):", int(semilla.sum()))
    print("umbral equivalente GMM:", round(umbral_equiv, 4))
    for c in ["m_lda", "m_nmf", "m_kmeans", "m_prototipo",
              "ens_operacion"]:
        print(f"  {c:18s}: {int(salida[c].sum()):5d} "
              f"({salida[c].mean():.2%})")
    print(pd.DataFrame(sens).to_string(index=False))


if __name__ == "__main__":
    main()
