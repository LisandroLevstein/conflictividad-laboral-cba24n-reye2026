# -*- coding: utf-8 -*-
"""09 — Inferencia sobre la muestra de EVALUACIÓN.

Estima prevalencia, precisión, exhaustividad y F1 con pesos de expansión
por estrato e intervalos bootstrap. Se ejecuta EXCLUSIVAMENTE sobre la
muestra de evaluación (1.400 notas, doble codificación completa y
adjudicación); la muestra de ajuste no interviene, de modo que ninguna
decisión tomada mirando etiquetas —el corte de operación del ensamble—
entra en las cifras reportadas.

PONDERACIÓN. La muestra es estratificada por votos (h = 0..4) con
tamaños fijados de antemano; los pesos de inclusión son
w_h = N_h_eval / n_h, con n_h el tamaño ORIGINAL del estrato. Los casos
DUDOSO se excluyen del análisis principal mediante un INDICADOR DE
DOMINIO, conservando los pesos originales: no se recalibra el estrato
como si solo se hubiesen muestreado los casos decidibles, porque eso
supondría que toda la población del estrato es decidible e inflaría las
masas estimadas de forma no uniforme (los DUDOSO se concentran en los
estratos ricos en positivos).

Estimadores de razón para un clasificador c, sobre el dominio decidible:
  precisión = Σ w·y·c / Σ w·c        exhaustividad = Σ w·y·c / Σ w·y

PREVALENCIA (t08). El estimando principal es la proporción de positivos
entre casos DECIDIBLES, Σ w·y / Σ w sobre SÍ∪NO. Se acompaña de una banda
de sensibilidad sobre el marco completo: DUDOSO como NO (piso) y como SÍ
(techo).

BOOTSTRAP. Remuestrea con reposición los n_h ORIGINALES dentro de cada
estrato y aplica la exclusión de DUDOSO en cada réplica, reproduciendo
literalmente el diseño. B = 2.000.

Salidas: results/tables/{t07_metricas.csv, t08_prevalencia.csv,
         t18_sensibilidad_codificadores.csv, t19_comparacion_pareada.csv}
         results/figures/f09_metricas.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DATA_PROCESSED, DATA_VALIDATION, FIGURAS, SEMILLA, TABLAS

B_BOOT = 2000
CLASIFICADORES = {
    "proto_estricta": "Regla protocolar estricta",
    "proto_amplia": "Regla protocolar amplia",
    "ens_v1": "Ensamble >= 1",
    "ens_v2": "Ensamble >= 2 (punto de operación)",
    "ens_v3": "Ensamble >= 3",
    "ens_v4": "Ensamble = 4",
    "m_lda": "LDA anclado",
    "m_nmf": "NMF anclado",
    "m_kmeans": "K-medias anclado",
    "m_prototipo": "Prototipo (umbral GMM)",
}


def metricas(y, w, c, dom):
    """Estimadores de razón sobre el dominio (dom = 1 si decidible)."""
    m = dom > 0
    y, w, c = y[m], w[m], c[m]
    vp = float(np.sum(w * y * c))
    den_p, den_r = float(np.sum(w * c)), float(np.sum(w * y))
    p = vp / den_p if den_p > 0 else 0.0
    r = vp / den_r if den_r > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def main() -> None:
    rng = np.random.default_rng(SEMILLA)
    et = pd.read_csv(DATA_VALIDATION / "muestra_evaluacion_etiquetada.csv")
    dis = pd.read_csv(DATA_VALIDATION / "muestra_evaluacion_diseno.csv")
    pred = pd.read_parquet(DATA_PROCESSED / "predicciones.parquet")

    df = et[["id_doc", "etiqueta", "etiqueta_cod1", "etiqueta_cod2"]].merge(
        dis[["id_doc", "votos", "N_h_eval"]], on="id_doc").merge(
        pred[["id_doc", "m_lda", "m_nmf", "m_kmeans", "m_prototipo",
              "proto_amplia", "proto_estricta", "votos"]].drop(
            columns=["votos"]), on="id_doc").reset_index(drop=True)
    for v in [1, 2, 3, 4]:
        df[f"ens_v{v}"] = (df[["m_lda", "m_nmf", "m_kmeans",
                               "m_prototipo"]].sum(axis=1) >= v)

    # ---- pesos ORIGINALES (n_h sobre las 1.400) e indicador de dominio ----
    n_h = df.groupby("votos")["id_doc"].count()
    df["w"] = df["N_h_eval"] / df["votos"].map(n_h)
    df["dom"] = (df["etiqueta"] != "DUDOSO").astype(float)
    df["y"] = (df["etiqueta"] == "SI").astype(float)
    N = float(dis.drop_duplicates("votos")["N_h_eval"].sum())
    print("n_h:", n_h.to_dict(), "| N marco:", int(N))

    y, w, dom = df["y"].to_numpy(), df["w"].to_numpy(), df["dom"].to_numpy()
    grupos = [np.where(df["votos"].to_numpy() == h)[0] for h in range(5)]
    replicas = [np.concatenate([rng.choice(g, len(g), True) for g in grupos])
                for _ in range(B_BOOT)]

    # ------------------------------------------------- métricas (t07) ----
    escenarios = {
        "principal (DUDOSO excluidos)": dom,
        "DUDOSO como SI": np.ones_like(dom),
        "DUDOSO como NO": np.ones_like(dom),
    }
    y_esc = {
        "principal (DUDOSO excluidos)": y,
        "DUDOSO como SI": np.where(df["etiqueta"] == "DUDOSO", 1.0, y),
        "DUDOSO como NO": y,
    }
    filas = []
    for esc, dm in escenarios.items():
        ye = y_esc[esc]
        for col, nom in CLASIFICADORES.items():
            c = df[col].astype(float).to_numpy()
            p, r, f = metricas(ye, w, c, dm)
            fila = {"escenario": esc, "clasificador": col, "nombre": nom,
                    "precision": round(p, 4), "recall": round(r, 4),
                    "f1": round(f, 4)}
            if esc.startswith("principal"):
                bs = [metricas(ye[ix], w[ix], c[ix], dm[ix])[2]
                      for ix in replicas]
                fila["f1_lo"] = round(float(np.percentile(bs, 2.5)), 4)
                fila["f1_hi"] = round(float(np.percentile(bs, 97.5)), 4)
            filas.append(fila)
    t07 = pd.DataFrame(filas)
    t07.to_csv(TABLAS / "t07_metricas.csv", index=False)
    pr = t07[t07["escenario"].str.startswith("principal")].sort_values(
        "f1", ascending=False)
    print("\n--- t07 principal ---")
    print(pr[["clasificador", "precision", "recall", "f1",
              "f1_lo", "f1_hi"]].to_string(index=False))

    # ---------------------------------------------- prevalencia (t08) ----
    si_mass = float(np.sum(w * y))
    du_mass = float(np.sum(w * (df["etiqueta"] == "DUDOSO")))
    prev_dec = si_mass / float(np.sum(w * dom))
    bs = []
    for ix in replicas:
        d_ = dom[ix]
        bs.append(float(np.sum(w[ix] * y[ix] * d_) / np.sum(w[ix] * d_)))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    t08 = pd.DataFrame([
        {"escenario": "entre decidibles (SI/(SI+NO))",
         "prevalencia": round(prev_dec, 4), "ic95_lo": round(float(lo), 4),
         "ic95_hi": round(float(hi), 4), "n_estimado": int(round(si_mass))},
        {"escenario": "marco completo, DUDOSO como NO",
         "prevalencia": round(si_mass / N, 4), "ic95_lo": "", "ic95_hi": "",
         "n_estimado": int(round(si_mass))},
        {"escenario": "marco completo, DUDOSO como SI",
         "prevalencia": round((si_mass + du_mass) / N, 4), "ic95_lo": "",
         "ic95_hi": "", "n_estimado": int(round(si_mass + du_mass))}])
    t08.to_csv(TABLAS / "t08_prevalencia.csv", index=False)
    print("\n--- t08 ---")
    print(t08.to_string(index=False))

    # ------------------------------- sensibilidad por codificador (t18) --
    filas18 = []
    for cod in ["cod1", "cod2"]:
        dm = df[f"etiqueta_{cod}"].isin(["SI", "NO"]).astype(float).to_numpy()
        yy = (df[f"etiqueta_{cod}"] == "SI").astype(float).to_numpy()
        for col, nom in CLASIFICADORES.items():
            p, r, f = metricas(yy, w, df[col].astype(float).to_numpy(), dm)
            filas18.append({"codificacion": cod, "clasificador": col,
                            "precision": round(p, 4), "recall": round(r, 4),
                            "f1": round(f, 4)})
    t18 = pd.DataFrame(filas18)
    t18.to_csv(TABLAS / "t18_sensibilidad_codificadores.csv", index=False)
    piv = t18.pivot(index="clasificador", columns="codificacion", values="f1")
    piv["brecha"] = (piv["cod1"] - piv["cod2"]).abs().round(4)
    print("\n--- t18 (F1 por codificación) ---")
    print(piv.to_string())

    # ------------------------------------ comparación pareada (t19) ------
    ce = df["proto_estricta"].astype(float).to_numpy()
    co = df["ens_v2"].astype(float).to_numpy()
    dif = metricas(y, w, ce, dom)[2] - metricas(y, w, co, dom)[2]
    difs = np.array([metricas(y[ix], w[ix], ce[ix], dom[ix])[2]
                     - metricas(y[ix], w[ix], co[ix], dom[ix])[2]
                     for ix in replicas])
    pd.DataFrame([{
        "comparacion": "F1(regla estricta) - F1(ensamble >=2)",
        "diferencia_puntual": round(float(dif), 4),
        "ic95_lo": round(float(np.percentile(difs, 2.5)), 4),
        "ic95_hi": round(float(np.percentile(difs, 97.5)), 4),
        "prop_dif_positiva": round(float((difs > 0).mean()), 3),
        "B": B_BOOT}]).to_csv(TABLAS / "t19_comparacion_pareada.csv",
                              index=False)
    print(f"\n--- t19 --- dif={dif:+.4f} "
          f"IC[{np.percentile(difs,2.5):+.4f}, {np.percentile(difs,97.5):+.4f}]")

    # ------------------------------------------------------- figura ------
    ord_ = pr["clasificador"].tolist()
    x = np.arange(len(ord_))
    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    for i, (met, col) in enumerate([("precision", "#2e5c8a"),
                                    ("recall", "#c00000"),
                                    ("f1", "#548235")]):
        v = [float(pr[pr["clasificador"] == c][met].iloc[0]) for c in ord_]
        ax.bar(x + (i - 1) * 0.26, v, 0.26, label=met, color=col)
    err_lo = [float(pr[pr["clasificador"] == c]["f1"].iloc[0]
                    - pr[pr["clasificador"] == c]["f1_lo"].iloc[0])
              for c in ord_]
    err_hi = [float(pr[pr["clasificador"] == c]["f1_hi"].iloc[0]
                    - pr[pr["clasificador"] == c]["f1"].iloc[0]) for c in ord_]
    ax.errorbar(x + 0.26, [float(pr[pr["clasificador"] == c]["f1"].iloc[0])
                           for c in ord_], yerr=[err_lo, err_hi], fmt="none",
                ecolor="black", capsize=2, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([CLASIFICADORES[c].replace(" (punto de operación)", "")
                        for c in ord_], rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("Métricas ponderadas por diseño (IC 95 % bootstrap para F1)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURAS / "f09_metricas.png", dpi=200)


if __name__ == "__main__":
    main()
