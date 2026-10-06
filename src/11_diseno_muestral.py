# -*- coding: utf-8 -*-
"""11 — Análisis del diseño muestral.

Dos piezas, con fuentes de datos deliberadamente distintas:

(A) COTA DEL ESTRATO DE CERO VOTOS Y PISO CONDICIONAL DE EXHAUSTIVIDAD.
    Con x positivos observados en n0 lecturas de la muestra de
    evaluación, la cota superior exacta de Clopper-Pearson (97,5 %
    unilateral) para p0 acota los positivos no señalados por ningún
    método. El "piso" resultante NO es un límite inferior de confianza
    para la exhaustividad: los positivos de los demás estratos entran
    por su estimación puntual. Es un análisis de peor caso CONDICIONAL
    respecto de p0, y así se reporta. Si las etiquetas finales de
    evaluación aún no existen (doble codificación en curso), esta pieza
    emite la curva de diseño (t14) sin la parte observada.

(B) ASIGNACIÓN DE LA MUESTRA DE EVALUACIÓN (cobertura + Neyman).
    Se calcula sobre la MUESTRA DE AJUSTE, porque es la información de
    la que se disponía al decidir la asignación, y su reproducción aquí
    documenta esa decisión de forma prospectiva. Usar los datos de
    evaluación para justificar su propia asignación sería circular.

Salidas: results/tables/{t14_cota_estrato0.csv, t15_neyman.csv},
         results/figures/f10_diseno_muestral.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta

from config import CORTE_OPERACION, DATA_VALIDATION, FIGURAS, TABLAS


def cota_cp(x: int, n: int, conf: float = 0.975) -> float:
    """Cota superior exacta de Clopper-Pearson para una proporción."""
    return float(beta.ppf(conf, x + 1, n - x)) if x < n else 1.0


def main() -> None:
    # ------------------------------------------------------------------ (A)
    # Peor caso condicional del estrato de cero votos. Se estiman las masas
    # de positivos por estrato con los PESOS DE INCLUSIÓN ORIGINALES
    # (N_h_eval / n_h, con n_h el tamaño completo del estrato) y la
    # exclusión de DUDOSO como INDICADOR DE DOMINIO: no se recalibran los
    # pesos como si solo se hubiesen muestreado los casos decidibles.
    dis = pd.read_csv(DATA_VALIDATION / "muestra_evaluacion_diseno.csv")
    N = dis.drop_duplicates("votos").set_index("votos")["N_h_eval"].sort_index()
    ruta = DATA_VALIDATION / "muestra_evaluacion_etiquetada.csv"
    et = pd.read_csv(ruta)
    d = et[["id_doc", "etiqueta"]].merge(dis[["id_doc", "votos", "N_h_eval"]],
                                         on="id_doc")
    n_h = d.groupby("votos")["id_doc"].count().sort_index()
    d["w"] = d["N_h_eval"] / d["votos"].map(n_h)
    d["y"] = (d["etiqueta"] == "SI").astype(float)
    Y = d.groupby("votos").apply(lambda g: float((g["w"] * g["y"]).sum()),
                                 include_groups=False).sort_index()
    n0 = int(n_h.loc[0])
    x0 = int(d.loc[(d["votos"] == 0) & (d["y"] == 1)].shape[0])
    Y_ge1 = float(Y.loc[1:].sum())
    TP = float(Y.loc[CORTE_OPERACION:].sum())
    print(f"masas de positivos por estrato: "
          f"{ {h: round(v, 1) for h, v in Y.items()} }  total {Y.sum():.0f}")
    print(f"Estrato 0: n0 = {n0}, positivos observados x0 = {x0}")
    print(f"Positivos estimados en votos >= 1: {Y_ge1:.0f}")

    # --- t14: cota del estrato 0 y piso condicional de exhaustividad ---
    # Fila "observada": con el x0 efectivamente hallado. Filas de diseño:
    # curva de sensibilidad suponiendo x0 = 0 para distintos n0.
    filas14 = []
    for nn0 in [150, 300, 450, 630, 800, 1000]:
        c = cota_cp(0, nn0)
        H = c * float(N.loc[0])
        filas14.append({"n0": nn0, "x0": 0, "cota_cp_p0": round(c, 4),
                        "ocultos_max": int(round(H)),
                        "piso_cond_recall": round(TP / (Y_ge1 + H), 3),
                        "tipo": "diseño"})
    c_obs = cota_cp(x0, n0)
    H_obs = c_obs * float(N.loc[0])
    filas14.append({"n0": n0, "x0": x0, "cota_cp_p0": round(c_obs, 4),
                    "ocultos_max": int(round(H_obs)),
                    "piso_cond_recall": round(TP / (Y_ge1 + H_obs), 3),
                    "tipo": "observado"})
    t14 = pd.DataFrame(filas14)
    t14.to_csv(TABLAS / "t14_cota_estrato0.csv", index=False)
    print("\n(A) cota del estrato 0 y piso condicional (corte >= "
          f"{CORTE_OPERACION}):")
    print(t14.to_string(index=False))

    # ------------------------------------------------------------------ (B)
    # Asignación de la muestra de evaluación: reproduce el cálculo del
    # script 12 a partir de los p_h suavizados de la muestra de ajuste
    # (t15b). El marco es el corpus menos la muestra de ajuste.
    import pandas as _pd
    from config import DATA_PROCESSED
    dia = pd.read_csv(DATA_VALIDATION / "muestra_ajuste_diseno.csv")
    pred = _pd.read_parquet(DATA_PROCESSED / "predicciones.parquet")
    ph = pd.read_csv(TABLAS / "t15b_ph_ajuste.csv").set_index("votos")
    Na = pred[~pred["id_doc"].isin(set(dia["id_doc"]))].groupby(
        "votos")["id_doc"].count().sort_index()
    pa = ph["p_h_jeffreys"]
    Sa = np.sqrt(pa * (1 - pa))
    peso = (Na * Sa) / (Na * Sa).sum()
    t15 = pd.DataFrame({
        "estrato": Na.index, "N_h_marco": Na.values,
        "p_h_ajuste": pa.round(4).values, "S_h_ajuste": Sa.round(4).values,
        "peso_neyman": peso.round(4).values,
        "n_asignado": dis.groupby("votos")["id_doc"].count()
        .sort_index().values})
    t15.to_csv(TABLAS / "t15_neyman.csv", index=False)
    print("\n(B) Asignación de la evaluación, con la información de ajuste:")
    print(t15.to_string(index=False))
    p14 = peso.loc[1:] / peso.loc[1:].sum()
    print("Pesos de Neyman entre estratos 1-4:",
          {int(k): round(float(v), 3) for k, v in p14.items()})

    # --------------------------------------------------------------- figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.8))
    nn = np.arange(50, 1001, 10)
    ax1.plot(nn, [100 * cota_cp(0, int(v)) for v in nn], color="#1f4e79")
    ax1.axvline(n0, color="#c00000", ls="--", lw=1)
    ax1.annotate(f"n₀ = {n0}\ncota = {100 * cota_cp(0, n0):.1f} %",
                 xy=(n0, 100 * cota_cp(0, n0)), xytext=(n0 + 25, 8),
                 fontsize=8, color="#c00000")
    ax1.set_xlabel("lecturas en el estrato de cero votos (n₀)")
    ax1.set_ylabel("cota superior de p₀ (%)")
    ax1.set_title("Cobertura: cota de Clopper–Pearson con 0 positivos",
                  fontsize=9)
    ax2.bar([str(i) for i in t15["estrato"]], t15["peso_neyman"],
            color="#548235")
    ax2.set_xlabel("estrato (votos)")
    ax2.set_ylabel("peso de Neyman")
    ax2.set_title("Asignación de Neyman según la muestra de ajuste",
                  fontsize=9)
    for a_ in (ax1, ax2):
        a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURAS / "f10_diseno_muestral.png", dpi=200)
    print("\nfigura f10_diseno_muestral.png lista")


if __name__ == "__main__":
    main()
