# -*- coding: utf-8 -*-
"""Configuración global del proyecto.

Clasificación no supervisada de acciones conflictivas laborales
en el corpus Cba24n 2016-2017.
"""
from pathlib import Path

# ---------------------------------------------------------------- rutas ----
RAIZ = Path(__file__).resolve().parents[1]
DATA_RAW = RAIZ / "data" / "raw"
DATA_INTERIM = RAIZ / "data" / "interim"
DATA_PROCESSED = RAIZ / "data" / "processed"
DATA_VALIDATION = RAIZ / "data" / "validation"
DATA_LEXICON = RAIZ / "data" / "lexicon"
DATA_EXTERNAL = RAIZ / "data" / "external"   # recursos versionados (stopwords)
RESULTS = RAIZ / "results"
FIGURAS = RESULTS / "figures"
TABLAS = RESULTS / "tables"

for p in [DATA_RAW, DATA_INTERIM, DATA_PROCESSED, DATA_VALIDATION,
          DATA_LEXICON, DATA_EXTERNAL, FIGURAS, TABLAS]:
    p.mkdir(parents=True, exist_ok=True)

ARCHIVOS_CRUDOS = [
    DATA_RAW / "cba24n_con_url_amigable_2016.csv",
    DATA_RAW / "cba24n_con_url_amigable_2017.csv",
]

# ------------------------------------------------------------ parámetros ----
SEMILLA = 20162017          # reproducibilidad total
MIN_DF = 10                 # frec. documental mínima (vocabulario)
MAX_DF = 0.50               # frec. documental máxima (proporción)
N_COMPONENTES_LSA = 300     # dimensión del espacio semántico latente
GRILLA_K_TOPICOS = [20, 30, 40, 50, 60]
GRILLA_K_CLUSTERS = [40, 60, 80, 100, 120]
UMBRAL_MAYORIA_CLUSTER = 0.5   # proporción de docs "protocolo" para marcar un cluster

# Cortes del ensamble por número de votos coincidentes (de 4 detectores).
# Se guardan los dos cortes de interés con nombres inequívocos; el punto
# de operación de la tesis es CORTE_OPERACION (3 votos), elegido sobre la
# muestra de ajuste (script 08, tabla t16). CORTE_TAMIZ (2 votos) es el
# régimen de alta exhaustividad para tamizaje asistido. La evaluación
# (script 09) construye además todos los cortes 1..4 para comparar.
# Punto de operación elegido sobre la muestra de ajuste (script 08, t16):
# empate estadístico entre >=2 y >=3 en F1 (0,820 vs 0,815; IC de la
# diferencia [-0,060; +0,065]); se resolvió por criterio sustantivo a
# favor de la exhaustividad (recall 1,000 vs 0,804 en el ajuste).
CORTE_OPERACION = 2
