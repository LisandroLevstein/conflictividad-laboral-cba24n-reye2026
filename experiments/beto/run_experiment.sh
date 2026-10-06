#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

python 00_verificar_datos.py
python 01_probar_memoria.py
python 02_cv_ajuste.py
python 03_entrenar_predecir.py
python 04_evaluar.py
