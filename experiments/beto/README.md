# Benchmark supervisado BETO

Este directorio contiene un experimento exploratorio y **desmontable**, paralelo al pipeline central del artículo. No se incorpora a `run_all.sh`, no modifica archivos de `src/` ni sobrescribe resultados de la investigación principal.

## Pregunta del benchmark

Dadas exclusivamente las 250 noticias de la muestra de ajuste —238 con etiqueta decidible—, ¿qué rendimiento obtiene un encoder preentrenado en español cuando se ajusta de manera supervisada y se evalúa sobre las mismas 1.400 noticias utilizadas para evaluar el ensamble?

La comparación es válida en desempeño predictivo sobre una muestra común, pero no implica equivalencia en supervisión: BETO utiliza las etiquetas de ajuste para actualizar todos sus parámetros, mientras que el pipeline central sólo las utiliza para seleccionar el punto de operación.

## Separación entre ajuste y evaluación

- `02_cv_ajuste.py` usa sólo `muestra_ajuste_etiquetada.csv` y produce probabilidades out-of-fold.
- El umbral se selecciona con F1 ponderado por diseño sobre esas probabilidades OOF.
- `03_entrenar_predecir.py` lee las etiquetas de ajuste, pero identifica las 1.400 noticias únicamente mediante `muestra_evaluacion_diseno.csv`. No abre el archivo de etiquetas de evaluación.
- `04_evaluar.py` es el único script que abre `muestra_evaluacion_etiquetada.csv`.

## Especificación primaria fijada antes de ver el resultado

- Modelo: `dccuchile/bert-base-spanish-wwm-cased`.
- Revisión: fijada en `config.json`.
- Entrada: `texto_analisis`, sin variables derivadas del ensamble.
- Etiquetas de entrenamiento: `SI=1`, `NO=0`; `DUDOSO` se excluye.
- Validación: 5 folds estratificados conjuntamente por `votos × etiqueta`.
- Optimización: fine-tuning completo, 4 épocas, AdamW, LR `2e-5`, weight decay `0.01`.
- Batch físico: 1; acumulación de gradientes: 8.
- Pérdida: cross-entropy no ponderada.
- Los pesos de diseño se usan para seleccionar el umbral y para la evaluación, no en la función de pérdida.
- Semilla primaria: `20162017`.
- Regla de truncamiento: se conservan los primeros tokens (`truncation_side=right`).
- Longitud primaria: 512 tokens. Si no entra en la GPU, `01_probar_memoria.py` aplica la contingencia predefinida 384 y luego 256. La decisión se registra en `artifacts/hardware.json` y no depende de resultados predictivos.

No hay búsqueda extensa de hiperparámetros, early stopping ni selección basada en las 1.400 etiquetas.

## Ejecución recomendada

Desde el repositorio y con el entorno `tesis-beto` activado:

```bash
cd experiments/beto
python 00_verificar_datos.py
python 01_probar_memoria.py
python 02_cv_ajuste.py
python 03_entrenar_predecir.py
python 04_evaluar.py
```

También existe `run_experiment.sh`, pero en el servidor de la FCE conviene ejecutar primero cada etapa por separado. Los scripts 02 y 03 deben correrse dentro de `tmux`.

## Salidas

Los intermedios recreables se guardan en `artifacts/` y quedan ignorados por Git. Las tablas finales se escriben en `results/`:

- `metricas_seed_20162017.csv`
- `comparacion_pareada_seed_20162017.csv`
- `sensibilidad_dudoso_seed_20162017.csv`
- `resumen_seed_20162017.json`

La comparación pareada del benchmark originalmente ejecutado utiliza las mismas 2.000 réplicas bootstrap estratificadas para BETO y para el ensamble `>=2`. Este procedimiento se conserva como parte de la trazabilidad del experimento y como análisis de sensibilidad. La inferencia principal revisada del artículo utiliza linealización de Taylor con corrección por población finita (FPC); véase `../../src/16_revision_inferencia_fpc_public.py`.

## Sensibilidad a la semilla

La especificación primaria usa una única semilla predefinida. El código admite repetir el procedimiento sin seleccionar el mejor resultado:

```bash
python 02_cv_ajuste.py --seed 20162018
python 03_entrenar_predecir.py --seed 20162018
python 04_evaluar.py --seed 20162018
```

Las semillas adicionales declaradas en `config.json` deben presentarse como análisis de estabilidad, no reemplazar post hoc la corrida primaria.

## Limitaciones de interpretación

La muestra de ajuste fue estratificada por los votos del clasificador basado en ensamble. En particular, no contiene positivos observados en los estratos 0 y 1. Por ello, este benchmark no estima qué ocurriría con 250 noticias seleccionadas aleatoriamente del corpus. Estima el rendimiento de BETO con la muestra de ajuste efectivamente disponible bajo el diseño de la investigación.

## Consolidación de semillas

Después de ejecutar la corrida primaria y las semillas de sensibilidad, los resultados se consolidan con:

```bash
python 05_resumir_semillas.py
```

El script genera `results/resumen_semillas.csv`, que distingue la corrida primaria de las sensibilidades y reúne los umbrales, métricas ponderadas e intervalos de la comparación pareada.
