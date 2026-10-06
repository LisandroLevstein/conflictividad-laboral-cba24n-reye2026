# conflictividad-laboral-cba24n-reye2026

Materiales de replicación del artículo **“Conflictividad laboral en prensa digital: clasificación de una clase rara”**.

## Alcance

El repositorio contiene el código de clasificación, el lexicón protocolar, archivos de diseño muestral, etiquetas sanitizadas, resultados agregados, el benchmark BETO y una base compacta de evaluación que permite reproducir la inferencia final sin redistribuir el texto periodístico.

El corpus completo de noticias de **Cba24n (2016–2017)** no se incluye por razones de derechos y condiciones de acceso. Véanse `DATA_NOTICE.md` y `data/raw/README.md`.

## Reproducción de la inferencia publicada sin el corpus

Requiere Python 3 y `numpy`/`pandas`:

```bash
python src/16_revision_inferencia_fpc_public.py
```

El script lee `data/public/evaluation_compact.csv` y escribe los resultados en `results/inferencia_fpc_revision/`. Reproduce estimadores ponderados por diseño, intervalos por linealización de Taylor con corrección por población finita y el bootstrap estratificado original como sensibilidad.

## Reproducción completa con acceso autorizado al corpus

1. Colocar los dos CSV originales en `data/raw/` con los nombres y SHA-256 documentados en `data/raw/README.md`.
2. Instalar las dependencias de `requirements.txt`.
3. Ejecutar los scripts de `src/` según la secuencia documentada por sus prefijos numéricos.

El benchmark BETO está en `experiments/beto/`. El código corresponde al commit auditado `ef6312cf38224a17068144e1e0ab2765515ae327`.

## Contenido público y privacidad del corpus

Los archivos públicos de validación contienen identificadores y etiquetas, pero no títulos, copetes, cuerpos de noticias ni justificaciones que reproduzcan contenido periodístico. `data/public/evaluation_compact.csv` contiene únicamente variables necesarias para auditar las métricas e inferencia.

## Citación

GitHub puede generar la cita desde `CITATION.cff`. El archivo está orientado al artículo de 2026; el DOI se añadirá cuando esté disponible.
