# Codebook - Laboratorio 4

## Lagos y áreas de estudio

| Lago | Oeste | Este | Sur | Norte | Fechas |
|---|---:|---:|---:|---:|---:|
| Atitlán | -91.326256 | -91.071510 | 14.594800 | 14.750979 | 11 |
| Amatitlán | -90.638065 | -90.512924 | 14.412347 | 14.493799 | 11 |

Las coordenadas están en grados decimales y usan `EPSG:4326`. El análisis se
reproyecta a `EPSG:32615` y se trabaja a 120 metros.

## Fechas oficiales

### Atitlán

`2025-01-18`, `2025-04-13`, `2025-05-13`, `2025-07-17`, `2025-11-21`,
`2025-12-29`, `2026-02-12`, `2026-03-24`, `2026-04-13`, `2026-04-28` y
`2026-07-22`.

### Amatitlán

`2025-01-28`, `2025-04-15`, `2025-04-28`, `2025-11-24`, `2026-01-08`,
`2026-02-02`, `2026-02-07`, `2026-03-29`, `2026-04-13`, `2026-04-28` y
`2026-06-19`.

La observación de Amatitlán del 7 de febrero de 2026 tiene cobertura válida
parcial y se interpreta con cautela.

## Bandas e índices

| Elemento | Bandas | Significado |
|---|---|---|
| NDVI | B04 y B08 | Ayuda a reconocer vegetación y bordes del lago. |
| NDWI | B03 y B08 | Ayuda a separar agua de suelo y vegetación. |
| Cya | B02, B03 y B04 | Señal estimada de cianobacteria. |
| SCL | SCL | Retira nubes, sombras, nieve y datos inválidos. |

La ecuación de Se2WaQ es:

```text
Cya = 115530.31 * ((B03 * B04) / B02) ** 2.38
```

El resultado se expresa originalmente en `10^3 células/ml`, pero aquí se
presenta en la escala visual 0-100 del script. Es una estimación satelital y no
reemplaza una medición directa del agua.

## Archivos procesados

| Ruta | Contenido |
|---|---|
| `data/processed/resultados/metricas_por_fecha.csv` | Promedios, medianas, máximos, cobertura válida y extensión alta. |
| `data/processed/resultados/correlaciones_indices.csv` | Correlaciones de Pearson y Spearman con Cya. |
| `data/processed/resultados/cubo_*.npz` | NDVI, NDWI, Cya y máscara válida por fecha. |
| `data/processed/figuras/` | Gráficas y mapas regenerables. |

## Criterios del análisis

- La escala Cya se limita entre 0 y 100, igual que la visualización Se2WaQ.
- Se considera valor alto `Cya >= 20`, un corte explícito de esa escala.
- Se retiran las clases SCL de nubes, cirros, sombras, nieve, saturación y
  ausencia de datos.
- Las correlaciones usan una muestra reproducible de hasta 3,000 píxeles por
  lago y fecha.
- La comparación seca/lluviosa es descriptiva: las fechas no son mensuales ni
  cubren suficientes años para demostrar estacionalidad.
