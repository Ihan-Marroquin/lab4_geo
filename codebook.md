# Codebook - Laboratorio 4

## Lagos y áreas de estudio

| Lago | Oeste | Este | Sur | Norte | Fechas |
|---|---:|---:|---:|---:|---:|
| Atitlán | -91.326256 | -91.071510 | 14.594800 | 14.750979 | 11 |
| Amatitlán | -90.638065 | -90.512924 | 14.412347 | 14.493799 | 11 |

Las coordenadas están expresadas en grados decimales y usan `EPSG:4326`.

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
parcial, por lo que debe interpretarse con cautela.

## Bandas e índices

| Elemento | Bandas | Significado |
|---|---|---|
| NDVI | B04 y B08 | Ayuda a reconocer vegetación y bordes del lago. |
| NDWI | B03 y B08 | Ayuda a separar agua de suelo y vegetación. |
| Cya | B02, B03 y B04 | Densidad estimada de cianobacteria. |
| SCL | SCL | Permite retirar nubes, sombras, nieve y datos inválidos. |

La estimación Cya usa la ecuación del script Se2WaQ:

```text
Cya = 115530.31 * ((B03 * B04) / B02) ** 2.38
```

Su unidad es `10^3 células/ml`. Es una estimación obtenida por satélite y no
reemplaza una medición directa del agua.

## Archivos procesados

| Ruta | Contenido |
|---|---|
| `data/processed/atitlan/ndvi/` | Raster de NDVI de Atitlán. |
| `data/processed/atitlan/ndwi/` | Raster de NDWI de Atitlán. |
| `data/processed/atitlan/cianobacteria/` | Raster de Cya de Atitlán. |
| `data/processed/amatitlan/ndvi/` | Raster de NDVI de Amatitlán. |
| `data/processed/amatitlan/ndwi/` | Raster de NDWI de Amatitlán. |
| `data/processed/amatitlan/cianobacteria/` | Raster de Cya de Amatitlán. |
| `data/processed/series_temporales/` | CSV con promedios por lago y fecha. |

