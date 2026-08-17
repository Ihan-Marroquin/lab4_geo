# Laboratorio 4 - Análisis de Datos Geoespaciales

Este proyecto estudia la señal estimada de cianobacteria en los lagos Atitlán y
Amatitlán a partir de imágenes Sentinel-2. El laboratorio completo incluye la
serie temporal, mapas por fecha, persistencia espacial, correlaciones con NDVI y
NDWI, comparación entre lagos y un análisis exploratorio adicional.

El archivo principal es:

```text
notebooks/laboratorio-4-datos-geoespaciales.ipynb
```

El notebook está organizado para leerse de arriba hacia abajo. Las funciones más
largas viven en `src/`: `procesamiento_geoespacial.py` conserva el flujo de
openEO y `analisis_completo.py` ejecuta la consulta reproducible de las fechas
oficiales por medio de Sentinel-2 L2A en Planetary Computer.

## Estructura

```text
.
├── data/
│   ├── raw/                  insumos originales separados por lago
│   └── processed/            índices, tablas y figuras regenerables
├── notebooks/                cuaderno principal ejecutado
├── src/                      conexión, descarga y análisis
├── reports/                  informe final en PDF
├── codebook.md               datos, fechas, unidades y criterios
├── requirements.txt          dependencias de Python
└── README.md
```

## Cómo ejecutar el laboratorio

1. Crear un entorno e instalar las dependencias:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Abrir el notebook:

```bash
jupyter notebook notebooks/laboratorio-4-datos-geoespaciales.ipynb
```

3. Ejecutar de arriba hacia abajo. Si las tablas procesadas no están
   disponibles, el flujo consulta automáticamente las escenas oficiales y lee
   únicamente la ventana de cada lago.

4. Para repetir la descarga, cambiar `ACTUALIZAR_DATOS = True`. El proceso usa
   una resolución de análisis de 120 metros para mantener el laboratorio ligero.

## Salidas

- Tabla de métricas por lago y fecha.
- Mapas de cianobacteria para las 22 observaciones.
- Evolución temporal, extensión de valores altos y fechas críticas.
- Correlaciones de Cya con NDVI y NDWI.
- Mapas de persistencia, diferencias y distribuciones por fecha.
- Comparación entre lagos y lectura descriptiva por temporada.

Los datos procesados y las figuras regenerables no se versionan. El notebook
ejecutado conserva las salidas principales; el código, el informe y las
instrucciones sí quedan en el repositorio.
