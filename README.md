# Laboratorio 4 - Análisis de Datos Geoespaciales

Este proyecto estudia la señal estimada de cianobacteria en los lagos Atitlán y
Amatitlán a partir de imágenes Sentinel-2. El avance cubre los ejercicios 1 al
4: conexión con Copernicus, selección de datos, cálculo de NDVI, NDWI y Cya, y
preparación del análisis temporal.

El archivo principal es:

```text
notebooks/laboratorio-4-datos-geoespaciales.ipynb
```

El notebook está organizado para leerse de arriba hacia abajo. Primero presenta
el problema en lenguaje sencillo y después muestra el código y las salidas. Las
funciones más largas viven en `src/procesamiento_geoespacial.py` para que el
cuaderno no se convierta en una pared de código.

## Estructura

```text
.
├── data/
│   ├── raw/                  insumos originales separados por lago
│   └── processed/            índices, tablas y trabajos generados
├── notebooks/                cuaderno principal del laboratorio
├── src/                      funciones de conexión y procesamiento
├── reports/                  figuras y borradores del informe
├── output/
│   ├── docx/                 informe editable
│   └── pdf/                  informe para entregar
├── codebook.md               descripción de datos, fechas y unidades
├── requirements.txt          dependencias de Python
└── README.md
```

## Cómo ejecutar el avance

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

3. Ejecutar primero el cuaderno con `EJECUTAR_COPERNICUS = False`. Esto permite
   revisar fechas, áreas y calidad de las imágenes sin iniciar trabajos remotos.

4. Cuando se tenga abierta la cuenta de Copernicus, cambiar la variable a
   `True`, ejecutar nuevamente y completar la autenticación que aparece en el
   navegador.

## Salidas esperadas

- GeoTIFF de NDVI, NDWI y cianobacteria para ambos lagos.
- CSV con el promedio de cianobacteria por lago y fecha.
- Gráfico de línea con la evolución temporal y los posibles picos.
- Mensajes breves que resumen el máximo y mínimo observado en cada lago.

Los archivos grandes producidos por Copernicus no se versionan. El notebook, el
código, el informe y las instrucciones sí deben quedar en el repositorio.

