"""
Laboratorio 4 - Analisis de datos geoespaciales
Avance del 13 de agosto de 2026: ejercicios 1 al 4.

El programa se conecta a Copernicus Data Space Ecosystem por medio de openEO,
carga solo las fechas y bandas indicadas en el laboratorio, calcula NDVI, NDWI
y una estimacion de cianobacteria, y genera la serie temporal promedio por lago.

Antes de correrlo en Google Colab:
    !pip install -q openeo pandas matplotlib

La autenticacion es interactiva. El programa nunca guarda usuario o contrasena.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from functools import reduce
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import pandas as pd

try:
    import openeo
except ImportError:
    openeo = None


# ---------------------------------------------------------------------------
# Datos oficiales entregados en el laboratorio
# ---------------------------------------------------------------------------

LAGOS: Dict[str, dict] = {
    "Amatitlan": {
        "bbox": {
            "west": -90.638065,
            "east": -90.512924,
            "south": 14.412347,
            "north": 14.493799,
            "crs": "EPSG:4326",
        },
        "fechas": [
            "2025-01-28",
            "2025-04-15",
            "2025-04-28",
            "2025-11-24",
            "2026-01-08",
            "2026-02-02",
            "2026-02-07",
            "2026-03-29",
            "2026-04-13",
            "2026-04-28",
            "2026-06-19",
        ],
    },
    "Atitlan": {
        "bbox": {
            "west": -91.326256,
            "east": -91.071510,
            "south": 14.594800,
            "north": 14.750979,
            "crs": "EPSG:4326",
        },
        "fechas": [
            "2025-01-18",
            "2025-04-13",
            "2025-05-13",
            "2025-07-17",
            "2025-11-21",
            "2025-12-29",
            "2026-02-12",
            "2026-03-24",
            "2026-04-13",
            "2026-04-28",
            "2026-07-22",
        ],
    },
}

BANDAS_REFLECTANCIA = ["B02", "B03", "B04", "B08"]
BANDA_CALIDAD = ["SCL"]
RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
CARPETA_DATOS = RAIZ_PROYECTO / "data" / "processed"
CARPETA_FIGURAS = RAIZ_PROYECTO / "reports" / "figures"


def conectar():
    """Ejercicio 1: conecta y autentica la sesion de openEO."""

    if openeo is None:
        raise ImportError(
            "Falta instalar openEO. Ejecute: pip install -r requirements.txt"
        )

    conexion = openeo.connect("https://openeo.dataspace.copernicus.eu")
    conexion.authenticate_oidc()

    colecciones = conexion.list_collection_ids()
    if "SENTINEL2_L2A" not in colecciones:
        raise RuntimeError("La coleccion SENTINEL2_L2A no aparece en el servidor.")

    print("Conexion correcta.")
    print("Coleccion seleccionada: SENTINEL2_L2A")
    return conexion


def siguiente_dia(fecha: str) -> str:
    actual = datetime.strptime(fecha, "%Y-%m-%d")
    return (actual + timedelta(days=1)).strftime("%Y-%m-%d")


def bbox_como_geojson(nombre: str, bbox: dict) -> dict:
    """Convierte las coordenadas del laboratorio en un poligono GeoJSON."""

    oeste, este = bbox["west"], bbox["east"]
    sur, norte = bbox["south"], bbox["north"]
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"lago": nombre},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [oeste, sur],
                        [este, sur],
                        [este, norte],
                        [oeste, norte],
                        [oeste, sur],
                    ]],
                },
            }
        ],
    }


def cargar_solo_fechas(
    conexion,
    bbox: dict,
    fechas: Iterable[str],
    bandas: List[str],
):
    """
    Ejercicio 2: carga un cubo por fecha y luego los une.

    Se hace de esta forma para no incluir observaciones distintas de las 11
    fechas oficiales. max_cloud_cover se deja en 20 porque la fecha oficial
    mas nublada tiene 13 %.
    """

    cubos = []
    for fecha in fechas:
        cubo = conexion.load_collection(
            "SENTINEL2_L2A",
            spatial_extent=bbox,
            temporal_extent=[fecha, siguiente_dia(fecha)],
            bands=bandas,
            max_cloud_cover=20,
        )
        cubos.append(cubo)

    return reduce(lambda a, b: a.merge_cubes(b), cubos)


def preparar_indices(conexion, nombre_lago: str):
    """
    Ejercicio 3: calcula NDVI, NDWI y la estimacion de cianobacteria.

    La formula Cya proviene del script Se2WaQ de Sentinel Hub. Antes de usarla,
    los numeros digitales de Sentinel-2 se convierten a reflectancia con el
    factor 0.0001. Se excluyen nubes, sombras, nieve y pixeles que NDWI no
    identifica como agua.
    """

    datos = LAGOS[nombre_lago]
    reflectancia_dn = cargar_solo_fechas(
        conexion,
        datos["bbox"],
        datos["fechas"],
        BANDAS_REFLECTANCIA,
    )
    scl = cargar_solo_fechas(
        conexion,
        datos["bbox"],
        datos["fechas"],
        BANDA_CALIDAD,
    ).band("SCL")

    # Clases SCL excluidas: sin datos, saturado, sombra de nube, nubes,
    # cirros y nieve/hielo. La clase 6 (agua) se conserva.
    mascara_mala = (
        (scl == 0)
        | (scl == 1)
        | (scl == 3)
        | (scl == 8)
        | (scl == 9)
        | (scl == 10)
        | (scl == 11)
    )
    mascara_mala = mascara_mala.resample_cube_spatial(reflectancia_dn)
    reflectancia_dn = reflectancia_dn.mask(mascara_mala)

    # El producto L2A entrega numeros digitales. La documentacion de
    # Copernicus indica REFLECTANCIA = DN * 0.0001.
    azul = reflectancia_dn.band("B02") * 0.0001
    verde = reflectancia_dn.band("B03") * 0.0001
    rojo = reflectancia_dn.band("B04") * 0.0001
    infrarrojo = reflectancia_dn.band("B08") * 0.0001

    ndvi = (infrarrojo - rojo) / (infrarrojo + rojo)
    ndwi = (verde - infrarrojo) / (verde + infrarrojo)

    # Se2WaQ: densidad estimada en 10^3 celulas por mililitro.
    cya = 115530.31 * ((verde * rojo) / azul) ** 2.38

    # El script original calcula Cya solamente sobre agua (NDWI >= 0).
    mascara_no_agua = ndwi < 0
    mascara_division = azul <= 0
    ndvi = ndvi.mask(mascara_no_agua)
    ndwi = ndwi.mask(mascara_no_agua)
    cya = cya.mask(mascara_no_agua | mascara_division)

    # Si hay mas de una tesela el mismo dia, se obtiene una mediana diaria.
    ndvi = ndvi.aggregate_temporal_period("day", reducer="median")
    ndwi = ndwi.aggregate_temporal_period("day", reducer="median")
    cya = cya.aggregate_temporal_period("day", reducer="median")

    return {"NDVI": ndvi, "NDWI": ndwi, "CYA": cya}


def primer_csv(carpeta: Path) -> Path:
    archivos = sorted(carpeta.rglob("*.csv"))
    if not archivos:
        raise FileNotFoundError(f"No se encontro un CSV en {carpeta}")
    return archivos[0]


def ordenar_csv_openEO(archivo: Path, nombre_lago: str) -> pd.DataFrame:
    """Normaliza el CSV que entrega openEO, aunque cambie el nombre de banda."""

    tabla = pd.read_csv(archivo)
    if tabla.empty:
        raise ValueError(f"El archivo {archivo} esta vacio.")

    columnas_minusculas = {c: c.lower() for c in tabla.columns}
    fecha_col = next(
        (c for c, bajo in columnas_minusculas.items() if "date" in bajo or "time" in bajo),
        tabla.columns[0],
    )
    excluir = {fecha_col}
    excluir.update(c for c in tabla.columns if "feature" in c.lower())

    candidatas = [c for c in tabla.columns if c not in excluir]
    numericas = [c for c in candidatas if pd.api.types.is_numeric_dtype(tabla[c])]
    if not numericas:
        raise ValueError("No se encontro la columna numerica de cianobacteria.")

    valor_col = numericas[0]
    salida = tabla[[fecha_col, valor_col]].copy()
    salida.columns = ["fecha", "cianobacteria_promedio"]
    salida["fecha"] = pd.to_datetime(salida["fecha"]).dt.strftime("%Y-%m-%d")
    salida["lago"] = nombre_lago
    salida = salida[salida["fecha"].isin(LAGOS[nombre_lago]["fechas"])]
    return salida[["lago", "fecha", "cianobacteria_promedio"]].sort_values("fecha")


def descargar_serie_temporal(conexion, nombre_lago: str, cya):
    """Ejercicio 4.1: calcula y descarga el promedio diario del lago."""

    carpeta = (
        CARPETA_DATOS
        / "series_temporales"
        / "trabajos"
        / nombre_lago.lower()
    )
    carpeta.mkdir(parents=True, exist_ok=True)

    geometria = bbox_como_geojson(nombre_lago, LAGOS[nombre_lago]["bbox"])
    serie = cya.aggregate_spatial(geometries=geometria, reducer="mean")
    trabajo = serie.execute_batch(
        out_format="CSV",
        title=f"Lab4 Cianobacteria {nombre_lago}",
    )
    trabajo.get_results().download_files(carpeta)
    print(f"Trabajo terminado: {trabajo.job_id}")
    return ordenar_csv_openEO(primer_csv(carpeta), nombre_lago)


def descargar_rasters_de_una_fecha(
    indices: dict,
    nombre_lago: str,
    fecha: str,
):
    """
    Descarga NDVI, NDWI y CYA en GeoTIFF para una fecha.

    Para bajar todas las fechas se puede llamar esta funcion dentro de un ciclo.
    Cada indice queda en una carpeta diferente para no mezclar los resultados.
    """

    if fecha not in LAGOS[nombre_lago]["fechas"]:
        raise ValueError(f"{fecha} no es una fecha oficial de {nombre_lago}.")

    for nombre_indice, cubo in indices.items():
        salida = (
            CARPETA_DATOS
            / nombre_lago.lower()
            / nombre_indice.lower()
            / fecha
        )
        salida.mkdir(parents=True, exist_ok=True)
        cubo_fecha = cubo.filter_temporal(fecha, siguiente_dia(fecha))
        trabajo = cubo_fecha.execute_batch(
            out_format="GTiff",
            title=f"Lab4 {nombre_indice} {nombre_lago} {fecha}",
        )
        trabajo.get_results().download_files(salida)
        print(nombre_lago, fecha, nombre_indice, trabajo.job_id)


def descargar_rasters_multitemporales(indices: dict, nombre_lago: str):
    """Descarga un cubo GeoTIFF por indice con las 11 fechas oficiales.

    Esta opcion evita lanzar un trabajo independiente por cada fecha. El
    backend puede entregar uno o varios archivos por cubo, segun su formato de
    salida, pero todos quedan separados por lago e indice.
    """

    for nombre_indice, cubo in indices.items():
        salida = (
            CARPETA_DATOS
            / nombre_lago.lower()
            / nombre_indice.lower()
        )
        salida.mkdir(parents=True, exist_ok=True)
        trabajo = cubo.execute_batch(
            out_format="GTiff",
            title=f"Lab4 {nombre_indice} {nombre_lago} - 11 fechas",
        )
        trabajo.get_results().download_files(salida)
        print(nombre_lago, nombre_indice, trabajo.job_id)


def graficar_evolucion(tabla: pd.DataFrame) -> Path:
    """Ejercicios 4.2 y 4.3: crea el grafico y marca el maximo de cada lago."""

    CARPETA_FIGURAS.mkdir(parents=True, exist_ok=True)
    tabla = tabla.copy()
    tabla["fecha"] = pd.to_datetime(tabla["fecha"])

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=150)
    colores = {"Amatitlan": "#d95f02", "Atitlan": "#1b9e77"}

    for lago, grupo in tabla.groupby("lago"):
        grupo = grupo.sort_values("fecha")
        ax.plot(
            grupo["fecha"],
            grupo["cianobacteria_promedio"],
            marker="o",
            linewidth=2,
            label=lago,
            color=colores.get(lago),
        )
        if grupo["cianobacteria_promedio"].notna().any():
            fila_pico = grupo.loc[grupo["cianobacteria_promedio"].idxmax()]
            ax.annotate(
                f"Pico: {fila_pico['fecha']:%d/%m/%Y}",
                (fila_pico["fecha"], fila_pico["cianobacteria_promedio"]),
                xytext=(7, 9),
                textcoords="offset points",
                fontsize=8,
            )

    ax.set_title("Evolucion temporal de la cianobacteria estimada")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Promedio estimado (10^3 celulas/ml)")
    ax.grid(alpha=0.25)
    ax.legend(title="Lago")
    fig.autofmt_xdate()
    fig.tight_layout()

    destino = CARPETA_FIGURAS / "evolucion_cianobacteria.png"
    fig.savefig(destino, bbox_inches="tight")
    plt.show()
    return destino


def resumir_picos(tabla: pd.DataFrame) -> None:
    """Imprime una interpretacion basica para completar el informe."""

    for lago, grupo in tabla.groupby("lago"):
        grupo = grupo.dropna(subset=["cianobacteria_promedio"])
        if grupo.empty:
            print(f"{lago}: no hay datos validos para interpretar.")
            continue
        pico = grupo.loc[grupo["cianobacteria_promedio"].idxmax()]
        minimo = grupo.loc[grupo["cianobacteria_promedio"].idxmin()]
        print(
            f"{lago}: el valor mas alto aparece el {pico['fecha']} "
            f"({pico['cianobacteria_promedio']:.2f}); el mas bajo aparece "
            f"el {minimo['fecha']} ({minimo['cianobacteria_promedio']:.2f})."
        )


def ejecutar_avance(descargar_rasters: bool = True) -> pd.DataFrame:
    """Ejecuta de principio a fin los ejercicios 1 al 4.

    El argumento ``descargar_rasters`` permite omitir temporalmente los seis
    trabajos GeoTIFF cuando solo se quiere comprobar la serie temporal.
    """

    CARPETA_DATOS.mkdir(parents=True, exist_ok=True)
    conexion = conectar()

    # Se preparan los procesos sin descargar escenas completas.
    indices_amatitlan = preparar_indices(conexion, "Amatitlan")
    indices_atitlan = preparar_indices(conexion, "Atitlan")

    if descargar_rasters:
        descargar_rasters_multitemporales(indices_amatitlan, "Amatitlan")
        descargar_rasters_multitemporales(indices_atitlan, "Atitlan")

    # Promedios de cianobacteria para las 11 fechas oficiales de cada lago.
    amatitlan = descargar_serie_temporal(
        conexion, "Amatitlan", indices_amatitlan["CYA"]
    )
    atitlan = descargar_serie_temporal(
        conexion, "Atitlan", indices_atitlan["CYA"]
    )
    resultados = pd.concat([amatitlan, atitlan], ignore_index=True)

    carpeta_series = CARPETA_DATOS / "series_temporales"
    carpeta_series.mkdir(parents=True, exist_ok=True)
    ruta_csv = carpeta_series / "serie_temporal_cianobacteria.csv"
    resultados.to_csv(ruta_csv, index=False)
    graficar_evolucion(resultados)
    resumir_picos(resultados)
    print(f"Tabla final: {ruta_csv}")
    return resultados


def main():
    ejecutar_avance(descargar_rasters=True)


if __name__ == "__main__":
    main()
