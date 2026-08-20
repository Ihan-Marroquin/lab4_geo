"""
Laboratorio 4 - Analisis de datos geoespaciales
Entrega final del 16 de agosto de 2026: ejercicios 1 al 8.

El programa se conecta a Copernicus Data Space Ecosystem por medio de openEO,
carga solo las fechas y bandas indicadas en el laboratorio, calcula NDVI, NDWI
y una estimacion de cianobacteria, y produce los analisis temporal, espacial,
de correlacion, comparativo entre lagos y exploratorio adicional.

Antes de correrlo en Google Colab:
    !pip install -q openeo pandas matplotlib rasterio folium numpy scipy

La autenticacion es interactiva. El programa nunca guarda usuario o contrasena.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from functools import reduce
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import openeo
except ImportError:
    openeo = None

try:
    import rasterio
except ImportError:
    rasterio = None

try:
    import folium
    from folium.raster_layers import ImageOverlay
except ImportError:
    folium = None

try:
    from scipy import stats
except ImportError:
    stats = None


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


# ===========================================================================
# Ejercicios 5 al 8: analisis espacial, correlacion, comparacion y exploratorio
# ===========================================================================
#
# Estas funciones trabajan con los archivos descargados por el primer bloque.
# Asumen que en ``CARPETA_DATOS`` existen carpetas por lago e indice y que la
# serie temporal promedio ya esta en
# ``CARPETA_DATOS / "series_temporales" / "serie_temporal_cianobacteria.csv"``.
#
# Estructura esperada de los rasters descargados por openEO:
#
#   data/processed/<lago>/<indice>/<fecha>/*.tif    (un GeoTIFF por cubo)
#   data/processed/<lago>/<indice>/openEO/*.tif     (varios GeoTIFF por cubo)
#
# Cuando el cubo abierto por openEO entrega varios archivos (uno por banda o
# por fecha), las funciones auxiliares eligen el archivo que corresponde al
# indice buscado.


# Umbral para considerar que un pixel presenta una floracion alta. Se obtuvo
# siguiendo las recomendaciones de la OMS para aguas recreativas, que hablan
# de riesgo a partir de 100000 celulas/ml (100 en unidades 10^3 cel/ml).
UMBRAL_CYA_ALTO = 100.0


def _importar_rasterio():
    if rasterio is None:
        raise ImportError(
            "Falta rasterio. Ejecute: pip install -r requirements.txt"
        )
    return rasterio


def _importar_folium():
    if folium is None:
        raise ImportError(
            "Falta folium. Ejecute: pip install folium"
        )
    return folium


def _importar_scipy():
    if stats is None:
        raise ImportError(
            "Falta scipy. Ejecute: pip install scipy"
        )
    return stats


def rutas_raster(nombre_lago: str, indice: str, fecha: str) -> List[Path]:
    """Busca los raster disponibles para un lago, indice y fecha."""

    carpeta = CARPETA_DATOS / nombre_lago.lower() / indice.lower()
    if not carpeta.exists():
        return []
    candidatos = []
    for extension in ("*.tif", "*.tiff"):
        candidatos.extend(carpeta.rglob(extension))
    patron_fecha = fecha.replace("-", "")
    filtrados = [
        ruta for ruta in candidatos
        if patron_fecha in ruta.name or fecha in ruta.name
    ]
    return filtrados or candidatos


def hay_rasters(nombre_lago: str, indice: str = "cianobacteria") -> bool:
    """Indica si ya existe al menos un GeoTIFF para el análisis."""

    carpeta = CARPETA_DATOS / nombre_lago.lower() / indice.lower()
    return carpeta.exists() and any(carpeta.rglob("*.tif"))


def cargar_banda(ruta: Path, banda: int = 1) -> Tuple[np.ndarray, dict]:
    """Lee una banda de un GeoTIFF y devuelve el arreglo y el perfil."""

    rio = _importar_rasterio()
    with rio.open(ruta) as src:
        arreglo = src.read(banda).astype("float64")
        perfil = src.profile.copy()
        arreglo[arreglo <= -9999] = np.nan
        if src.nodata is not None:
            arreglo = np.where(arreglo == src.nodata, np.nan, arreglo)
    return arreglo, perfil


def bbox_desde_perfil(perfil: dict) -> Tuple[float, float, float, float]:
    """Convierte el perfil de rasterio en (oeste, sur, este, norte)."""

    rio = _importar_rasterio()
    return tuple(
        rio.transform.array_bounds(
            perfil["height"], perfil["width"], perfil["transform"]
        )
    )


def resumen_estadistico(arreglo: np.ndarray) -> Dict[str, float]:
    """Devuelve estadisticas basicas sin contar nodata."""

    valido = arreglo[~np.isnan(arreglo)]
    if valido.size == 0:
        return {
            "n": 0,
            "media": np.nan,
            "mediana": np.nan,
            "minimo": np.nan,
            "maximo": np.nan,
            "desviacion": np.nan,
            "percentil_25": np.nan,
            "percentil_75": np.nan,
        }
    return {
        "n": int(valido.size),
        "media": float(np.mean(valido)),
        "mediana": float(np.median(valido)),
        "minimo": float(np.min(valido)),
        "maximo": float(np.max(valido)),
        "desviacion": float(np.std(valido)),
        "percentil_25": float(np.percentile(valido, 25)),
        "percentil_75": float(np.percentile(valido, 75)),
    }


def mapear_cianobacteria(
    nombre_lago: str,
    fechas: Optional[Iterable[str]] = None,
    guardar: bool = True,
) -> Dict[str, Path]:
    """
    Ejercicio 5.1: produce mapas estaticos de Cya para cada fecha.

    Devuelve un diccionario ``{fecha: ruta_png}``. Si ``guardar`` es True, las
    figuras se escriben en ``reports/figures/mapas_espaciales``.
    """

    fechas = list(fechas or LAGOS[nombre_lago]["fechas"])
    carpeta_salida = CARPETA_FIGURAS / "mapas_espaciales" / nombre_lago.lower()
    if guardar:
        carpeta_salida.mkdir(parents=True, exist_ok=True)

    resultados: Dict[str, Path] = {}
    for fecha in fechas:
        archivos = rutas_raster(nombre_lago, "cianobacteria", fecha)
        if not archivos:
            print(f"[aviso] No hay raster de Cya para {nombre_lago} {fecha}.")
            continue

        fig, ax = plt.subplots(figsize=(7, 6), dpi=140)
        for ruta in archivos:
            arreglo, perfil = cargar_banda(ruta)
            extent = bbox_desde_perfil(perfil)
            im = ax.imshow(
                arreglo,
                cmap="YlOrRd",
                vmin=0,
                vmax=max(UMBRAL_CYA_ALTO * 1.5, np.nanmax(arreglo) if np.any(~np.isnan(arreglo)) else 1),
                extent=extent,
                origin="upper",
            )
        fig.colorbar(im, ax=ax, label="Cya (10^3 celulas/ml)")
        ax.set_title(f"{nombre_lago} - Cianaobacteria estimada - {fecha}")
        ax.set_xlabel("Longitud")
        ax.set_ylabel("Latitud")
        fig.tight_layout()
        ruta_png = carpeta_salida / f"cya_{fecha}.png"
        if guardar:
            fig.savefig(ruta_png, bbox_inches="tight")
        plt.show()
        resultados[fecha] = ruta_png
    return resultados


def comparar_fechas(
    nombre_lago: str,
    fechas: Iterable[str],
    indice: str = "cianobacteria",
    cmap: str = "YlOrRd",
) -> Optional[Path]:
    """
    Ejercicio 5.2: panel comparativo lado a lado para varias fechas.
    """

    fechas = list(fechas)
    n = len(fechas)
    if n == 0:
        return None
    ncols = min(3, n)
    nfilas = int(np.ceil(n / ncols))
    fig, ejes = plt.subplots(
        nfilas, ncols, figsize=(4.5 * ncols, 4.2 * nfilas), dpi=140
    )
    ejes = np.atleast_2d(ejes)
    valores_max: List[float] = []

    arreglos: List[Tuple[str, np.ndarray, tuple]] = []
    for fecha in fechas:
        archivos = rutas_raster(nombre_lago, indice, fecha)
        if not archivos:
            print(f"[aviso] No hay raster de {indice} para {nombre_lago} {fecha}.")
            continue
        arreglo, perfil = cargar_banda(archivos[0])
        extent = bbox_desde_perfil(perfil)
        arreglos.append((fecha, arreglo, extent))
        if np.any(~np.isnan(arreglo)):
            valores_max.append(float(np.nanmax(arreglo)))

    if not arreglos:
        return None

    vmax = max(valores_max) if valores_max else 1.0
    if indice == "ndvi":
        vmin, vmax, cmap = -0.2, 1.0, "RdYlGn"
    elif indice == "ndwi":
        vmin, vmax, cmap = -0.5, 0.8, "Blues"
    else:
        vmin = 0.0

    for indice_ax, (fecha, arreglo, extent) in enumerate(arreglos):
        fila, columna = divmod(indice_ax, ncols)
        ax = ejes[fila, columna]
        im = ax.imshow(
            arreglo, cmap=cmap, vmin=vmin, vmax=vmax,
            extent=extent, origin="upper",
        )
        ax.set_title(fecha)
        ax.set_xlabel("Longitud")
        ax.set_ylabel("Latitud")

    for vacio in range(len(arreglos), nfilas * ncols):
        fila, columna = divmod(vacio, ncols)
        ejes[fila, columna].axis("off")

    cbar = fig.colorbar(im, ax=ejes.ravel().tolist(), shrink=0.85)
    cbar.set_label(f"{indice}")
    fig.suptitle(
        f"Comparacion de {indice} en {nombre_lago}", fontweight="bold"
    )
    fig.tight_layout()
    destino = (
        CARPETA_FIGURAS
        / "mapas_espaciales"
        / nombre_lago.lower()
        / f"comparacion_{indice}.png"
    )
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, bbox_inches="tight")
    plt.show()
    return destino


def mapa_interactivo(
    nombre_lago: str,
    fechas: Optional[Iterable[str]] = None,
) -> Optional[Path]:
    """
    Ejercicio 5.1 (alternativa): crea un mapa interactivo HTML con folium.

    Para cada fecha, exporta el raster de Cya a PNG y lo coloca como capa
    superpuesta sobre un mapa base. Si rasterio no esta disponible, devuelve
    None y se debe usar ``mapear_cianobacteria``.
    """

    folium_mod = _importar_folium()
    fechas = list(fechas or LAGOS[nombre_lago]["fechas"])
    datos = LAGOS[nombre_lago]["bbox"]
    centro = (
        (datos["north"] + datos["south"]) / 2,
        (datos["west"] + datos["east"]) / 2,
    )
    mapa = folium_mod.Map(location=centro, zoom_start=11, control_scale=True)

    for fecha in fechas:
        archivos = rutas_raster(nombre_lago, "cianobacteria", fecha)
        if not archivos:
            continue
        arreglo, perfil = cargar_banda(archivos[0])
        extent = bbox_desde_perfil(perfil)
        normalizado = np.clip(
            arreglo / max(UMBRAL_CYA_ALTO, np.nanmax(arreglo) or 1), 0, 1
        )
        rgba = plt.cm.YlOrRd(normalizado)
        rgba[..., 3] = (~np.isnan(arreglo)).astype(float) * 0.85
        # folium espera coordenadas en [lat_min, lon_min, lat_max, lon_max].
        bounds = [[extent[1], extent[0]], [extent[3], extent[2]]]
        ImageOverlay(
            image=rgba,
            bounds=bounds,
            name=f"Cya {fecha}",
            opacity=0.75,
        ).add_to(mapa)

    folium_mod.LayerControl().add_to(mapa)

    destino = CARPETA_FIGURAS / "mapas_espaciales" / f"mapa_{nombre_lago.lower()}.html"
    destino.parent.mkdir(parents=True, exist_ok=True)
    mapa.save(destino)
    return destino


def extraer_perfiles_por_pixel(
    nombre_lago: str,
) -> Optional[pd.DataFrame]:
    """
    Reune los valores de NDVI, NDWI y Cya en una sola tabla por pixel.

    La idea es muestrear un numero fijo de pixeles (los que tienen datos
    validos en todas las fechas) y devolver una tabla larga con las columnas
    ``lago``, ``fecha``, ``pixel_id``, ``ndvi``, ``ndwi``, ``cya``. Esta tabla
    sirve para los analisis 6 (correlacion) y 8.5 (interpretacion).
    """

    fechas = LAGOS[nombre_lago]["fechas"]
    grilla: Dict[Tuple[int, int], Dict[str, float]] = {}

    for fecha in fechas:
        archivos_cya = rutas_raster(nombre_lago, "cianobacteria", fecha)
        archivos_ndvi = rutas_raster(nombre_lago, "ndvi", fecha)
        archivos_ndwi = rutas_raster(nombre_lago, "ndwi", fecha)
        if not (archivos_cya and archivos_ndvi and archivos_ndwi):
            continue
        cya, _ = cargar_banda(archivos_cya[0])
        ndvi, _ = cargar_banda(archivos_ndvi[0])
        ndwi, _ = cargar_banda(archivos_ndwi[0])
        valido = (~np.isnan(cya)) & (~np.isnan(ndvi)) & (~np.isnan(ndwi))
        filas, columnas = np.where(valido)
        for fila, columna in zip(filas, columnas):
            clave = (int(fila), int(columna))
            if clave not in grilla:
                grilla[clave] = {}
            grilla[clave][fecha] = {
                "ndvi": float(ndvi[fila, columna]),
                "ndwi": float(ndwi[fila, columna]),
                "cya": float(cya[fila, columna]),
            }

    if not grilla:
        return pd.DataFrame(
            columns=["lago", "fecha", "pixel_id", "ndvi", "ndwi", "cya"]
        )

    registros = []
    for (fila, columna), valores in grilla.items():
        for fecha in fechas:
            if fecha in valores:
                registros.append({
                    "lago": nombre_lago,
                    "fecha": fecha,
                    "pixel_id": f"{fila}-{columna}",
                    "ndvi": valores[fecha]["ndvi"],
                    "ndwi": valores[fecha]["ndwi"],
                    "cya": valores[fecha]["cya"],
                })
    return pd.DataFrame(registros)


def correlacion_con_cya(
    tabla: pd.DataFrame,
    columnas: Iterable[str] = ("ndvi", "ndwi"),
) -> pd.DataFrame:
    """
    Ejercicio 6: correlacion de Pearson y Spearman entre cada columna y Cya.

    Devuelve una tabla con coeficientes, p-valores e interpretaciones
    cualitativas. La correlacion se calcula por lago para respetar la
    estructura espacial.
    """

    columnas = list(columnas)
    if tabla.empty:
        return pd.DataFrame(
            columns=[
                "lago", "variable", "n", "pearson_r", "pearson_p",
                "spearman_rho", "spearman_p", "direccion", "fortaleza",
            ]
        )
    scipy_stats = _importar_scipy()
    resumenes = []
    for lago, grupo in tabla.groupby("lago"):
        grupo = grupo.dropna(subset=["cya", *columnas])
        for columna in columnas:
            serie = grupo[columna]
            cya = grupo["cya"]
            if len(serie) < 3:
                continue
            pearson = scipy_stats.pearsonr(serie, cya)
            spearman = scipy_stats.spearmanr(serie, cya)
            resumenes.append({
                "lago": lago,
                "variable": columna,
                "n": int(len(serie)),
                "pearson_r": float(pearson.statistic),
                "pearson_p": float(pearson.pvalue),
                "spearman_rho": float(spearman.statistic),
                "spearman_p": float(spearman.pvalue),
                "direccion": (
                    "positiva" if pearson.statistic > 0 else "negativa"
                ),
                "fortaleza": clasificar_fortaleza(abs(pearson.statistic)),
            })
    return pd.DataFrame(resumenes)


def clasificar_fortaleza(abs_r: float) -> str:
    if abs_r < 0.1:
        return "muy debil"
    if abs_r < 0.3:
        return "debil"
    if abs_r < 0.5:
        return "moderada"
    if abs_r < 0.7:
        return "fuerte"
    return "muy fuerte"


def graficar_dispersion(
    tabla: pd.DataFrame,
    variable: str,
    columna_objetivo: str = "cya",
) -> Path:
    """Genera graficos de dispersion con linea de tendencia."""

    fig, ejes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=140, sharey=True)
    colores = {"Atitlan": "#1b9e77", "Amatitlan": "#d95f02"}
    resumen = []

    for ax, (lago, grupo) in zip(ejes, tabla.groupby("lago")):
        grupo = grupo.dropna(subset=[variable, columna_objetivo])
        ax.scatter(
            grupo[variable], grupo[columna_objetivo],
            s=10, alpha=0.4, color=colores.get(lago, "#444"),
        )
        if len(grupo) >= 3:
            pendiente, intercepto, r, _, _ = (
                _importar_scipy().linregress(grupo[variable], grupo[columna_objetivo])
            )
            xs = np.linspace(grupo[variable].min(), grupo[variable].max(), 50)
            ax.plot(
                xs, pendiente * xs + intercepto,
                color=colores.get(lago, "#222"), linewidth=2,
                label=f"r = {r:.2f}",
            )
            resumen.append({
                "lago": lago, "variable": variable,
                "pendiente": float(pendiente),
                "r": float(r),
            })
            ax.legend(loc="upper right", fontsize=8)
        ax.set_title(lago)
        ax.set_xlabel(variable.upper())
        ax.set_ylabel("Cya (10^3 celulas/ml)")

    fig.suptitle(
        f"Relacion entre {variable.upper()} y cianobacteria",
        fontweight="bold",
    )
    fig.tight_layout()
    destino = (
        CARPETA_FIGURAS
        / "correlacion"
        / f"dispersion_{variable}.png"
    )
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, bbox_inches="tight")
    plt.show()
    if resumen:
        (destino.parent / f"pendientes_{variable}.csv").write_text(
            pd.DataFrame(resumen).to_csv(index=False)
        )
    return destino


def extension_floracion(
    nombre_lago: str,
    umbral: float = UMBRAL_CYA_ALTO,
) -> pd.DataFrame:
    """
    Ejercicio 8.1: porcentaje del lago con valores altos de Cya por fecha.
    """

    registros = []
    for fecha in LAGOS[nombre_lago]["fechas"]:
        archivos = rutas_raster(nombre_lago, "cianobacteria", fecha)
        if not archivos:
            continue
        arreglo, _ = cargar_banda(archivos[0])
        valido = ~np.isnan(arreglo)
        alto = valido & (arreglo >= umbral)
        porcentaje = (
            100.0 * float(alto.sum()) / float(valido.sum())
            if valido.any()
            else np.nan
        )
        registros.append({
            "lago": nombre_lago,
            "fecha": fecha,
            "pixeles_validos": int(valido.sum()),
            "pixeles_altos": int(alto.sum()),
            "porcentaje_floracion": porcentaje,
        })
    return pd.DataFrame(registros)


def zonas_persistentes(
    nombre_lago: str,
    umbral: float = UMBRAL_CYA_ALTO,
    minimo_fechas: int = 3,
) -> Optional[Tuple[np.ndarray, dict]]:
    """
    Ejercicio 8.2: contabiliza cuantas fechas superan el umbral en cada pixel.

    Devuelve una matriz con la frecuencia y el perfil asociado al primer raster
    valido encontrado, para poder graficarla.
    """

    archivos_por_fecha = []
    for fecha in LAGOS[nombre_lago]["fechas"]:
        archivos = rutas_raster(nombre_lago, "cianobacteria", fecha)
        if archivos:
            archivos_por_fecha.append((fecha, archivos[0]))

    if not archivos_por_fecha:
        return None

    _, perfil = cargar_banda(archivos_por_fecha[0][1])
    forma = (perfil["height"], perfil["width"])
    acumulador = np.zeros(forma, dtype="int32")
    cobertura = np.zeros(forma, dtype="bool")

    for _, ruta in archivos_por_fecha:
        arreglo, _ = cargar_banda(ruta)
        if arreglo.shape != forma:
            continue
        cobertura |= ~np.isnan(arreglo)
        acumulador += (arreglo >= umbral).astype("int32")

    persistencia = np.where(
        cobertura & (acumulador >= minimo_fechas), acumulador, np.nan
    )
    return persistencia, perfil


def graficar_zonas_persistentes(
    nombre_lago: str,
    minimo_fechas: int = 3,
) -> Optional[Path]:
    resultado = zonas_persistentes(nombre_lago, minimo_fechas=minimo_fechas)
    if resultado is None:
        return None
    persistencia, perfil = resultado
    extent = bbox_desde_perfil(perfil)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=140)
    im = ax.imshow(
        persistencia,
        cmap="hot_r",
        extent=extent,
        origin="upper",
        vmin=0,
        vmax=int(np.nanmax(persistencia)) if np.any(~np.isnan(persistencia)) else 1,
    )
    fig.colorbar(im, ax=ax, label="Fechas con Cya alta")
    ax.set_title(
        f"{nombre_lago} - persistencia de zonas con floracion ({minimo_fechas}+ fechas)"
    )
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    fig.tight_layout()
    destino = (
        CARPETA_FIGURAS
        / "mapas_espaciales"
        / nombre_lago.lower()
        / "zonas_persistentes.png"
    )
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, bbox_inches="tight")
    plt.show()
    return destino


def histogramas_por_fecha(nombre_lago: str) -> Optional[Path]:
    """Ejercicio 8.3: histogramas comparativos entre fechas."""

    fig, ax = plt.subplots(figsize=(9, 5), dpi=140)
    grafico = False
    cmap = plt.get_cmap("viridis")
    n_fechas = len(LAGOS[nombre_lago]["fechas"])
    for i, fecha in enumerate(LAGOS[nombre_lago]["fechas"]):
        archivos = rutas_raster(nombre_lago, "cianobacteria", fecha)
        if not archivos:
            continue
        arreglo, _ = cargar_banda(archivos[0])
        valores = arreglo[~np.isnan(arreglo)]
        if valores.size == 0:
            continue
        ax.hist(
            valores,
            bins=40,
            alpha=0.45,
            label=fecha,
            color=cmap(i / max(1, n_fechas - 1)),
            density=True,
        )
        grafico = True
    if not grafico:
        return None
    ax.set_xlabel("Cya (10^3 celulas/ml)")
    ax.set_ylabel("Densidad")
    ax.set_title(f"Distribucion de Cya por fecha - {nombre_lago}")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    destino = (
        CARPETA_FIGURAS
        / "exploratorio"
        / nombre_lago.lower()
        / "histogramas_cya.png"
    )
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, bbox_inches="tight")
    plt.show()
    return destino


def boxplots_por_fecha(nombre_lago: str) -> Optional[Path]:
    """Ejercicio 8.3: boxplots lado a lado."""

    datos = []
    etiquetas = []
    for fecha in LAGOS[nombre_lago]["fechas"]:
        archivos = rutas_raster(nombre_lago, "cianobacteria", fecha)
        if not archivos:
            continue
        arreglo, _ = cargar_banda(archivos[0])
        valores = arreglo[~np.isnan(arreglo)]
        if valores.size == 0:
            continue
        datos.append(valores)
        etiquetas.append(fecha)
    if not datos:
        return None

    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=140)
    ax.boxplot(datos, labels=etiquetas, showfliers=False)
    ax.set_ylabel("Cya (10^3 celulas/ml)")
    ax.set_title(f"Distribucion de Cya por fecha - {nombre_lago}")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    destino = (
        CARPETA_FIGURAS
        / "exploratorio"
        / nombre_lago.lower()
        / "boxplots_cya.png"
    )
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, bbox_inches="tight")
    plt.show()
    return destino


def mapa_diferencia(nombre_lago: str, fecha_a: str, fecha_b: str) -> Optional[Path]:
    """Ejercicio 8.3: diferencia pixel a pixel entre dos fechas."""

    archivos_a = rutas_raster(nombre_lago, "cianobacteria", fecha_a)
    archivos_b = rutas_raster(nombre_lago, "cianobacteria", fecha_b)
    if not (archivos_a and archivos_b):
        return None
    arreglo_a, perfil_a = cargar_banda(archivos_a[0])
    arreglo_b, perfil_b = cargar_banda(archivos_b[0])
    if arreglo_a.shape != arreglo_b.shape:
        return None
    diferencia = arreglo_b - arreglo_a
    extent = bbox_desde_perfil(perfil_a)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=140)
    limite = float(np.nanpercentile(np.abs(diferencia), 98))
    if not np.isfinite(limite) or limite == 0:
        limite = 1.0
    im = ax.imshow(
        diferencia,
        cmap="RdBu_r",
        extent=extent,
        origin="upper",
        vmin=-limite,
        vmax=limite,
    )
    fig.colorbar(im, ax=ax, label=f"Cya {fecha_b} - Cya {fecha_a}")
    ax.set_title(
        f"{nombre_lago} - cambio entre {fecha_a} y {fecha_b}"
    )
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    fig.tight_layout()
    destino = (
        CARPETA_FIGURAS
        / "exploratorio"
        / nombre_lago.lower()
        / f"diferencia_{fecha_a}_{fecha_b}.png"
    )
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, bbox_inches="tight")
    plt.show()
    return destino


def patron_estacional(
    serie: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ejercicio 8.4: agrega la serie promedio por mes y por lago.

    Devuelve una tabla con el mes, el valor promedio y el conteo de
    observaciones. Si la tabla final solo tiene un mes o ninguno, devuelve un
    DataFrame vacio con la estructura esperada.
    """

    if serie.empty:
        return pd.DataFrame(columns=["lago", "mes", "valor_promedio", "n"])
    serie = serie.copy()
    serie["fecha"] = pd.to_datetime(serie["fecha"])
    serie["mes"] = serie["fecha"].dt.month
    return (
        serie.groupby(["lago", "mes"])["cianobacteria_promedio"]
        .agg(valor_promedio="mean", n="count")
        .reset_index()
    )


def comparacion_entre_lagos(
    serie: pd.DataFrame,
    extension: pd.DataFrame,
) -> Dict[str, float]:
    """
    Ejercicio 7.2: intensidad y frecuencia relativa de floracion entre lagos.

    Devuelve un diccionario con medias, maximos y porcentaje promedio de
    pixeles altos por lago.
    """

    resumen: Dict[str, float] = {}
    for lago, grupo in serie.groupby("lago"):
        valores = grupo["cianobacteria_promedio"].dropna()
        if valores.empty:
            continue
        resumen[f"{lago}_media"] = float(valores.mean())
        resumen[f"{lago}_maximo"] = float(valores.max())
        resumen[f"{lago}_mediana"] = float(valores.median())
    if not extension.empty:
        for lago, grupo in extension.groupby("lago"):
            porcentajes = grupo["porcentaje_floracion"].dropna()
            if porcentajes.empty:
                continue
            resumen[f"{lago}_floracion_promedio_pct"] = float(porcentajes.mean())
    return resumen


def resumen_ejecutivo(
    serie: pd.DataFrame,
    extension: pd.DataFrame,
    correlaciones: pd.DataFrame,
) -> str:
    """Emite un texto listo para pegar en el informe final."""

    lineas = ["## Resumen ejecutivo", ""]
    for lago, grupo in serie.groupby("lago"):
        valores = grupo.dropna(subset=["cianobacteria_promedio"])
        if valores.empty:
            continue
        pico = valores.loc[valores["cianobacteria_promedio"].idxmax()]
        minimo = valores.loc[valores["cianobacteria_promedio"].idxmin()]
        lineas.append(
            f"- **{lago}**: promedio {valores['cianobacteria_promedio'].mean():.2f}, "
            f"maximo {pico['cianobacteria_promedio']:.2f} el "
            f"{pico['fecha']:%d/%m/%Y}, minimo {minimo['cianobacteria_promedio']:.2f} "
            f"el {minimo['fecha']:%d/%m/%Y}."
        )
    if not extension.empty:
        lineas.append("")
        lineas.append("### Extension promedio de la floracion")
        for lago, grupo in extension.groupby("lago"):
            porcentajes = grupo["porcentaje_floracion"].dropna()
            if porcentajes.empty:
                continue
            lineas.append(
                f"- {lago}: {porcentajes.mean():.1f}% del area con Cya alta "
                f"en promedio."
            )
    if not correlaciones.empty:
        lineas.append("")
        lineas.append("### Correlaciones")
        for _, fila in correlaciones.iterrows():
            lineas.append(
                f"- {fila['lago']} - {fila['variable'].upper()} vs Cya: "
                f"r = {fila['pearson_r']:.2f} (p = {fila['pearson_p']:.2g}), "
                f"asociacion {fila['direccion']} {fila['fortaleza']}."
            )
    return "\n".join(lineas)


def ejecutar_analisis_completos(
    descargar_datos: bool = False,
    ruta_serie: Optional[Path] = None,
) -> Dict[str, object]:
    """
    Orquestador de los ejercicios 5 a 8.

    Si ``descargar_datos`` es True, ejecuta primero ``ejecutar_avance`` para
    regenerar los insumos. Cuando ya estan descargados, basta con pasar
    ``descargar_datos=False`` y la ruta al CSV de la serie temporal.
    """

    if descargar_datos:
        serie = ejecutar_avance(descargar_rasters=True)
    else:
        ruta = ruta_serie or (
            CARPETA_DATOS / "series_temporales" / "serie_temporal_cianobacteria.csv"
        )
        serie = (
            pd.read_csv(ruta)
            if ruta.exists()
            else pd.DataFrame(columns=["lago", "fecha", "cianobacteria_promedio"])
        )

    salidas: Dict[str, object] = {"serie": serie}

    # Mapas estaticos por lago
    mapas = {
        lago: mapear_cianobacteria(lago)
        for lago in LAGOS
        if hay_rasters(lago)
    }
    salidas["mapas_estaticos"] = mapas

    # Panel comparativo entre fechas para cada lago
    comparativos = {}
    for lago in LAGOS:
        comparativos[lago] = {}
        for indice in ("cianobacteria", "ndvi", "ndwi"):
            comparativos[lago][indice] = (
                comparar_fechas(lago, LAGOS[lago]["fechas"][:4], indice)
                if hay_rasters(lago, indice)
                else None
            )
    salidas["comparativos"] = comparativos

    # Tabla por pixel con los tres indices
    pixeles = pd.concat(
        [extraer_perfiles_por_pixel(lago) for lago in LAGOS],
        ignore_index=True,
    )
    salidas["pixeles"] = pixeles
    ruta_pixeles = CARPETA_DATOS / "pixeles" / "pixeles_indices.csv"
    ruta_pixeles.parent.mkdir(parents=True, exist_ok=True)
    pixeles.to_csv(ruta_pixeles, index=False)

    # Correlacion
    correlaciones = correlacion_con_cya(pixeles)
    salidas["correlaciones"] = correlaciones
    (CARPETA_DATOS / "correlaciones").mkdir(parents=True, exist_ok=True)
    correlaciones.to_csv(
        CARPETA_DATOS / "correlaciones" / "correlaciones_cya.csv",
        index=False,
    )
    for variable in ("ndvi", "ndwi"):
        graficar_dispersion(pixeles, variable)

    # Extension de la floracion
    extensiones = pd.concat(
        [extension_floracion(lago) for lago in LAGOS],
        ignore_index=True,
    )
    (CARPETA_DATOS / "extension").mkdir(parents=True, exist_ok=True)
    extensiones.to_csv(
        CARPETA_DATOS / "extension" / "extension_floracion.csv",
        index=False,
    )
    salidas["extensiones"] = extensiones

    # Zonas persistentes y figuras exploratorias
    for lago in LAGOS:
        if not hay_rasters(lago):
            continue
        graficar_zonas_persistentes(lago)
        histogramas_por_fecha(lago)
        boxplots_por_fecha(lago)
        fechas_validas = [
            f for f in LAGOS[lago]["fechas"]
            if rutas_raster(lago, "cianobacteria", f)
        ]
        if len(fechas_validas) >= 2:
            mapa_diferencia(lago, fechas_validas[0], fechas_validas[-1])

    # Patron estacional y comparacion entre lagos
    estacional = patron_estacional(serie)
    (CARPETA_DATOS / "estacional").mkdir(parents=True, exist_ok=True)
    estacional.to_csv(
        CARPETA_DATOS / "estacional" / "patron_estacional.csv",
        index=False,
    )
    salidas["estacional"] = estacional

    comparacion = comparacion_entre_lagos(serie, extensiones)
    salidas["comparacion"] = comparacion

    resumen = resumen_ejecutivo(serie, extensiones, correlaciones)
    CARPETA_FIGURAS.mkdir(parents=True, exist_ok=True)
    (CARPETA_FIGURAS / "resumen_ejecutivo.md").write_text(resumen)
    print(resumen)
    return salidas


if __name__ == "__main__":
    main()
