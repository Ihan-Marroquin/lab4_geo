"""Analisis completo del Laboratorio 4 con escenas Sentinel-2 L2A.

El modulo consulta las fechas oficiales por medio del catalogo STAC de
Microsoft Planetary Computer y lee solo las ventanas de los lagos. No descarga
escenas completas. Los indices y criterios siguen el avance construido con
openEO y el script Se2WaQ de Sentinel Hub.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import planetary_computer as pc
from pystac_client import Client
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from scipy.stats import pearsonr, spearmanr

from src.procesamiento_geoespacial import LAGOS


RAIZ = Path(__file__).resolve().parents[1]
CARPETA_RESULTADOS = RAIZ / "data" / "processed" / "resultados"
CARPETA_FIGURAS = RAIZ / "data" / "processed" / "figuras"
CRS_ANALISIS = "EPSG:32615"
RESOLUCION_METROS = 120
UMBRAL_CYA_ALTO = 20.0
CLASES_SCL_INVALIDAS = {0, 1, 3, 8, 9, 10, 11}
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
SATELITES_OFICIALES = {
    "Amatitlan": ["S2B", "S2A", "S2B", "S2B", "S2C", "S2B", "S2C", "S2C", "S2B", "S2C", "S2A"],
    "Atitlan": ["S2B", "S2C", "S2C", "S2A", "S2A", "S2C", "S2B", "S2B", "S2B", "S2C", "S2B"],
}


def _bbox(nombre_lago: str) -> list[float]:
    caja = LAGOS[nombre_lago]["bbox"]
    return [caja["west"], caja["south"], caja["east"], caja["north"]]


def _area_interseccion(a: list[float], b: list[float]) -> float:
    ancho = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    alto = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ancho * alto


def buscar_escena(catalogo: Client, nombre_lago: str, fecha: str):
    """Devuelve la tesela que cubre mejor el rectangulo del lago."""

    caja = _bbox(nombre_lago)
    busqueda = catalogo.search(
        collections=["sentinel-2-l2a"],
        bbox=caja,
        datetime=f"{fecha}/{fecha}",
        max_items=20,
    )
    escenas = list(busqueda.items())
    if not escenas:
        raise RuntimeError(f"No se encontro Sentinel-2 para {nombre_lago} el {fecha}.")

    posicion = LAGOS[nombre_lago]["fechas"].index(fecha)
    satelite = SATELITES_OFICIALES[nombre_lago][posicion]
    escenas_oficiales = [item for item in escenas if item.id.startswith(satelite)]
    if escenas_oficiales:
        escenas = escenas_oficiales

    escenas.sort(
        key=lambda item: (
            -_area_interseccion(caja, list(item.bbox)),
            float(item.properties.get("eo:cloud_cover", 999.0)),
        )
    )
    return pc.sign(escenas[0])


def crear_grilla(nombre_lago: str) -> dict:
    oeste, sur, este, norte = transform_bounds(
        "EPSG:4326", CRS_ANALISIS, *_bbox(nombre_lago), densify_pts=21
    )
    oeste = math.floor(oeste / RESOLUCION_METROS) * RESOLUCION_METROS
    sur = math.floor(sur / RESOLUCION_METROS) * RESOLUCION_METROS
    este = math.ceil(este / RESOLUCION_METROS) * RESOLUCION_METROS
    norte = math.ceil(norte / RESOLUCION_METROS) * RESOLUCION_METROS
    ancho = int(round((este - oeste) / RESOLUCION_METROS))
    alto = int(round((norte - sur) / RESOLUCION_METROS))
    return {
        "transform": from_origin(oeste, norte, RESOLUCION_METROS, RESOLUCION_METROS),
        "width": ancho,
        "height": alto,
        "extent": [oeste, este, sur, norte],
    }


def _leer_banda(url: str, grilla: dict, *, categorica: bool = False) -> np.ndarray:
    metodo = Resampling.nearest if categorica else Resampling.bilinear
    with rasterio.Env(
        GDAL_HTTP_MULTIRANGE="YES",
        GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
    ):
        with rasterio.open(url) as fuente:
            with WarpedVRT(
                fuente,
                crs=CRS_ANALISIS,
                transform=grilla["transform"],
                width=grilla["width"],
                height=grilla["height"],
                resampling=metodo,
                nodata=0,
            ) as vrt:
                return vrt.read(1).astype("float32")


def _reflectancia(datos: np.ndarray, baseline: str | float | None) -> np.ndarray:
    """Aplica la escala y el offset de Sentinel-2 L2A."""

    try:
        baseline_num = float(baseline)
    except (TypeError, ValueError):
        baseline_num = 4.0
    offset = -0.1 if baseline_num >= 4.0 else 0.0
    salida = datos * 0.0001 + offset
    salida[datos == 0] = np.nan
    return salida


def procesar_escena(item, grilla: dict) -> dict[str, np.ndarray]:
    nombres = {"azul": "B02", "verde": "B03", "rojo": "B04", "nir": "B08"}
    baseline = item.properties.get("s2:processing_baseline")
    bandas = {
        nombre: _reflectancia(_leer_banda(item.assets[clave].href, grilla), baseline)
        for nombre, clave in nombres.items()
    }
    scl = _leer_banda(item.assets["SCL"].href, grilla, categorica=True)

    azul, verde = bandas["azul"], bandas["verde"]
    rojo, nir = bandas["rojo"], bandas["nir"]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ndvi = (nir - rojo) / (nir + rojo)
        ndwi = (verde - nir) / (verde + nir)
        cya = 115530.31 * ((verde * rojo) / azul) ** 2.38

    calidad = ~np.isin(scl.astype("int16"), list(CLASES_SCL_INVALIDAS))
    agua = ndwi >= 0
    reflectancia_valida = (azul > 0) & (verde > 0) & (rojo > 0) & (nir > 0)
    valido = calidad & agua & reflectancia_valida & np.isfinite(cya)

    ndvi = np.where(valido, ndvi, np.nan).astype("float32")
    ndwi = np.where(valido, ndwi, np.nan).astype("float32")
    # La escala visual del script Se2WaQ llega a 100. Los valores superiores
    # se conservan como categoria saturada para evitar que pocos pixeles
    # inestables dominen el promedio del lago.
    cya = np.where(valido, np.clip(cya, 0, 100), np.nan).astype("float32")
    return {"NDVI": ndvi, "NDWI": ndwi, "CYA": cya, "VALIDO": valido}


def _guardar_cubo(nombre_lago: str, fechas: list[str], cubos: dict, grilla: dict) -> Path:
    destino = CARPETA_RESULTADOS / f"cubo_{nombre_lago.lower()}.npz"
    np.savez_compressed(
        destino,
        fechas=np.array(fechas),
        ndvi=np.stack(cubos["NDVI"]),
        ndwi=np.stack(cubos["NDWI"]),
        cya=np.stack(cubos["CYA"]),
        valido=np.stack(cubos["VALIDO"]),
        extent=np.array(grilla["extent"]),
        resolucion=np.array([RESOLUCION_METROS]),
        crs=np.array([CRS_ANALISIS]),
    )
    return destino


def descargar_y_calcular() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Procesa las 22 fechas y crea tablas y cubos compactos."""

    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    catalogo = Client.open(STAC_URL)
    filas, muestras = [], []
    rng = np.random.default_rng(3084)

    for nombre_lago, datos_lago in LAGOS.items():
        grilla = crear_grilla(nombre_lago)
        cubos = {clave: [] for clave in ("NDVI", "NDWI", "CYA", "VALIDO")}
        for fecha in datos_lago["fechas"]:
            item = buscar_escena(catalogo, nombre_lago, fecha)
            capas = procesar_escena(item, grilla)
            for clave in cubos:
                cubos[clave].append(capas[clave])

            validos = np.isfinite(capas["CYA"])
            total = int(validos.sum())
            if total == 0:
                raise RuntimeError(f"No quedaron pixeles de agua para {nombre_lago} {fecha}.")
            cya = capas["CYA"][validos]
            ndvi = capas["NDVI"][validos]
            ndwi = capas["NDWI"][validos]
            filas.append(
                {
                    "lago": nombre_lago,
                    "fecha": fecha,
                    "escena": item.id,
                    "nubosidad_escena_pct": float(item.properties.get("eo:cloud_cover", np.nan)),
                    "pixeles_agua_validos": total,
                    "ndvi_promedio": float(np.nanmean(ndvi)),
                    "ndwi_promedio": float(np.nanmean(ndwi)),
                    "cianobacteria_promedio": float(np.nanmean(cya)),
                    "cianobacteria_mediana": float(np.nanmedian(cya)),
                    "cianobacteria_maximo": float(np.nanmax(cya)),
                    "area_cya_alta_pct": float(np.mean(cya >= UMBRAL_CYA_ALTO) * 100),
                }
            )

            cantidad = min(3000, total)
            indices = rng.choice(total, size=cantidad, replace=False)
            muestras.extend(
                {
                    "lago": nombre_lago,
                    "fecha": fecha,
                    "ndvi": float(ndvi[i]),
                    "ndwi": float(ndwi[i]),
                    "cya": float(cya[i]),
                }
                for i in indices
            )
            print(f"{nombre_lago} {fecha}: {total} pixeles validos - {item.id}", flush=True)

        _guardar_cubo(nombre_lago, datos_lago["fechas"], cubos, grilla)

    metricas = pd.DataFrame(filas)
    muestra = pd.DataFrame(muestras)
    metricas.to_csv(CARPETA_RESULTADOS / "metricas_por_fecha.csv", index=False)
    muestra.to_csv(CARPETA_RESULTADOS / "muestra_correlaciones.csv", index=False)
    return metricas, muestra


def _cargar_cubo(nombre_lago: str) -> dict:
    datos = np.load(CARPETA_RESULTADOS / f"cubo_{nombre_lago.lower()}.npz")
    return {clave: datos[clave] for clave in datos.files}


def calcular_correlaciones(muestra: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for lago, grupo in muestra.groupby("lago"):
        for indice in ("ndvi", "ndwi"):
            pearson_r, pearson_p = pearsonr(grupo[indice], grupo["cya"])
            spearman_r, spearman_p = spearmanr(grupo[indice], grupo["cya"])
            filas.append(
                {
                    "lago": lago,
                    "indice": indice.upper(),
                    "n": len(grupo),
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                    "spearman_r": spearman_r,
                    "spearman_p": spearman_p,
                }
            )
    salida = pd.DataFrame(filas)
    salida.to_csv(CARPETA_RESULTADOS / "correlaciones_indices.csv", index=False)
    return salida


def resumir_lagos(metricas: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for lago, grupo in metricas.groupby("lago"):
        grupo = grupo.sort_values("fecha")
        q1, q3 = grupo["cianobacteria_promedio"].quantile([0.25, 0.75])
        umbral_pico = q3 + 1.5 * (q3 - q1)
        pico = grupo.loc[grupo["cianobacteria_promedio"].idxmax()]
        filas.append(
            {
                "lago": lago,
                "promedio_periodo": grupo["cianobacteria_promedio"].mean(),
                "mediana_periodo": grupo["cianobacteria_promedio"].median(),
                "fecha_maxima": pico["fecha"],
                "valor_maximo": pico["cianobacteria_promedio"],
                "fechas_pico_iqr": int((grupo["cianobacteria_promedio"] > umbral_pico).sum()),
                "area_alta_promedio_pct": grupo["area_cya_alta_pct"].mean(),
            }
        )
    salida = pd.DataFrame(filas)
    salida.to_csv(CARPETA_RESULTADOS / "resumen_lagos.csv", index=False)
    return salida


def calcular_estacionalidad(metricas: pd.DataFrame) -> pd.DataFrame:
    tabla = metricas.copy()
    tabla["mes"] = pd.to_datetime(tabla["fecha"]).dt.month
    tabla["temporada"] = np.where(tabla["mes"].between(5, 10), "Lluviosa", "Seca")
    salida = (
        tabla.groupby(["lago", "temporada"], as_index=False)
        .agg(
            observaciones=("fecha", "count"),
            cya_promedio=("cianobacteria_promedio", "mean"),
            area_alta_promedio_pct=("area_cya_alta_pct", "mean"),
        )
    )
    salida.to_csv(CARPETA_RESULTADOS / "estacionalidad.csv", index=False)
    return salida


def _estilo_figura():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
        }
    )


def graficar_temporal(metricas: pd.DataFrame) -> Path:
    colores = {"Atitlan": "#249C92", "Amatitlan": "#F06D4F"}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 7.2), dpi=160, sharex=True)
    for lago, grupo in metricas.groupby("lago"):
        grupo = grupo.sort_values("fecha").copy()
        grupo["fecha"] = pd.to_datetime(grupo["fecha"])
        color = colores[lago]
        ax1.plot(grupo["fecha"], grupo["cianobacteria_promedio"], "o-", lw=2, color=color, label=lago)
        ax2.plot(grupo["fecha"], grupo["area_cya_alta_pct"], "o-", lw=2, color=color, label=lago)
    ax1.set_ylabel("Cya promedio (0-100)")
    ax1.set_title("Evolucion temporal de la señal de cianobacteria")
    ax1.legend(frameon=False)
    ax2.set_ylabel("Area con Cya >= 20 (%)")
    ax2.set_xlabel("Fecha")
    ax2.grid(alpha=0.25)
    ax1.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    ruta = CARPETA_FIGURAS / "evolucion_temporal.png"
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    return ruta


def graficar_mapas(nombre_lago: str, cubo: dict) -> Path:
    fechas = [str(x) for x in cubo["fechas"]]
    cya = cubo["cya"]
    extent = cubo["extent"].tolist()
    fig, ejes = plt.subplots(3, 4, figsize=(13, 9.5), dpi=150, constrained_layout=True)
    imagen = None
    for i, ax in enumerate(ejes.flat):
        if i >= len(fechas):
            ax.axis("off")
            continue
        imagen = ax.imshow(cya[i], cmap="turbo", vmin=0, vmax=100, extent=extent, origin="upper")
        ax.set_title(fechas[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Distribucion de cianobacteria - Lago {nombre_lago}", fontsize=16, fontweight="bold")
    if imagen is not None:
        barra = fig.colorbar(imagen, ax=ejes, shrink=0.72, pad=0.02)
        barra.set_label("Indice Cya (0-100)")
    ruta = CARPETA_FIGURAS / f"mapas_cianobacteria_{nombre_lago.lower()}.png"
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    return ruta


def graficar_persistencia(nombre_lago: str, cubo: dict) -> Path:
    cya = cubo["cya"]
    validos = np.isfinite(cya)
    conteo = validos.sum(axis=0)
    persistencia = np.divide(
        (cya >= UMBRAL_CYA_ALTO).sum(axis=0) * 100.0,
        conteo,
        out=np.full(conteo.shape, np.nan, dtype="float32"),
        where=conteo > 0,
    )
    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=160)
    im = ax.imshow(persistencia, cmap="magma", vmin=0, vmax=100, extent=cubo["extent"].tolist(), origin="upper")
    ax.set_title(f"Persistencia de valores altos - Lago {nombre_lago}")
    ax.set_xlabel("Este UTM (m)")
    ax.set_ylabel("Norte UTM (m)")
    barra = fig.colorbar(im, ax=ax)
    barra.set_label("Fechas con Cya >= 20 (%)")
    fig.tight_layout()
    ruta = CARPETA_FIGURAS / f"persistencia_{nombre_lago.lower()}.png"
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    return ruta


def graficar_diferencia(nombre_lago: str, cubo: dict) -> Path:
    diferencia = cubo["cya"][-1] - cubo["cya"][0]
    limite = float(np.nanpercentile(np.abs(diferencia), 95))
    limite = max(limite, 1.0)
    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=160)
    im = ax.imshow(
        diferencia,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limite, vcenter=0, vmax=limite),
        extent=cubo["extent"].tolist(),
        origin="upper",
    )
    ax.set_title(f"Cambio entre primera y ultima fecha - Lago {nombre_lago}")
    ax.set_xlabel("Este UTM (m)")
    ax.set_ylabel("Norte UTM (m)")
    barra = fig.colorbar(im, ax=ax)
    barra.set_label("Cambio en Cya")
    fig.tight_layout()
    ruta = CARPETA_FIGURAS / f"diferencia_{nombre_lago.lower()}.png"
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    return ruta


def graficar_distribuciones(nombre_lago: str, cubo: dict) -> Path:
    fechas = [str(x)[5:] for x in cubo["fechas"]]
    grupos = [capa[np.isfinite(capa)] for capa in cubo["cya"]]
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=160)
    partes = ax.violinplot(grupos, showmeans=False, showmedians=True, widths=0.8)
    for cuerpo in partes["bodies"]:
        cuerpo.set_facecolor("#249C92")
        cuerpo.set_edgecolor("#17645F")
        cuerpo.set_alpha(0.65)
    ax.set_xticks(range(1, len(fechas) + 1), fechas, rotation=45, ha="right")
    ax.set_ylim(0, 102)
    ax.set_ylabel("Indice Cya (0-100)")
    ax.set_title(f"Distribucion por fecha - Lago {nombre_lago}")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    ruta = CARPETA_FIGURAS / f"distribuciones_{nombre_lago.lower()}.png"
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    return ruta


def graficar_correlaciones(nombre_lago: str, muestra: pd.DataFrame) -> Path:
    datos = muestra[muestra["lago"] == nombre_lago]
    fig, ejes = plt.subplots(1, 2, figsize=(12.2, 4.7), dpi=160, sharey=True)
    for ax, indice, color in zip(ejes, ("ndvi", "ndwi"), ("#5B8E3E", "#3678A8")):
        ax.hexbin(datos[indice], datos["cya"], gridsize=45, mincnt=1, cmap="viridis")
        r, _ = spearmanr(datos[indice], datos["cya"])
        ax.set_title(f"{indice.upper()} vs. Cya - rho={r:.2f}", fontsize=13)
        ax.set_xlabel(indice.upper())
        ax.grid(alpha=0.15)
    ejes[0].set_ylabel("Indice Cya (0-100)")
    fig.suptitle(f"Relacion entre indices - Lago {nombre_lago}", fontweight="bold", y=1.02)
    fig.tight_layout(w_pad=2.5)
    ruta = CARPETA_FIGURAS / f"correlaciones_{nombre_lago.lower()}.png"
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    return ruta


def crear_figuras(metricas: pd.DataFrame, muestra: pd.DataFrame) -> list[Path]:
    CARPETA_FIGURAS.mkdir(parents=True, exist_ok=True)
    _estilo_figura()
    rutas = [graficar_temporal(metricas)]
    for lago in LAGOS:
        cubo = _cargar_cubo(lago)
        rutas.extend(
            [
                graficar_mapas(lago, cubo),
                graficar_persistencia(lago, cubo),
                graficar_diferencia(lago, cubo),
                graficar_distribuciones(lago, cubo),
                graficar_correlaciones(lago, muestra),
            ]
        )
    return rutas


def generar_hallazgos(
    metricas: pd.DataFrame,
    correlaciones: pd.DataFrame,
    resumen: pd.DataFrame,
    estacionalidad: pd.DataFrame,
) -> dict:
    hallazgos = {"lagos": {}, "comparacion": "", "cautela": ""}
    for lago in LAGOS:
        serie = metricas[metricas["lago"] == lago].sort_values("fecha")
        pico = serie.loc[serie["cianobacteria_promedio"].idxmax()]
        persistente = serie.loc[serie["area_cya_alta_pct"].idxmax()]
        corr = correlaciones[correlaciones["lago"] == lago].set_index("indice")
        hallazgos["lagos"][lago] = {
            "pico_fecha": str(pico["fecha"]),
            "pico_valor": round(float(pico["cianobacteria_promedio"]), 2),
            "mayor_extension_fecha": str(persistente["fecha"]),
            "mayor_extension_pct": round(float(persistente["area_cya_alta_pct"]), 2),
            "rho_ndvi": round(float(corr.loc["NDVI", "spearman_r"]), 3),
            "rho_ndwi": round(float(corr.loc["NDWI", "spearman_r"]), 3),
        }
    orden = resumen.sort_values("promedio_periodo", ascending=False)["lago"].tolist()
    hallazgos["comparacion"] = (
        f"{orden[0]} presenta el promedio temporal mas alto en la escala analizada; "
        "la intensidad y la extension se interpretan por separado para evitar confundir "
        "un foco localizado con una floracion extendida."
    )
    hallazgos["cautela"] = (
        "Las fechas no forman una serie mensual regular y cubren poco mas de un anio. "
        "Las diferencias entre epoca seca y lluviosa son descriptivas, no una prueba causal."
    )
    destino = CARPETA_RESULTADOS / "hallazgos.json"
    destino.write_text(json.dumps(hallazgos, ensure_ascii=False, indent=2), encoding="utf-8")
    return hallazgos


def ejecutar_pipeline(forzar_descarga: bool = False) -> dict:
    """Ejecuta los ejercicios 2 al 8 y devuelve todas las tablas."""

    archivo_metricas = CARPETA_RESULTADOS / "metricas_por_fecha.csv"
    archivo_muestra = CARPETA_RESULTADOS / "muestra_correlaciones.csv"
    if forzar_descarga or not archivo_metricas.exists() or not archivo_muestra.exists():
        metricas, muestra = descargar_y_calcular()
    else:
        metricas = pd.read_csv(archivo_metricas)
        muestra = pd.read_csv(archivo_muestra)

    correlaciones = calcular_correlaciones(muestra)
    resumen = resumir_lagos(metricas)
    estacionalidad = calcular_estacionalidad(metricas)
    figuras = crear_figuras(metricas, muestra)
    hallazgos = generar_hallazgos(metricas, correlaciones, resumen, estacionalidad)
    return {
        "metricas": metricas,
        "muestra": muestra,
        "correlaciones": correlaciones,
        "resumen": resumen,
        "estacionalidad": estacionalidad,
        "figuras": figuras,
        "hallazgos": hallazgos,
    }


if __name__ == "__main__":
    ejecutar_pipeline(forzar_descarga=True)
