"""
Laboratorio 4, Parte 2. Ejercicios 1, 2 y 3.

Construye el conjunto de datos tabular para Machine Learning a partir de los
raster de la Parte I, define la variable respuesta binaria y separa los
predictores validos de los que producirian fuga de informacion.

El modulo no descarga nada por si mismo salvo que se llame
``descargar_bandas``. Todo lo demas trabaja sobre los GeoTIFF que ya estan en
``data/processed``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from src.procesamiento_geoespacial import (
    BANDAS_REFLECTANCIA,
    CARPETA_DATOS,
    CARPETA_FIGURAS,
    LAGOS,
    _importar_rasterio,
    cargar_banda,
    cargar_solo_fechas,
    rutas_raster,
    siguiente_dia,
)

# ---------------------------------------------------------------------------
# Ejercicio 2: punto de corte de la variable respuesta
# ---------------------------------------------------------------------------
# El script Se2WaQ entrega la densidad estimada de cianobacteria en unidades de
# 10^3 celulas por mililitro. La OMS usa 100 000 cel/mL como el umbral a partir
# del cual el riesgo para la salud en aguas recreativas deja de ser bajo y pasa
# a ser moderado (Alert Level 2 / Vigilance Level 2). En las unidades del raster
# ese valor son 100 unidades.
#
#   OMS (2003). Guidelines for safe recreational water environments, Vol. 1.
#   OMS (2021). Guidelines on recreational water quality, Vol. 1.
#   Chorus & Welch (2021). Toxic Cyanobacteria in Water, 2a ed.
#
UMBRAL_OMS_CEL_ML = 100_000.0
UMBRAL_CYA = UMBRAL_OMS_CEL_ML / 1_000.0  # = 100.0 en unidades del raster

# ---------------------------------------------------------------------------
# Ejercicio 3: fuga de informacion
# ---------------------------------------------------------------------------
# Cya = 115530.31 * ((B03 * B04) / B02) ** 2.38
# La respuesta se construye umbralando Cya, asi que Cya y las tres bandas que
# entran en su formula quedan prohibidas como predictoras. NDVI usa B04 y NDWI
# usa B03, de modo que ambos comparten insumo con la respuesta: se reportan en
# el dataset porque el enunciado los pide, pero no entran al conjunto estricto.
BANDAS_DE_LA_RESPUESTA = ["B02", "B03", "B04"]
COLUMNAS_PROHIBIDAS = ["cya", "cya_log", "alta_presencia"] + BANDAS_DE_LA_RESPUESTA
COLUMNAS_CONTAMINADAS = ["ndvi", "ndwi"]  # derivadas parciales de B03/B04

INDICES_RASTER = {"cya": "cianobacteria", "ndvi": "ndvi", "ndwi": "ndwi"}


# ---------------------------------------------------------------------------
# Ejercicio 1: construccion del dataset
# ---------------------------------------------------------------------------

def descargar_bandas(conexion, nombre_lago: str) -> None:
    """Guarda las bandas de reflectancia como GeoTIFF, una carpeta por fecha.

    La Parte I solo bajo NDVI, NDWI y Cya. Para tener predictoras espectrales
    reales hace falta la reflectancia, sobre todo B08, que es la unica banda
    del conjunto original que no participa en la formula de Cya.
    """

    datos = LAGOS[nombre_lago]
    cubo = cargar_solo_fechas(
        conexion, datos["bbox"], datos["fechas"], BANDAS_REFLECTANCIA
    )
    for fecha in datos["fechas"]:
        salida = CARPETA_DATOS / nombre_lago.lower() / "bandas" / fecha
        salida.mkdir(parents=True, exist_ok=True)
        trabajo = cubo.filter_temporal(fecha, siguiente_dia(fecha)).execute_batch(
            out_format="GTiff", title=f"Lab4P2 bandas {nombre_lago} {fecha}"
        )
        trabajo.get_results().download_files(salida)
        print(nombre_lago, fecha, "bandas", trabajo.job_id)


def _coordenadas(perfil: dict) -> tuple:
    """Devuelve las coordenadas del centro de cada pixel, aplanadas."""

    rio = _importar_rasterio()
    filas, columnas = np.indices((perfil["height"], perfil["width"]))
    xs, ys = rio.transform.xy(
        perfil["transform"], filas.ravel(), columnas.ravel(), offset="center"
    )
    return np.asarray(xs), np.asarray(ys)


def _tabla_de_una_fecha(nombre_lago: str, fecha: str) -> Optional[pd.DataFrame]:
    """Apila los raster de una fecha en una fila por pixel."""

    capas: Dict[str, np.ndarray] = {}
    perfil_base = None

    for columna, carpeta in INDICES_RASTER.items():
        rutas = rutas_raster(nombre_lago, carpeta, fecha)
        if not rutas:
            continue
        arreglo, perfil = cargar_banda(rutas[0])
        if perfil_base is None:
            perfil_base = perfil
        elif arreglo.shape != (perfil_base["height"], perfil_base["width"]):
            print(f"  aviso: {columna} de {fecha} no cuadra en malla, se omite")
            continue
        capas[columna] = arreglo.ravel()

    if perfil_base is None or "cya" not in capas:
        return None

    # Las bandas viven en un solo GeoTIFF multibanda, en el orden pedido.
    rutas_bandas = rutas_raster(nombre_lago, "bandas", fecha)
    if rutas_bandas:
        for indice, nombre in enumerate(BANDAS_REFLECTANCIA, start=1):
            arreglo, perfil = cargar_banda(rutas_bandas[0], banda=indice)
            if arreglo.shape != (perfil_base["height"], perfil_base["width"]):
                print(f"  aviso: {nombre} de {fecha} no cuadra en malla, se omite")
                break
            capas[nombre] = arreglo.ravel() * 0.0001  # DN -> reflectancia

    xs, ys = _coordenadas(perfil_base)
    tabla = pd.DataFrame(capas)
    tabla.insert(0, "lat", ys)
    tabla.insert(0, "lon", xs)
    tabla.insert(0, "fecha", pd.Timestamp(fecha))
    tabla.insert(0, "lago", nombre_lago)
    tabla["crs"] = str(perfil_base.get("crs", "EPSG:4326"))
    return tabla


def construir_dataset(
    lagos: Iterable[str] = ("Atitlan", "Amatitlan"),
    guardar: bool = True,
) -> pd.DataFrame:
    """Ejercicio 1.1 y 1.3: arma la tabla y descarta observaciones invalidas.

    Una fila por pixel valido dentro del lago. Se eliminan los pixeles sin dato
    (nubes, sombras y nieve ya venian enmascarados desde openEO), los que NDWI
    no reconoce como agua y los valores de Cya no finitos o negativos.
    """

    partes: List[pd.DataFrame] = []
    for lago in lagos:
        for fecha in LAGOS[lago]["fechas"]:
            tabla = _tabla_de_una_fecha(lago, fecha)
            if tabla is None:
                print(f"  pendiente: no hay raster de Cya para {lago} {fecha}")
                continue
            partes.append(tabla)

    if not partes:
        raise FileNotFoundError(
            "No se encontro ningun raster en data/processed. Ejecute primero la "
            "descarga de la Parte I."
        )

    crudo = pd.concat(partes, ignore_index=True)
    antes = len(crudo)

    limpio = crudo.dropna(subset=["cya"])
    limpio = limpio[np.isfinite(limpio["cya"])]
    limpio = limpio[limpio["cya"] >= 0]
    if "ndwi" in limpio.columns:
        # NDWI >= 0 deja solamente agua; fuera del lago el indice es negativo.
        limpio = limpio[limpio["ndwi"].notna() & (limpio["ndwi"] >= 0)]

    limpio = limpio.reset_index(drop=True)
    print(f"Pixeles leidos: {antes:,} -> validos: {len(limpio):,} "
          f"({antes - len(limpio):,} descartados)")

    if guardar:
        salida = CARPETA_DATOS / "dataset_ml.parquet"
        salida.parent.mkdir(parents=True, exist_ok=True)
        try:
            limpio.to_parquet(salida, index=False)
            print(f"Guardado en {salida}")
        except Exception:  # sin pyarrow instalado
            salida = salida.with_suffix(".csv")
            limpio.to_csv(salida, index=False)
            print(f"Guardado en {salida}")

    return limpio


def resumen_dataset(datos: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Ejercicio 1.4: totales, cobertura y calidad de cada variable."""

    variables = pd.DataFrame(
        {
            "tipo": datos.dtypes.astype(str),
            "faltantes": datos.isna().sum(),
            "pct_faltantes": (datos.isna().mean() * 100).round(2),
            "unicos": datos.nunique(),
        }
    )
    por_lago = datos.groupby("lago").size().rename("observaciones").to_frame()
    por_fecha = (
        datos.groupby(["lago", datos["fecha"].dt.date])
        .size()
        .rename("observaciones")
        .reset_index()
    )
    return {
        "total": pd.DataFrame({"observaciones": [len(datos)]}),
        "por_lago": por_lago,
        "por_fecha": por_fecha,
        "variables": variables,
    }


def eda(datos: pd.DataFrame, guardar: bool = True) -> Dict[str, pd.DataFrame]:
    """Ejercicio 1.5: estadisticas y figuras del analisis exploratorio."""

    import matplotlib.pyplot as plt

    numericas = datos.select_dtypes("number")
    descriptivas = numericas.describe().T
    correlaciones = numericas.corr(method="spearman")

    if guardar:
        carpeta = CARPETA_FIGURAS / "ml"
        carpeta.mkdir(parents=True, exist_ok=True)

        columnas = [c for c in ["cya", "ndvi", "ndwi", "B08"] if c in datos.columns]
        fig, ejes = plt.subplots(1, len(columnas), figsize=(4 * len(columnas), 3.4), dpi=140)
        for eje, columna in zip(np.atleast_1d(ejes), columnas):
            valores = datos[columna]
            if columna == "cya":
                valores = np.log10(valores.clip(lower=1e-3))
                eje.axvline(np.log10(UMBRAL_CYA), color="crimson", ls="--", lw=1)
                eje.set_xlabel("log10(Cya)")
            else:
                eje.set_xlabel(columna)
            eje.hist(valores.dropna(), bins=60, color="#2a9d8f")
        fig.suptitle("Distribucion de las variables principales")
        fig.tight_layout()
        fig.savefig(carpeta / "histogramas.png")
        plt.close(fig)

        fig, eje = plt.subplots(figsize=(6, 5), dpi=140)
        imagen = eje.imshow(correlaciones, cmap="RdBu_r", vmin=-1, vmax=1)
        eje.set_xticks(range(len(correlaciones)), correlaciones.columns, rotation=90)
        eje.set_yticks(range(len(correlaciones)), correlaciones.columns)
        fig.colorbar(imagen, label="Spearman")
        eje.set_title("Correlacion entre variables")
        fig.tight_layout()
        fig.savefig(carpeta / "correlaciones.png")
        plt.close(fig)

        fig, ejes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=140)
        for eje, lago in zip(ejes, datos["lago"].unique()):
            sub = datos[datos["lago"] == lago]
            puntos = eje.scatter(
                sub["lon"], sub["lat"],
                c=np.log10(sub["cya"].clip(lower=1e-3)),
                s=1, cmap="viridis",
            )
            eje.set_title(lago)
            fig.colorbar(puntos, ax=eje, label="log10(Cya)")
        fig.suptitle("Distribucion espacial de las observaciones")
        fig.tight_layout()
        fig.savefig(carpeta / "dispersion_espacial.png")
        plt.close(fig)
        print(f"Figuras en {carpeta}")

    return {"descriptivas": descriptivas, "correlaciones": correlaciones}


# ---------------------------------------------------------------------------
# Ejercicio 2: variable respuesta
# ---------------------------------------------------------------------------

def agregar_respuesta(datos: pd.DataFrame, umbral: float = UMBRAL_CYA) -> pd.DataFrame:
    """Ejercicio 2.1: variable binaria de alta presencia de cianobacteria."""

    salida = datos.copy()
    salida["alta_presencia"] = (salida["cya"] >= umbral).astype(int)
    salida["cya_log"] = np.log10(salida["cya"].clip(lower=1e-3))
    return salida


def distribucion_respuesta(datos: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Ejercicio 2.3 y 2.4: reparto de clases global, por lago y por fecha."""

    def _tasa(grupo):
        return pd.Series(
            {
                "n": len(grupo),
                "altas": int(grupo["alta_presencia"].sum()),
                "pct_altas": round(grupo["alta_presencia"].mean() * 100, 2),
            }
        )

    global_ = _tasa(datos).to_frame("valor")
    por_lago = datos.groupby("lago").apply(_tasa, include_groups=False)
    por_fecha = (
        datos.groupby(["lago", datos["fecha"].dt.date])
        .apply(_tasa, include_groups=False)
        .reset_index()
    )

    positivos = datos["alta_presencia"].mean()
    razon = (1 - positivos) / positivos if 0 < positivos < 1 else np.inf
    print(f"Clase positiva: {positivos:.2%}  |  razon negativos:positivos = {razon:.1f}:1")
    if positivos < 0.10 or positivos > 0.90:
        print("Hay desbalance marcado: use F1/ROC-AUC y class_weight, no Accuracy.")

    return {"global": global_, "por_lago": por_lago, "por_fecha": por_fecha}


def sensibilidad_umbral(datos: pd.DataFrame) -> pd.DataFrame:
    """Muestra como cambia el balance con otros cortes posibles."""

    candidatos = {
        "OMS riesgo bajo (20 000 cel/mL)": 20.0,
        "OMS riesgo moderado (100 000 cel/mL)": UMBRAL_CYA,
        "Percentil 75 observado": float(datos["cya"].quantile(0.75)),
        "Percentil 90 observado": float(datos["cya"].quantile(0.90)),
    }
    return pd.DataFrame(
        [
            {
                "criterio": nombre,
                "umbral": round(valor, 2),
                "pct_altas": round((datos["cya"] >= valor).mean() * 100, 2),
            }
            for nombre, valor in candidatos.items()
        ]
    )


# ---------------------------------------------------------------------------
# Ejercicio 3: predictores
# ---------------------------------------------------------------------------

def ingenieria_caracteristicas(datos: pd.DataFrame) -> pd.DataFrame:
    """Ejercicio 3.3: variables temporales y espaciales derivadas.

    Ninguna usa Cya ni las bandas de su formula.
    """

    salida = datos.copy()
    salida["mes"] = salida["fecha"].dt.month
    salida["dia_del_anio"] = salida["fecha"].dt.dayofyear
    # En Guatemala la epoca lluviosa va de mayo a octubre; el aporte de
    # nutrientes por escorrentia cambia entre ambas epocas.
    salida["epoca_lluviosa"] = salida["mes"].between(5, 10).astype(int)
    salida["es_amatitlan"] = (salida["lago"] == "Amatitlan").astype(int)

    # Distancia relativa al centroide del lago: aproxima la cercania a la
    # orilla, donde entran los rios y se acumula la floracion.
    centros = salida.groupby("lago")[["lon", "lat"]].transform("mean")
    salida["dist_centro"] = np.hypot(
        salida["lon"] - centros["lon"], salida["lat"] - centros["lat"]
    )
    return salida


PREDICTORES_BASE = [
    "B08", "mes", "dia_del_anio", "epoca_lluviosa",
    "es_amatitlan", "lon", "lat", "dist_centro",
]


def predictores(datos: pd.DataFrame, estricto: bool = True) -> List[str]:
    """Devuelve las columnas usables como predictoras.

    ``estricto=True`` excluye NDVI y NDWI porque comparten bandas con la
    formula de Cya. Con ``estricto=False`` se agregan, para poder medir cuanto
    desempeno viene de esa dependencia.
    """

    candidatas = list(PREDICTORES_BASE)
    if not estricto:
        candidatas += COLUMNAS_CONTAMINADAS
    return [
        c for c in candidatas
        if c in datos.columns and c not in COLUMNAS_PROHIBIDAS
    ]


def descripcion_predictores(datos: pd.DataFrame) -> pd.DataFrame:
    """Ejercicio 3.2: que representa cada variable y por que se incluye."""

    catalogo = {
        "B08": ("Banda espectral", "Reflectancia en infrarrojo cercano. El agua limpia lo absorbe casi por completo, asi que valores altos delatan material particulado o biomasa flotante. Es la unica banda original ajena a la formula de Cya."),
        "ndvi": ("Indice", "Vigor de la vegetacion. Sobre agua, valores altos indican biomasa flotante. Comparte B04 con la formula de Cya."),
        "ndwi": ("Indice", "Separa agua de tierra y responde a turbidez. Comparte B03 con la formula de Cya."),
        "mes": ("Temporal", "Captura la estacionalidad de las floraciones."),
        "dia_del_anio": ("Temporal", "Version continua del mes, evita el corte artificial entre diciembre y enero."),
        "epoca_lluviosa": ("Temporal", "Mayo a octubre. La escorrentia arrastra nutrientes hacia el lago."),
        "es_amatitlan": ("Categorica", "Distingue los dos lagos, que difieren en profundidad, carga de nutrientes y altitud."),
        "lon": ("Espacial", "Posicion este-oeste de la observacion."),
        "lat": ("Espacial", "Posicion norte-sur de la observacion."),
        "dist_centro": ("Espacial", "Distancia al centroide del lago. Las floraciones se concentran cerca de la orilla y de las desembocaduras."),
    }
    disponibles = predictores(datos, estricto=False)
    return pd.DataFrame(
        [
            {"variable": v, "tipo": catalogo[v][0], "justificacion": catalogo[v][1],
             "en_conjunto_estricto": v not in COLUMNAS_CONTAMINADAS}
            for v in disponibles if v in catalogo
        ]
    )


def revisar_fuga(datos: pd.DataFrame, columnas: Iterable[str]) -> pd.DataFrame:
    """Ejercicio 2.5: deja explicito que se excluye y por que."""

    motivos = {
        "cya": "Es la variable con la que se construyo la respuesta.",
        "cya_log": "Transformacion directa de Cya.",
        "alta_presencia": "Es la respuesta.",
        "B02": "Entra en el denominador de la formula de Cya.",
        "B03": "Entra en el numerador de la formula de Cya.",
        "B04": "Entra en el numerador de la formula de Cya.",
        "ndvi": "Se calcula con B04, que participa en la formula de Cya.",
        "ndwi": "Se calcula con B03, que participa en la formula de Cya.",
    }
    usadas = set(columnas)
    filas = [
        {"variable": v, "motivo": m, "esta_en_uso": v in usadas}
        for v, m in motivos.items() if v in datos.columns
    ]
    tabla = pd.DataFrame(filas)
    problemas = tabla[tabla["esta_en_uso"] & tabla["variable"].isin(COLUMNAS_PROHIBIDAS)]
    if not problemas.empty:
        raise ValueError(f"Fuga de informacion: {list(problemas['variable'])}")
    return tabla


def demo() -> None:
    """Chequeo minimo con datos sinteticos, no necesita raster."""

    n = 400
    generador = np.random.default_rng(0)
    falso = pd.DataFrame(
        {
            "lago": ["Atitlan"] * (n // 2) + ["Amatitlan"] * (n // 2),
            "fecha": pd.to_datetime(["2026-02-02"] * (n // 2) + ["2026-06-19"] * (n // 2)),
            "lon": generador.uniform(-91.3, -90.5, n),
            "lat": generador.uniform(14.4, 14.75, n),
            "cya": generador.lognormal(3.5, 1.5, n),
            "ndvi": generador.uniform(-0.3, 0.4, n),
            "ndwi": generador.uniform(0.0, 0.6, n),
            "B08": generador.uniform(0.0, 0.2, n),
        }
    )

    con_respuesta = agregar_respuesta(falso)
    assert set(con_respuesta["alta_presencia"].unique()) <= {0, 1}
    assert (con_respuesta.loc[con_respuesta["cya"] >= UMBRAL_CYA, "alta_presencia"] == 1).all()
    assert (con_respuesta.loc[con_respuesta["cya"] < UMBRAL_CYA, "alta_presencia"] == 0).all()

    listo = ingenieria_caracteristicas(con_respuesta)
    estrictos = predictores(listo, estricto=True)
    assert "ndvi" not in estrictos and "ndwi" not in estrictos
    assert not set(estrictos) & set(COLUMNAS_PROHIBIDAS)
    assert "B08" in estrictos and "dist_centro" in estrictos

    amplios = predictores(listo, estricto=False)
    assert "ndvi" in amplios and "B03" not in amplios

    revisar_fuga(listo, estrictos)
    try:
        revisar_fuga(listo, estrictos + ["cya"])
    except ValueError:
        pass
    else:
        raise AssertionError("revisar_fuga debio rechazar cya como predictor")

    resumen = resumen_dataset(listo)
    assert resumen["por_lago"]["observaciones"].sum() == n
    reparto = distribucion_respuesta(listo)
    assert reparto["por_fecha"]["n"].sum() == n
    assert len(sensibilidad_umbral(listo)) == 4
    assert len(descripcion_predictores(listo)) >= len(estrictos)
    print("demo ok")


if __name__ == "__main__":
    demo()
