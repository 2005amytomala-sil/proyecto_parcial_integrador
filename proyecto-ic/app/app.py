"""Prototipo Streamlit para detección binaria con UNSW-NB15.

Permite cargar y revisar un CSV aunque el modelo final aún no esté disponible.
Cuando exista models/pipeline_final.joblib, la aplicación podrá utilizarlo para predecir.
"""

from pathlib import Path
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
import joblib
import pandas as pd
import streamlit as st


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "pipeline_final.joblib"

TARGET = "label"
NON_FEATURES = {"label", "attack_cat"}

st.set_page_config(
    page_title="Detector de intrusiones | UNSW-NB15",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --lila-oscuro: #5b3b91;
        --lila: #7956b5;
        --lila-claro: #e8e1f4;
        --tinta: #29243b;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 8% 5%, rgba(255, 255, 255, 0.62) 0, transparent 30rem),
            radial-gradient(ellipse at 94% 80%, rgba(160, 143, 209, 0.35) 0, transparent 34rem),
            linear-gradient(135deg, #c4d9f2 0%, #d8d8f0 48%, #cbd9ee 100%);
        color: var(--tinta);
    }

    [data-testid="stHeader"] {
        position: relative !important;
        top: auto !important;
        z-index: 1;
        height: 3rem;
        background: rgba(205, 218, 239, 0.76);
    }

    [data-testid="stMainBlockContainer"] {
        width: 100% !important;
        max-width: none !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

    .block-container {
        width: min(97vw, 1600px) !important;
        max-width: 1600px !important;
        box-sizing: border-box;
        margin: 1.5rem auto 2.5rem !important;
        padding: 2.1rem 2.4rem 3.2rem;
        border: 1px solid rgba(203, 184, 231, 0.55);
        border-radius: 26px;
        background: linear-gradient(145deg, #29263d 0%, #332c49 58%, #29263d 100%);
        box-shadow:
            0 24px 70px rgba(48, 57, 91, 0.19),
            0 5px 16px rgba(72, 59, 111, 0.10),
            inset 0 1px 0 rgba(255, 255, 255, 0.08);
    }

    .block-container h1,
    .block-container h2,
    .block-container h3 {
        color: #f7f1ff !important;
        letter-spacing: -0.025em;
    }

    .block-container h1 {
        text-align: center;
        padding-bottom: 0.7rem;
        border-bottom: 3px solid #9275bd;
    }

    h2, h3 {
        margin-top: 1.2rem;
    }

    [data-testid="stCaptionContainer"] {
        color: #c4b8d8;
        text-align: center;
    }

    .block-container [data-testid="stMarkdownContainer"] p,
    .block-container label,
    .block-container [data-testid="stCaptionContainer"] {
        color: #eee7f8 !important;
    }

    .block-container [data-testid="stMetric"] {
        min-height: 112px;
        background: linear-gradient(145deg, #fff 0%, #f0eafa 100%);
        border: 1px solid #c9b5e5;
        border-radius: 18px;
        padding: 1.15rem 1.25rem;
        box-shadow:
            0 14px 28px rgba(8, 7, 18, 0.28),
            0 4px 10px rgba(8, 7, 18, 0.18),
            inset 0 1px 0 rgba(255, 255, 255, 0.95);
        transition: transform 160ms ease, box-shadow 160ms ease;
    }

    .block-container [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow:
            0 18px 34px rgba(8, 7, 18, 0.32),
            0 5px 12px rgba(8, 7, 18, 0.20),
            inset 0 1px 0 rgba(255, 255, 255, 0.95);
    }

    .block-container [data-testid="stMetricLabel"],
    .block-container [data-testid="stMetricLabel"] *,
    .block-container [data-testid="stMetricValue"],
    .block-container [data-testid="stMetricValue"] * {
        color: #563786 !important;
    }

    .block-container [data-testid="stMetricValue"] {
        font-weight: 700;
    }

    .stButton > button,
    [data-testid="stDownloadButton"] > button {
        border-radius: 10px;
        border: 1px solid #cbbbe3;
        color: var(--lila-oscuro);
        background: linear-gradient(145deg, #fff, #f1ebfa);
        font-weight: 600;
        transition: all 160ms ease;
        box-shadow: 0 4px 10px rgba(72, 53, 112, 0.08);
    }

    .stButton > button:hover,
    [data-testid="stDownloadButton"] > button:hover {
        border-color: var(--lila);
        color: #56318f;
        box-shadow: 0 4px 14px rgba(104, 65, 165, 0.14);
    }

    .stButton > button[kind="primary"] {
        color: #fff;
        border-color: var(--lila-oscuro);
        background: linear-gradient(135deg, #8261bc, #5b3b91);
        box-shadow: 0 7px 16px rgba(91, 59, 145, 0.24);
    }

    .stButton > button[kind="primary"]:hover {
        color: #fff;
        border-color: #56318f;
        background: linear-gradient(135deg, #7952b8, #56318f);
    }

    [data-testid="stFileUploader"] section {
        border: 1px dashed #a992ce;
        border-radius: 16px;
        background: linear-gradient(145deg, #f7f3fc, #eaf1fa);
        box-shadow: inset 0 1px 5px rgba(77, 68, 109, 0.07);
        color: #403650 !important;
    }

    [data-testid="stFileUploader"] section *,
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] small {
        color: #403650 !important;
    }

    [data-testid="stFileUploader"] section button {
        color: #fff !important;
        border: 1px solid #5b3b91 !important;
        border-radius: 9px !important;
        background: linear-gradient(135deg, #8261bc, #5b3b91) !important;
        box-shadow: 0 4px 10px rgba(91, 59, 145, 0.2);
    }

    [data-testid="stFileUploader"] section button:hover {
        border-color: #43286f !important;
        background: linear-gradient(135deg, #7350ae, #4d307f) !important;
    }

    [data-testid="stFileUploader"] section:hover {
        border-color: var(--lila);
        background: #eee8f8;
    }

    .block-container [data-testid="stAlert"] {
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(50, 59, 91, 0.08);
    }

    .block-container [data-testid="stAlert"] [data-testid="stMarkdownContainer"],
    .block-container [data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {
        color: #26354a !important;
    }

    .block-container [data-testid="stDataFrame"] {
        overflow: hidden;
        border: 1px solid #c9b5e5;
        border-radius: 14px;
        box-shadow:
            0 12px 26px rgba(8, 7, 18, 0.25),
            0 3px 8px rgba(8, 7, 18, 0.16);
    }

    .block-container [data-testid="stExpander"] {
        border: 1px solid #c9b5e5;
        border-radius: 12px;
        background: linear-gradient(145deg, #fff, #f0ebf8);
        box-shadow:
            0 10px 22px rgba(8, 7, 18, 0.23),
            0 3px 7px rgba(8, 7, 18, 0.15);
    }

    .block-container [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p {
        color: #403650 !important;
    }

    .block-container [data-testid="stMetric"] [data-testid="stMetricLabel"] *,
    .block-container [data-testid="stMetric"] [data-testid="stMetricValue"] *,
    .block-container [data-testid="stMetric"] label {
        color: #4b316f !important;
        opacity: 1 !important;
        -webkit-text-fill-color: #4b316f !important;
    }

    .block-container [data-testid="stAlert"],
    .block-container [data-testid="stAlert"] [role="alert"] {
        color: #26354a !important;
        background: #e6edf8 !important;
        border-color: #c2d2e9 !important;
    }

    .block-container [data-testid="stAlert"] *,
    .block-container [data-testid="stAlert"] [role="alert"] * {
        color: #26354a !important;
        opacity: 1 !important;
        -webkit-text-fill-color: #26354a !important;
    }

    .block-container [data-testid="stExpander"] summary,
    .block-container [data-testid="stExpander"] summary * {
        color: #403650 !important;
        opacity: 1 !important;
        -webkit-text-fill-color: #403650 !important;
    }

    .block-container .stButton > button,
    .block-container .stButton > button *,
    .block-container [data-testid="stDownloadButton"] > button,
    .block-container [data-testid="stDownloadButton"] > button * {
        color: #563786 !important;
        opacity: 1 !important;
        -webkit-text-fill-color: #563786 !important;
    }

    .block-container .stButton > button[kind="primary"],
    .block-container .stButton > button[kind="primary"] *,
    .block-container [data-testid="stFileUploader"] section button,
    .block-container [data-testid="stFileUploader"] section button * {
        color: #fff !important;
        -webkit-text-fill-color: #fff !important;
    }

    div[data-baseweb="slider"] [role="slider"] {
        background-color: var(--lila-oscuro);
        border-color: var(--lila-oscuro);
    }

    hr {
        border-color: #67577f;
    }

    @media (max-width: 700px) {
        [data-testid="stMainBlockContainer"] {
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
        }

        .block-container {
            width: calc(100vw - 1rem) !important;
            max-width: none !important;
            margin: 0.75rem auto 1.5rem !important;
            padding: 1.35rem 1.1rem 2rem;
            border-radius: 18px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 2. ENCABEZADO
# ============================================================

st.title("🛡️ Detector de intrusiones de red")

st.caption(
    "Proyecto de Inteligencia Computacional · Dataset UNSW-NB15"
)

st.info(
    "Prototipo académico. Las predicciones dependen del modelo entrenado "
    "y del dataset utilizado; no sustituye un sistema de seguridad en producción."
)


# ============================================================
# 3. FUNCIONES DEL MODELO
# ============================================================

@st.cache_resource
def cargar_modelo(ruta: str, mtime: float):
    """Carga el modelo y permite actualizar la caché si cambia el archivo."""
    del mtime
    return joblib.load(ruta)


def columnas_esperadas(modelo) -> list[str] | None:
    """Obtiene los nombres de las columnas que espera el modelo."""

    if hasattr(modelo, "feature_names_in_"):
        return list(modelo.feature_names_in_)

    pasos = getattr(modelo, "named_steps", {})
    preprocesador = pasos.get("pre")

    if preprocesador is not None and hasattr(
        preprocesador, "feature_names_in_"
    ):
        return list(preprocesador.feature_names_in_)

    return None


def preparar_entrada(
    df: pd.DataFrame,
    esperadas: list[str] | None
) -> pd.DataFrame:
    """Valida las columnas y prepara las características de entrada."""

    datos = df.copy()

    # Normalizar nombres de columnas.
    datos.columns = datos.columns.astype(str).str.strip()

    # Eliminar las etiquetas para no utilizarlas como predictores.
    datos = datos.drop(
        columns=[
            columna
            for columna in NON_FEATURES
            if columna in datos.columns
        ],
        errors="ignore",
    )

    if datos.empty:
        raise ValueError(
            "El archivo no contiene registros para predecir."
        )

    if esperadas:
        faltantes = [
            columna
            for columna in esperadas
            if columna not in datos.columns
        ]

        if faltantes:
            muestra = ", ".join(faltantes[:12])
            sufijo = "…" if len(faltantes) > 12 else ""

            raise ValueError(
                f"Faltan {len(faltantes)} columnas requeridas: "
                f"{muestra}{sufijo}"
            )

        # Mantener el orden esperado por el modelo.
        # Las columnas adicionales se ignoran.
        datos = datos[esperadas]

    return datos


def obtener_probabilidades(modelo, X: pd.DataFrame):
    """Obtiene la probabilidad correspondiente a la clase ataque (1)."""

    if not hasattr(modelo, "predict_proba"):
        raise ValueError(
            "El modelo no ofrece predict_proba(). "
            "Se necesita un clasificador compatible para calcular "
            "la probabilidad y el nivel de confianza."
        )

    probabilidades = modelo.predict_proba(X)

    clases = list(getattr(modelo, "classes_", []))

    if 1 not in clases:
        raise ValueError(
            "No se encontró la clase 1 (ataque) entre las clases del modelo."
        )

    indice_ataque = clases.index(1)

    return probabilidades[:, indice_ataque]


def nivel_confianza(
    prob_ataque: float,
    umbral_bajo: float,
    umbral_alto: float
) -> str:
    """Asigna un nivel orientativo según la claridad de la predicción."""

    claridad = max(prob_ataque, 1 - prob_ataque)

    if claridad >= umbral_alto:
        return "Alta"

    if claridad >= umbral_bajo:
        return "Media"

    return "Baja"

def validar_datos_csv(df: pd.DataFrame) -> dict:
    """
    Revisa la calidad básica de los datos cargados.
    No modifica el DataFrame original.
    """

    resultado = {
        "vacio": df.empty,
        "filas": len(df),
        "columnas": len(df.columns),
        "faltantes_total": int(df.isna().sum().sum()),
        "columnas_con_faltantes": {},
        "columnas_con_infinitos": {},
        "columnas_numericas_con_texto": {},
        "label_invalida": [],
        "duplicados": int(df.drop(columns=["id"], errors="ignore").duplicated().sum()),
    }

    # 1. Detectar valores faltantes.
    faltantes = df.isna().sum()

    resultado["columnas_con_faltantes"] = {
        str(columna): int(cantidad)
        for columna, cantidad in faltantes.items()
        if cantidad > 0
    }

    # 2. Identificar columnas categóricas que no deben
    # interpretarse como números.
    columnas_categoricas = {
        "proto",
        "service",
        "state",
        "attack_cat",
    }

    # label es una variable objetivo binaria: solo admite 0 y 1.
    # Las demás columnas se revisan como posibles variables numéricas.
    columnas_numericas = [
        columna
        for columna in df.columns
        if str(columna).strip().lower() not in columnas_categoricas
        and str(columna).strip().lower() != "label"
    ]

    # 3. Detectar texto no numérico e infinitos.
    for columna in columnas_numericas:
        valores = df[columna].dropna()

        # Convertir temporalmente a números para comprobar
        # qué valores son válidos.
        convertidos = pd.to_numeric(valores, errors="coerce")

        # Detectar valores que no pueden interpretarse como números.
        invalidos = convertidos.isna()

        if invalidos.any():
            resultado["columnas_numericas_con_texto"][str(columna)] = (
                int(invalidos.sum())
            )

        # Detectar infinitos, incluso si originalmente estaban
        # escritos como texto, por ejemplo "inf" o "-inf".
        infinitos = convertidos.isin(
            [float("inf"), float("-inf")]
        )

        if infinitos.any():
            resultado["columnas_con_infinitos"][str(columna)] = (
                int(infinitos.sum())
            )

    # 4. Validar la variable objetivo label, si está presente.
    if "label" in df.columns:
        valores_label = df["label"].dropna()
        label_numerica = pd.to_numeric(
            valores_label,
            errors="coerce",
        )

        invalidos_label = (
            label_numerica.isna()
            | (~label_numerica.isin([0, 1]))
        )

        if invalidos_label.any():
            resultado["label_invalida"] = (
                valores_label[invalidos_label]
                .astype(str)
                .unique()
                .tolist()[:10]
            )

    return resultado

# ============================================================
# 4. CARGA DEL ARCHIVO CSV
# ============================================================

st.header("1. Cargar datos de tráfico")

st.write(
    "Selecciona un archivo CSV que contenga registros de tráfico de red. "
    "Podrás revisar sus datos aunque el modelo final todavía no esté disponible."
)

archivo = st.file_uploader(
    "Selecciona un archivo CSV",
    type=["csv"],
    help="El archivo debe contener los nombres de las columnas en la primera fila.",
)

if archivo is None:
    st.info("Carga un archivo CSV para comenzar.")
    st.stop()


# ============================================================
# 5. LECTURA Y VALIDACIÓN INICIAL DEL CSV
# ============================================================

try:
    df = pd.read_csv(
        archivo,
        encoding="utf-8-sig",
        low_memory=False,
    )

except UnicodeDecodeError:
    try:
        archivo.seek(0)

        df = pd.read_csv(
            archivo,
            encoding="latin-1",
            low_memory=False,
        )

    except Exception as error:
        st.error(
            "No se pudo leer el archivo CSV. "
            f"Comprueba su codificación y formato. Detalle: {error}"
        )
        st.stop()

except pd.errors.EmptyDataError:
    st.error(
        "El archivo está vacío o no contiene una tabla CSV válida."
    )
    st.stop()

except Exception as error:
    st.error(
        f"No se pudo leer el archivo CSV. Detalle: {error}"
    )
    st.stop()


# Comprobar que el archivo contiene registros.
if df.empty:
    st.error("El CSV no contiene registros.")
    st.stop()


# Limpiar espacios de los nombres de las columnas.
df.columns = df.columns.astype(str).str.strip()


# Comprobar nombres de columnas duplicados.
if df.columns.duplicated().any():

    duplicadas = df.columns[
        df.columns.duplicated()
    ].tolist()

    st.error(
        "Se encontraron nombres de columnas duplicados: "
        f"{duplicadas}"
    )

    st.stop()


st.success("Archivo CSV cargado correctamente.")


# ============================================================
# 6. RESUMEN DEL ARCHIVO
# ============================================================

st.subheader("Resumen del archivo")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Registros",
    f"{len(df):,}"
)

col2.metric(
    "Columnas",
    f"{len(df.columns):,}"
)

col3.metric(
    "Tamaño del archivo",
    f"{archivo.size / 1024:.1f} KB"
)


# ============================================================
# 7. NOMBRES DE LAS COLUMNAS
# ============================================================

with st.expander("Consultar nombres de las columnas"):

    st.write(list(df.columns))


# ============================================================
# 8. VISTA PREVIA
# ============================================================

st.subheader("Vista previa de los datos")

cantidad = min(20, len(df))

st.caption(
    f"Se muestran los primeros {cantidad} de {len(df):,} registros."
)

st.dataframe(
    df.head(cantidad),
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# 9. REVISIÓN Y VALIDACIÓN DE LOS DATOS
# ============================================================

st.subheader("Revisión inicial")

validacion = validar_datos_csv(df)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Celdas vacías",
    f"{validacion['faltantes_total']:,}"
)

col2.metric(
    "Filas duplicadas",
    f"{validacion['duplicados']:,}"
)

col3.metric(
    "¿Incluye label?",
    "Sí" if TARGET in df.columns else "No"
)


# ------------------------------------------------------------
# Valores faltantes
# ------------------------------------------------------------

if validacion["columnas_con_faltantes"]:

    st.warning(
        "Se encontraron valores faltantes en algunas columnas."
    )

    faltantes_df = pd.DataFrame(
        list(
            validacion["columnas_con_faltantes"].items()
        ),
        columns=["Columna", "Valores faltantes"],
    )

    st.dataframe(
        faltantes_df,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.success("No se encontraron valores faltantes.")


# ------------------------------------------------------------
# Valores infinitos
# ------------------------------------------------------------

if validacion["columnas_con_infinitos"]:

    st.warning(
        "Se detectaron valores infinitos. "
        "Revisa estas columnas antes de realizar predicciones."
    )

    infinitos_df = pd.DataFrame(
        list(
            validacion["columnas_con_infinitos"].items()
        ),
        columns=["Columna", "Valores infinitos"],
    )

    st.dataframe(
        infinitos_df,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.success("No se detectaron valores infinitos en columnas numéricas.")


# ------------------------------------------------------------
# Columnas numéricas con problemas
# ------------------------------------------------------------

if validacion["columnas_numericas_con_texto"]:

    st.warning(
        "Se encontraron valores que no se pueden interpretar "
        "como números en columnas numéricas."
    )

    texto_df = pd.DataFrame(
        list(
            validacion["columnas_numericas_con_texto"].items()
        ),
        columns=["Columna", "Valores inválidos"],
    )

    st.dataframe(
        texto_df,
        use_container_width=True,
        hide_index=True,
    )


# ------------------------------------------------------------
# Validación de label
# ------------------------------------------------------------

if TARGET in df.columns:

    with st.expander(
        "Consultar distribución de la variable label"
    ):

        distribucion = (
            df[TARGET]
            .value_counts(dropna=False)
            .rename_axis("label")
            .reset_index(name="cantidad")
        )

        st.dataframe(
            distribucion,
            use_container_width=True,
            hide_index=True,
        )

    if validacion["label_invalida"]:

        st.error(
            "La columna label contiene valores distintos de 0 y 1, "
            "o valores que no son numéricos. "
            f"Ejemplos: {validacion['label_invalida']}"
        )

    elif df[TARGET].isna().any():

        st.warning(
            "La columna label contiene valores faltantes."
        )

    else:

        st.success(
            "La columna label solo contiene valores válidos: 0 y 1."
        )

# ============================================================
# 10. DESCARGA DEL CSV CARGADO
# ============================================================

st.download_button(
    label="Descargar una copia del CSV cargado",
    data=df.to_csv(index=False).encode("utf-8-sig"),
    file_name="datos_cargados.csv",
    mime="text/csv",
)

# ============================================================
# 11. PREDICCIÓN DE INTRUSIONES
# ============================================================

st.divider()

st.header("2. Predicción de intrusiones")


if not MODEL_PATH.exists():

    st.warning(
        "El modelo final todavía no está disponible. "
        "Puedes cargar archivos, revisar sus columnas y consultar "
        "sus estadísticas. Las predicciones se habilitarán cuando "
        "se encuentre el archivo models/pipeline_final.joblib."
    )

else:

    try:
        # --------------------------------------------------------
        # 1. Cargar el modelo y preparar los datos
        # --------------------------------------------------------

        modelo = cargar_modelo(
            str(MODEL_PATH),
            MODEL_PATH.stat().st_mtime,
        )

        esperadas = columnas_esperadas(modelo)

        X = preparar_entrada(
            df,
            esperadas,
        )

        # --------------------------------------------------------
        # 2. Determinar si existen errores que bloquean el análisis
        # --------------------------------------------------------

        problemas_bloqueantes = []

        # Bloquear si existen valores infinitos.
        if validacion["columnas_con_infinitos"]:

            for columna, cantidad in (
                validacion["columnas_con_infinitos"].items()
            ):
                problemas_bloqueantes.append(
                    f"La columna '{columna}' contiene "
                    f"{cantidad} valor(es) infinito(s)."
                )

        # Bloquear si existen textos en columnas numéricas.
        if validacion["columnas_numericas_con_texto"]:

            for columna, cantidad in (
                validacion["columnas_numericas_con_texto"].items()
            ):
                problemas_bloqueantes.append(
                    f"La columna '{columna}' contiene "
                    f"{cantidad} valor(es) que no son numéricos."
                )

        # --------------------------------------------------------
        # 3. Mostrar los problemas encontrados
        # --------------------------------------------------------

        if problemas_bloqueantes:

            st.error(
                "No se pueden realizar las predicciones porque "
                "el archivo contiene errores en los datos de entrada."
            )

            st.markdown(
                "**Corrige los siguientes problemas y vuelve a cargar "
                "el archivo CSV:**"
            )

            for problema in problemas_bloqueantes:
                st.write(f"- {problema}")

        else:

            st.success(
                "No se detectaron valores infinitos ni textos "
                "inválidos en las columnas numéricas revisadas."
            )

            # Los valores faltantes se informan, pero no se bloquean
            # automáticamente: el pipeline podría incluir imputación.
            if validacion["faltantes_total"] > 0:

                st.warning(
                    f"El archivo contiene {validacion['faltantes_total']} "
                    "valor(es) faltante(s). Comprueba que el modelo "
                    "pueda gestionarlos antes de interpretar los resultados."
                )

            # ----------------------------------------------------
            # 4. Validar umbrales de confianza
            # ----------------------------------------------------

            st.subheader("Configuración de confianza")

            col1, col2 = st.columns(2)

            with col1:
                umbral_bajo = st.slider(
                    "Confianza media desde",
                    min_value=0.50,
                    max_value=0.95,
                    value=0.65,
                    step=0.05,
                    key="umbral_bajo",
                )

            with col2:
                umbral_alto = st.slider(
                    "Confianza alta desde",
                    min_value=0.55,
                    max_value=1.00,
                    value=0.85,
                    step=0.05,
                    key="umbral_alto",
                )

            if umbral_bajo >= umbral_alto:

                st.error(
                    "El umbral de confianza media debe ser menor "
                    "que el umbral de confianza alta."
                )

            # ----------------------------------------------------
            # 5. Ejecutar las predicciones
            # ----------------------------------------------------

            elif st.button(
                "Analizar tráfico",
                type="primary",
            ):

                try:
                    predicciones = modelo.predict(X)

                    probabilidades = obtener_probabilidades(
                        modelo,
                        X,
                    )

                    resultados = pd.DataFrame({
                        "Registro": range(1, len(X) + 1),

                        "Predicción": [
                            "Ataque" if int(prediccion) == 1 else "Normal"
                            for prediccion in predicciones
                        ],

                        "Probabilidad de ataque (%)": (
                            probabilidades * 100
                        ).round(2),

                        "Confianza": [
                            nivel_confianza(
                                float(probabilidad),
                                umbral_bajo,
                                umbral_alto,
                            )
                            for probabilidad in probabilidades
                        ],
                    })

                    # --------------------------------------------
                    # 6. Resumen de resultados
                    # --------------------------------------------

                    total_ataques = int(
                        (predicciones == 1).sum()
                    )

                    total_normales = int(
                        (predicciones == 0).sum()
                    )

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Registros analizados",
                        f"{len(resultados):,}",
                    )

                    col2.metric(
                        "Detectados como ataque",
                        f"{total_ataques:,}",
                    )

                    col3.metric(
                        "Clasificados como normales",
                        f"{total_normales:,}",
                    )

                    # --------------------------------------------
                    # 7. Tabla de resultados
                    # --------------------------------------------

                    st.subheader("Resultados de la clasificación")

                    st.dataframe(
                        resultados,
                        use_container_width=True,
                        hide_index=True,
                    )

                    if (TARGET in df.columns
                            and not validacion["label_invalida"]
                            and not df[TARGET].isna().any()):

                        y_real = pd.to_numeric(df[TARGET]).astype(int).to_numpy()
                        y_pred = np.asarray(predicciones).astype(int)

                        st.subheader("Comparación con la etiqueta real (label)")

                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Exactitud", f"{accuracy_score(y_real, y_pred):.3f}")
                        c2.metric("Precisión", f"{precision_score(y_real, y_pred, zero_division=0):.3f}")
                        c3.metric("Recall", f"{recall_score(y_real, y_pred, zero_division=0):.3f}")
                        c4.metric("F1", f"{f1_score(y_real, y_pred, zero_division=0):.3f}")

                        matriz = confusion_matrix(y_real, y_pred, labels=[0, 1])
                        st.dataframe(
                            pd.DataFrame(
                                matriz,
                                index=["Real: normal", "Real: ataque"],
                                columns=["Predicho: normal", "Predicho: ataque"],
                            ),
                            use_container_width=True,
                        )

                    # --------------------------------------------
                    # 8. Descargar resultados
                    # --------------------------------------------

                    st.download_button(
                        label="Descargar resultados de predicción",
                        data=resultados.to_csv(
                            index=False
                        ).encode("utf-8-sig"),
                        file_name="resultados_prediccion.csv",
                        mime="text/csv",
                    )

                except Exception as error:

                    st.error(
                        "No se pudo analizar el tráfico. "
                        "Comprueba los datos y la compatibilidad "
                        "con el modelo entrenado. "
                        f"Detalle: {error}"
                    )

    except Exception as error:

        st.error(
            "No se pudo preparar el archivo para el modelo. "
            "Comprueba que incluya las columnas requeridas y "
            "que sus nombres sean correctos. "
            f"Detalle: {error}"
        )


# ============================================================
# 12. ALCANCE Y LIMITACIONES
# ============================================================

with st.expander("Alcance y limitaciones"):

    st.markdown(
        """
        - Este prototipo tiene fines académicos y no reemplaza
          herramientas profesionales de seguridad.
        - Las predicciones requieren un modelo previamente entrenado.
        - Si el archivo contiene la columna `label`, esta representa
          la respuesta esperada y no debe utilizarse como predictor.
        - Los niveles de confianza son orientativos y deben validarse
          con los resultados del modelo definitivo.
        - El dataset UNSW-NB15 fue creado en un entorno de laboratorio;
          su rendimiento no garantiza los mismos resultados en redes reales.
        """
    )