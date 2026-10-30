# Detector de intrusiones de red (UNSW-NB15)

Proyecto integrador de Inteligencia Computacional. Clasificación binaria de tráfico de red
(`label`: 0 = normal, 1 = ataque) con el dataset UNSW-NB15, optimización de hiperparámetros con
un algoritmo genético y un prototipo en Streamlit.

> Prototipo académico. No sustituye un sistema de seguridad en producción.
**Prototipo en línea:**
 https://detector-intrusiones-unsw.streamlit.app (también se puede ejecutar en local con el comando de abajo).

## Estructura

| Carpeta | Contenido |
|---|---|
| `data/` | CSV originales y `processed/` (submuestra de entrenamiento y prueba completa) |
| `notebooks/` | `01_eda`, `02_modelos`, `03_optimizacion_ga` |
| `src/` | Código reutilizable: preprocesamiento, modelos, GA, figuras del EDA, verificación del modelo y casos de prueba |
| `models/` | `pipeline_final.joblib` (preprocesamiento + Random Forest) |
| `app/` | Prototipo Streamlit (`app.py`) |
| `reports/` | Figuras, tablas de resultados, ficha y diccionario del dataset, capturas del prototipo |

## Datos

Dataset UNSW-NB15 (Moustafa y Slay, 2015), partición oficial: 175.341 registros de entrenamiento y
82.332 de prueba, 45 columnas (uso académico).

**Atención:** en esta copia los nombres de los CSV están intercambiados respecto al contenido:
`UNSW_NB15_testing-set.csv` tiene 175.341 filas (se usa como **entrenamiento**) y
`UNSW_NB15_training-set.csv` tiene 82.332 (se usa como **prueba**). Verifica esta asignación si
reemplazas los archivos.

## Preparar el entorno (Windows)

Desde la carpeta `proyecto-ic`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Las versiones de `requirements.txt` son las usadas para entrenar el modelo final. No las cambies:
un modelo guardado con una versión de scikit-learn puede fallar al cargarse con otra.

## Cómo reproducir

1. **EDA y figuras** (solo entrenamiento):
   ```powershell
   .\.venv\Scripts\python.exe .\src\eda_figuras.py --train .\data\UNSW_NB15_testing-set.csv --out .\reports\figuras
   ```
2. **Notebooks**, en orden: `01_eda` → `02_modelos` → `03_optimizacion_ga`
   (`03` tiene `MODO_RAPIDO = True` para una prueba rápida; ponlo en `False` para la corrida final).
3. **Verificar el modelo** sobre la prueba completa:
   ```powershell
   .\.venv\Scripts\python.exe .\src\verificar_modelo.py --modelo models\pipeline_final.joblib --test data\processed\test_completo.csv
   ```
   El informe se guarda en `reports/verificacion_modelo.txt`.

## Ejecutar el prototipo

```powershell
.\.venv\Scripts\python.exe -m streamlit run app\app.py
```

1. Sube un CSV con las columnas de UNSW-NB15 (la columna `label` es opcional).
2. Revisa el resumen y la validación (faltantes, infinitos, duplicados sin contar `id`).
3. Ajusta los umbrales de confianza y pulsa **Analizar tráfico**.
4. Si el CSV trae `label`, la app muestra exactitud, precisión, recall, F1 y la matriz de confusión.

### Casos de prueba

```powershell
.\.venv\Scripts\python.exe .\src\crear_casos_prueba.py --modelo models\pipeline_final.joblib --test data\processed\test_completo.csv
```

Crea 4 CSV en `data/casos_prueba/`:

| Archivo | Qué prueba | Resultado esperado |
|---|---|---|
| `caso1_normal.csv` | 10 registros normales bien clasificados | Todos "Normal" |
| `caso2_ataque.csv` | 10 ataques bien clasificados | Todos "Ataque", confianza alta |
| `caso3_limite.csv` | Un registro dudoso y otro con categorías nunca vistas | Probabilidad cercana a 50 % (confianza baja) y una predicción pese a categorías desconocidas |
| `caso4_columna_faltante.csv` | Falta la columna `dur` | La app muestra un error claro, sin caerse |

Las capturas están en `reports/capturas_prototipo/`.

## Resultados

Validación cruzada de 5 pliegues sobre una submuestra estratificada de 25.000 registros de entrenamiento:

| Modelo | F1 | Recall | ROC-AUC |
|---|---|---|---|
| Dummy (línea base) | 0,000 | 0,000 | 0,500 |
| KNN | 0,905 | 0,918 | 0,965 |
| SVM (RBF) | 0,909 | 0,975 | 0,972 |
| MLP | 0,912 | 0,937 | 0,977 |
| Random Forest | 0,923 | 0,944 | 0,982 |

Optimización de Random Forest (validación cruzada, F1): inicial 0,9248, búsqueda aleatoria 0,9256 y
algoritmo genético 0,9253. Las diferencias están dentro del ruido (desviación de 0,002 a 0,003), por lo que
no hay una mejora significativa; el GA iguala a la búsqueda aleatoria con menor costo de entrenamiento e inferencia.
Modelo final: Random Forest con 150 árboles, `max_depth=20`, `min_samples_leaf=1`, `max_features=0.3`.

**Prueba oficial (82.332 registros, usada una sola vez):**

| Exactitud | Precisión | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| 0,871 | 0,818 | 0,986 | 0,894 | 0,982 |

El modelo detecta casi todos los ataques (617 no detectados), pero marca como ataque cerca del 27 %
del tráfico normal (9.968 falsos positivos). El rendimiento en la prueba es menor que en la validación
cruzada por la diferencia de distribución entre las particiones de UNSW-NB15.

## Limitaciones

- El dataset fue generado en un entorno de laboratorio; el rendimiento no garantiza los mismos resultados en redes reales.
- Los niveles de confianza de la app son orientativos y no equivalen a la probabilidad real de acierto.
- Los registros con categorías nunca vistas (`proto`, `service`, `state`) se agrupan sin aviso al usuario.
- Los resultados de optimización provienen de una sola ejecución por método.

## Referencias

Moustafa, N., y Slay, J. (2015). UNSW-NB15: a comprehensive data set for network intrusion detection
systems. *2015 Military Communications and Information Systems Conference (MilCIS)*.
Ver la ficha del dataset en `reports/` para las demás citas.