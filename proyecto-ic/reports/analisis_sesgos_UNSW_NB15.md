# Análisis de posibles sesgos y limitaciones del dataset UNSW-NB15

**Proyecto:** Inteligencia Computacional — detección de tráfico normal y ataques de red  
**Fecha de revisión:** 28 de octubre de 2026  
**Objetivo:** identificar condiciones del dataset que podrían afectar el entrenamiento, la evaluación y la generalización de los modelos.

## 1. Propósito y criterio de análisis

Este documento identifica posibles sesgos y limitaciones del dataset UNSW-NB15 relevantes para el proyecto. Se distingue entre:

- **Hecho documentado:** característica descrita por la fuente oficial o por la estructura de los archivos.
- **Observación local:** resultado obtenido al revisar los CSV usados en el proyecto.
- **Riesgo potencial:** consecuencia metodológica posible que debe evaluarse; no implica que necesariamente haya ocurrido.
- **Medida recomendada:** acción para reducir el riesgo o comunicar sus límites.

El análisis no demuestra por sí solo que un modelo esté sesgado. Su función es orientar la preparación de los datos y el diseño de una evaluación válida.

## 2. Posibles sesgos y medidas de mitigación

### 2.1. Entorno de generación del tráfico

**Hecho documentado.** UNSW-NB15 se construyó en un laboratorio de ciberseguridad de UNSW Canberra. La fuente oficial describe una combinación de actividad normal contemporánea y comportamientos de ataque sintéticos generados con IXIA PerfectStorm; el tráfico fue capturado y procesado para obtener características de red.

**Riesgo potencial.** Los patrones de tráfico, dispositivos, servicios y ataques de un entorno de laboratorio pueden no representar toda la diversidad de redes reales. Un modelo podría aprender características propias del entorno de captura y rendir peor en una organización, infraestructura o configuración distinta.

**Medida recomendada.** Presentar los resultados como evaluación experimental sobre UNSW-NB15, no como garantía de eficacia en cualquier red real. Si fuera posible, validar posteriormente con datos independientes y autorizados de otro entorno.

### 2.2. Antigüedad y evolución de las amenazas

**Hecho documentado.** UNSW-NB15 es un conjunto de referencia publicado a partir de investigaciones de 2015 y 2016.

**Riesgo potencial.** El comportamiento del tráfico, las aplicaciones, las configuraciones de red y las técnicas de ataque cambian con el tiempo. Por ello, un buen resultado en este dataset no garantiza la detección de amenazas nuevas o de variantes que no estén representadas.

**Medida recomendada.** Reconocer la antigüedad del dataset como limitación y evitar afirmar que el modelo detecta todas las amenazas actuales. Una validación futura debería incluir datos recientes, si se dispone de ellos de forma legítima.

### 2.3. Distribución desigual de las clases

**Observación local.** En el CSV local de entrenamiento antes de la limpieza, la etiqueta binaria `label` contiene 119.341 registros de ataque (1) y 56.000 registros normales (0), aproximadamente 68,06 % y 31,94 %, respectivamente. En el archivo local de prueba hay 45.332 ataques y 37.000 registros normales, aproximadamente 55,05 % y 44,95 %.

**Hecho documentado.** La distribución entre categorías de ataque también es desigual. Por ejemplo, la categoría `Worms` tiene muchos menos registros que categorías como `Generic` o `Exploits` en la partición oficial.

**Riesgo potencial.** La exactitud (*accuracy*) puede ocultar un rendimiento bajo en categorías minoritarias. Además, la proporción de clases cambia entre las particiones locales, lo que puede influir en la comparación de resultados y en la interpretación de métricas.

**Medida recomendada.** Informar, además de *accuracy*, precisión, *recall*, F1 y matriz de confusión. Para el objetivo binario, reportar resultados para ambas clases; si se estudian categorías de ataque, informar métricas por categoría y tener en cuenta la escasez de ejemplos. La muestra de entrenamiento debe seleccionarse de forma estratificada y solo desde el conjunto de entrenamiento.

### 2.4. Registros duplicados y patrones repetidos

**Observación local.** En el archivo de entrenamiento original se identificaron numerosos registros repetidos bajo distintos criterios de comparación. En la revisión del proyecto se contabilizaron 74.072 duplicados al comparar las variables predictoras y las etiquetas `label`, excluyendo `id` y `attack_cat`. La cifra depende del criterio utilizado para definir un duplicado; no debe presentarse sin explicar qué columnas se compararon.

**Riesgo potencial.** Los registros repetidos pueden dar un peso desproporcionado a ciertos patrones. Si registros idénticos o muy similares se distribuyen entre entrenamiento y evaluación, las métricas pueden sobreestimar la capacidad de generalización. El riesgo concreto depende de la relación entre los archivos y del protocolo de evaluación.

**Medida recomendada.** Documentar exactamente las columnas usadas para detectar duplicados y cuántas filas se eliminaron en cada etapa. Mantener intacto el archivo de prueba original para una evaluación comparable y comprobar si hay solapamiento entre entrenamiento y prueba. Si se limpia el entrenamiento, describirlo como una decisión metodológica, no como una corrección universalmente obligatoria.

### 2.5. Características idénticas con etiquetas contradictorias

**Observación local.** En la revisión del entrenamiento, al comparar las características excluyendo `id`, `attack_cat` y `label`, se detectaron 229 combinaciones de características asociadas a más de un valor de `label`. Tras una deduplicación anterior, estas combinaciones involucraban 458 filas. La limpieza posterior retiró las filas de esas combinaciones y dejó 100.811 registros. Por tanto, las 458 filas corresponden al estado del archivo después de la deduplicación previa, no al CSV original de 175.341 filas.

**Riesgo potencial.** Si las mismas características se asocian a etiquetas binarias distintas, el modelo recibe ejemplos ambiguos y puede resultar más difícil aprender una frontera de decisión consistente. Sin embargo, no se debe asumir automáticamente que toda contradicción sea un error: primero hay que verificar la definición de las variables, el proceso de etiquetado y los criterios de comparación.

**Medida recomendada.** Conservar un registro reproducible de la cantidad de conflictos y de la regla aplicada para tratarlos. En el informe, explicar que se excluyeron las combinaciones conflictivas para evitar entrenar con etiquetas binarias contradictorias, y reconocer que esta decisión reduce el tamaño del conjunto y podría eliminar casos legítimos. No aplicar silenciosamente la misma eliminación al conjunto de prueba.

### 2.6. Riesgo de fuga de información (*data leakage*)

**Hecho metodológico.** El proyecto transforma variables numéricas y categóricas y contempla una muestra para experimentación. Si los parámetros de imputación, escalado, selección de variables o cualquier transformación se calculan usando datos de validación o prueba, parte de la información de evaluación puede influir en el entrenamiento.

**Riesgo potencial.** Las métricas resultantes podrían ser demasiado optimistas y no reflejar el rendimiento sobre datos realmente no vistos. La fuga también puede aparecer si se incluyen como predictores identificadores o etiquetas que revelan directamente el resultado.

**Medida recomendada.**
1. Definir `label` como objetivo binario.
2. Excluir `id` por ser un identificador, y excluir `attack_cat` si el objetivo es predecir `label`, porque es una etiqueta adicional que puede revelar la clase.
3. Ajustar imputación, codificación y escalado solo con los datos de entrenamiento de cada partición.
4. Para validación cruzada, incluir preprocesamiento y clasificador dentro de un `Pipeline` de scikit-learn.
5. Mantener el conjunto de prueba separado hasta la evaluación final y no usar sus etiquetas para tomar decisiones sobre el modelo.

### 2.7. Redundancia y correlación entre variables

**Observación local.** En el conjunto de entrenamiento limpio, `ct_ftp_cmd` e `is_ftp_login` resultaron idénticas. En el conjunto de prueba no son idénticas. Por esta diferencia, la exclusión de `ct_ftp_cmd` es una decisión provisional que debe verificarse con el equipo y documentarse.

**Riesgo potencial.** Las variables redundantes pueden aumentar la complejidad o afectar algunos modelos, especialmente aquellos sensibles a escalas o distancias. No obstante, una correlación alta no significa automáticamente que una variable deba eliminarse; la utilidad depende del modelo y de la evaluación.

**Medida recomendada.** Confirmar que los nombres y contenidos de las particiones son los esperados; cuantificar las diferencias entre las dos columnas en entrenamiento y prueba; decidir la lista final de predictores sin modificar artificialmente los valores de prueba. Comparar, cuando sea viable, el rendimiento con y sin la variable antes de afirmar que la eliminación mejora el modelo.

### 2.8. Nombres de archivo y trazabilidad de las particiones

**Observación local.** Los archivos disponibles en el proyecto parecen tener nombres cuyo tamaño no coincide con el reparto oficial: el archivo local `UNSW_NB15_testing-set.csv` contiene 175.341 filas y el archivo local `UNSW_NB15_training-set.csv` contiene 82.332. La fuente oficial atribuye 175.341 filas al entrenamiento y 82.332 a prueba.

**Riesgo potencial.** Si el código asigna los roles según el nombre del archivo sin comprobar su contenido, el entrenamiento y la evaluación podrían quedar invertidos o describirse incorrectamente en el informe.

**Medida recomendada.** Verificar el contenido, el origen y, si están disponibles, las sumas de comprobación de los archivos. Usar nombres de variables inequívocos en el código (`train_raw`, `test_raw`) y explicar la discrepancia en el README y el informe. No renombrar ni intercambiar archivos sin verificar primero su procedencia.

## 3. Prioridades para este proyecto

Antes de entrenar y comparar modelos, se recomienda priorizar:

1. **Confirmar la identidad de los archivos** y dejar documentada la discrepancia de nombres y tamaños.
2. **Fijar una única regla de limpieza**, conservar las cifras de cada etapa y asegurar que el procedimiento se puede reproducir.
3. **Definir los predictores y el objetivo**, incluyendo la decisión pendiente sobre `ct_ftp_cmd`.
4. **Evitar fuga de información** con un `Pipeline` dentro de la validación cruzada y con el conjunto de prueba reservado.
5. **Evaluar más allá de la exactitud**, reportando métricas por clase y matriz de confusión.
6. **Explicar los límites de generalización**, especialmente por el entorno de laboratorio y la antigüedad del dataset.

## 4. Conclusión

UNSW-NB15 es útil como conjunto de referencia para experimentar con la detección de intrusiones, pero sus resultados deben interpretarse considerando el entorno de generación, la antigüedad, la distribución desigual de clases, la presencia de duplicados y posibles conflictos de etiquetas. En este proyecto también es esencial controlar la trazabilidad de los archivos y prevenir la fuga de información. Estas medidas no eliminan todos los sesgos posibles, pero hacen que el proceso y la evaluación sean más transparentes y reproducibles.

## 5. Fuentes de referencia

- Moustafa, N., & Slay, J. (2015). *UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set).* Military Communications and Information Systems Conference (MilCIS). DOI: 10.1109/MilCIS.2015.7348942.
- Moustafa, N., & Slay, J. (2016). *The evaluation of Network Anomaly Detection Systems: Statistical analysis of the UNSW-NB15 dataset and the comparison with the KDD99 dataset.* Information Security Journal: A Global Perspective.
- UNSW Canberra. *The UNSW-NB15 Dataset*. https://research.unsw.edu.au/projects/unsw-nb15-dataset

**Nota de trazabilidad:** las cifras de archivos locales y de limpieza de este documento proceden de la revisión del proyecto y deben actualizarse si cambia el código o se repite la limpieza desde los CSV originales. Las afirmaciones sobre el entorno de generación y las particiones oficiales proceden de la descripción del dataset.
