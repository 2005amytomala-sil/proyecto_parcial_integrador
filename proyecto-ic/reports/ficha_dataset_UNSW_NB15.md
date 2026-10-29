# Ficha técnica del conjunto de datos UNSW-NB15

**Fecha de consulta de la fuente oficial:** 29 de octubre de 2026\
**Proyecto:** Detección binaria de intrusiones en tráfico de red\
**Responsable de documentación:** Persona B

## 1. Identificación y procedencia

  ----------------------------------------------------------------------------------------------------------------
  Campo                               Descripción
  ----------------------------------- ----------------------------------------------------------------------------
  Nombre                              UNSW-NB15

  Tipo de recurso                     Conjunto de datos etiquetado para investigación en detección de intrusiones
                                      de red

  Institución de referencia           UNSW Canberra, Australian Centre for Cyber Security (ACCS), según la
                                      descripción del conjunto

  Fuente oficial                      [Página oficial de
                                      UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

  Artículo principal                  Moustafa, N. y Slay, J. (2015). *UNSW-NB15: a comprehensive data set for
                                      network intrusion detection systems (UNSW-NB15 network data set).* Military
                                      Communications and Information Systems Conference (MilCIS), 2015. DOI:
                                      [10.1109/MilCIS.2015.7348942](https://doi.org/10.1109/MilCIS.2015.7348942)

  Fecha de consulta                   28 de octubre de 2026

  Condiciones de uso                  La página oficial indica que se permite el uso académico y de investigación;
                                      el uso comercial debe acordarse con los autores. La fuente solicita citar
                                      los trabajos indicados por sus creadores. No se debe atribuir una licencia
                                      abierta específica sin verificarla en la distribución exacta descargada.
  ----------------------------------------------------------------------------------------------------------------

## 2. Descripción

UNSW-NB15 fue creado en un entorno de laboratorio de ciberseguridad. La
fuente oficial describe una combinación de actividades normales de red y
comportamientos de ataque contemporáneos sintetizados mediante la
herramienta IXIA PerfectStorm. El tráfico se capturó y procesó con
herramientas como tcpdump, Argus y Bro-IDS para generar características
de los flujos.

El conjunto contempla tráfico normal y nueve categorías de ataque:
Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance,
Shellcode y Worms. La etiqueta `label` permite plantear la clasificación
binaria (`0` = normal; `1` = ataque), mientras que `attack_cat`
identifica la categoría detallada.

## 3. Archivos y dimensiones

La página oficial informa que la partición preparada para aprendizaje
automático contiene 175.341 registros de entrenamiento y 82.332
registros de prueba. En la copia utilizada por este proyecto se detectó
una discrepancia entre los nombres de archivo y sus dimensiones; por
ello, las rutas se configuraron de acuerdo con el contenido comprobado y
no solamente con el nombre del archivo.

  -------------------------------------------------------------------------------------------
  Archivo local                            Filas × columnas Uso en el        Observación
                                                 observadas proyecto         
  ----------------------------------- --------------------- ---------------- ----------------
  `data/UNSW_NB15_testing-set.csv`             175.341 × 45 Entrenamiento de Aunque el nombre
                                                            origen           contiene
                                                                             `testing`, su
                                                                             dimensión
                                                                             coincide con la
                                                                             partición de
                                                                             entrenamiento
                                                                             indicada por la
                                                                             fuente oficial.

  `data/UNSW_NB15_training-set.csv`             82.332 × 45 Prueba de origen Aunque el nombre
                                                                             contiene
                                                                             `training`, su
                                                                             dimensión
                                                                             coincide con la
                                                                             partición de
                                                                             prueba indicada
                                                                             por la fuente
                                                                             oficial.
  -------------------------------------------------------------------------------------------

**Advertencia de trazabilidad:** la coincidencia de dimensiones no
prueba por sí sola la procedencia o integridad de los archivos. Se debe
conservar esta observación en el informe y, si es posible, verificar los
archivos con la descarga original o con sumas de comprobación publicadas
por la fuente.

En la versión local revisada, los archivos tienen 45 columnas. La página
oficial también describe los archivos de partición y el archivo
`UNSW-NB15_features.csv` como referencia para las definiciones de
características. Debe consultarse ese diccionario oficial para confirmar
las definiciones técnicas exactas.

## 4. Estadísticas observadas en el proyecto

Las siguientes cifras corresponden a la copia local analizada y deben
distinguirse de las dimensiones originales reportadas por la fuente.

  ------------------------------------------------------------------------------
  Etapa                        Registros             Columnas Observaciones
  ----------------- -------------------- -------------------- ------------------
  Archivo local                  175.341                   45 `label`: 119.341
  usado como                                                  ataques (68,06 %)
  entrenamiento,                                              y 56.000 normales
  antes de la                                                 (31,94 %).
  limpieza del                                                
  notebook                                                    

  Entrenamiento                  101.269                   45 Estado encontrado
  después de una                                              antes de excluir
  depuración previa                                           las combinaciones
  existente en el                                             con etiquetas
  notebook                                                    contradictorias.

  Entrenamiento                  100.811                   45 Sin duplicados
  después de                                                  restantes al
  excluir 458 filas                                           comparar las
  asociadas a 229                                             características
  combinaciones de                                            predictoras según
  características                                             la comprobación
  conflictivas                                                ejecutada.

  Submuestra                      25.000                   45 Conserva
  estratificada de                                            aproximadamente la
  entrenamiento                                               proporción de
                                                              clases del
                                                              entrenamiento
                                                              limpio: 48,8 % de
                                                              ataques.

  Archivo local                   82.332                   45 Se mantiene sin la
  usado como prueba                                           limpieza de
                                                              entrenamiento
                                                              descrita.
  ------------------------------------------------------------------------------

La transformación exploratoria actual genera matrices de 59 columnas
para la submuestra de entrenamiento y el conjunto de prueba. Esta
cantidad corresponde a las variables después de la codificación y
transformación, no al número de columnas originales del CSV.

## 5. Variables objetivo y alcance

-   **Objetivo binario:** `label`.
-   **Codificación:** `0` para tráfico normal y `1` para tráfico de
    ataque, de acuerdo con la descripción habitual de UNSW-NB15.
-   **Categoría detallada:** `attack_cat`; se excluye de las variables
    predictoras para evitar introducir directamente la categoría del
    ataque en una tarea cuyo objetivo es predecir `label`.
-   **Identificador:** `id`; se excluye de los predictores.
-   **Variables predictoras:** se documentan en
    `diccionario_variables.csv`. La selección final debe ser coherente
    entre la documentación, el código de preprocesamiento y el pipeline
    experimental.

## 6. Limitaciones y consideraciones

1.  **Entorno de laboratorio:** el tráfico fue generado en un entorno
    controlado. Sus patrones pueden no representar toda la diversidad,
    configuración y evolución de redes de producción.
2.  **Comportamientos de ataque sintetizados:** el conjunto combina
    tráfico normal con comportamientos de ataque generados mediante
    herramientas; los resultados no garantizan la detección de ataques
    nuevos o no representados.
3.  **Distribución de clases y categorías:** la frecuencia de las clases
    y de las distintas categorías de ataque no es uniforme. Las métricas
    agregadas pueden ocultar un desempeño bajo en categorías poco
    frecuentes.
4.  **Antigüedad del conjunto:** el conjunto fue publicado en 2015; las
    tácticas, técnicas y comportamientos de los atacantes evolucionan
    con el tiempo.
5.  **Duplicados y etiquetas contradictorias:** en el entrenamiento
    local se identificaron repeticiones y combinaciones de
    características con etiquetas binarias diferentes. La limpieza
    aplicada debe describirse con claridad y evaluarse como una decisión
    metodológica, no como una propiedad universal de todos los archivos
    UNSW-NB15.
6.  **Discrepancia de nombres locales:** los nombres de los archivos no
    coinciden con el uso asignado a partir de las dimensiones. Es
    necesario mantener trazabilidad de las rutas y confirmar la
    procedencia de los archivos.
7.  **Generalización:** el rendimiento medido en esta partición no
    garantiza el mismo desempeño en otras redes, periodos,
    infraestructuras o familias de ataques.
8.  **Prevención de fuga de información:** el preprocesamiento debe
    ajustarse solo con los datos de entrenamiento de cada partición o
    pliegue de validación cruzada. Las decisiones finales sobre
    variables y transformaciones deben integrarse en el pipeline de
    evaluación.

## 7. Referencia bibliográfica

Moustafa, N., & Slay, J. (2015). *UNSW-NB15: a comprehensive data set
for network intrusion detection systems (UNSW-NB15 network data set).*
In **2015 Military Communications and Information Systems Conference
(MilCIS)**. IEEE. https://doi.org/10.1109/MilCIS.2015.7348942

## 8. Fuente consultada

Página oficial del conjunto de datos UNSW-NB15, UNSW Research:\
https://research.unsw.edu.au/projects/unsw-nb15-dataset

**Nota para la versión final del informe:** conservar la fecha de
consulta real. Las estadísticas locales se deben volver a comprobar si
cambian los archivos o el procedimiento de limpieza.
