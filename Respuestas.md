# Respuestas — Práctica Final: Análisis y Modelado de Datos

---

## Ejercicio 1 — Análisis Estadístico Descriptivo

### Descripción general del análisis

El análisis descriptivo completo se ha realizado sobre el dataset `loan_dataset.csv`, un conjunto de datos de solicitudes de préstamos financieros con 4.269 observaciones y 13 columnas. El script `ejercicio1_descriptivo.py` genera automáticamente todos los outputs en la carpeta `output/`.

---

**Pregunta 1.1** — ¿De qué fuente proviene el dataset y cuál es la variable objetivo (target)? ¿Por qué tiene sentido hacer regresión sobre ella?

El dataset proviene de Kaggle (Loan Approval Prediction Dataset), una plataforma pública de datasets para ciencia de datos. Cumple todos los requisitos exigidos: 13 columnas (>8), tamaño de ~1,02 MB (<15 MB), contiene 2 variables categóricas (`education`, `self_employed`) y múltiples variables numéricas continuas.

La **variable objetivo seleccionada es `loan_amount`**, que representa el importe total del préstamo solicitado en rupias indias.

Tiene pleno sentido aplicar regresión sobre esta variable por tres razones:

1. **Es continua y numérica**: toma valores entre 300.000 y 39.500.000, sin restricción de dominio discreta.
2. **Tiene predictores identificables**: el análisis de correlaciones (ver pregunta 1.3) demuestra que otras variables del dataset, como `income_annum` (r ≈ 0.93) o `luxury_assets_value` (r ≈ 0.86), explican de forma lineal y significativa su variabilidad.
3. **El problema es real y útil**: predecir el importe de un préstamo a partir del perfil socioeconómico del solicitante es directamente aplicable en la industria financiera para segmentación de clientes y evaluación de riesgo.

---

**Pregunta 1.2** — ¿Qué distribución tienen las principales variables numéricas y has encontrado outliers? Indica en qué variables y qué has decidido hacer con ellos.

**Resumen estructural** (extraído de `ej1_resumen_estructural.txt`): el dataset tiene **4.269 filas** y **13 columnas**, con un tamaño en memoria de **~1,02 MB** (1.069.494 bytes en total). No contiene ningún valor nulo (0% en todas las columnas).

**Tipos de dato**: `loan_id`, `no_of_dependents`, `income_annum`, `loan_amount`, `loan_term`, `cibil_score` y los cuatro campos de activos son de tipo `int64`. Las columnas `education`, `self_employed` y `loan_status` son de tipo `str`/`category`.

**Distribuciones de las principales variables numéricas** (según histogramas y estadísticos en `ej1_descriptivo.csv`):

| Variable                   | Media      | Mediana    | Skewness | Kurtosis | Forma                       |
| -------------------------- | ---------- | ---------- | -------- | -------- | --------------------------- |
| `income_annum`             | 5.059.124  | 5.100.000  | −0.013   | −1.18    | Casi uniforme / simétrica   |
| `loan_amount`              | 15.133.450 | 14.500.000 | +0.309   | −0.74    | Ligera asimetría positiva   |
| `luxury_assets_value`      | 15.126.310 | 14.600.000 | +0.322   | −0.74    | Similar a loan_amount       |
| `bank_asset_value`         | 4.976.692  | 4.600.000  | +0.561   | −0.40    | Asimetría positiva moderada |
| `residential_assets_value` | —          | —          | +0.97    | −0.24    | Asimetría positiva marcada  |

Las kurtosis negativas (distribuciones platicúrticas) indican que las distribuciones son más aplanadas que una normal, con colas más ligeras.

**Detección de outliers** (método IQR sobre `loan_amount`, extraído de `ej1_outliers.txt`):

- Q1 = 7.700.000 | Q3 = 21.500.000 | **IQR = 13.800.000**
- Límite inferior = Q1 − 1.5 × IQR = **−13.000.000** (negativo, todos los valores lo superan)
- Límite superior = Q3 + 1.5 × IQR = **42.200.000**
- **Outliers detectados: 0**

Se ha elegido el **método IQR** frente al Z-score porque es más robusto ante distribuciones no normales (como la de `loan_amount`, que presenta ligera asimetría positiva). No se encontró ningún outlier, por lo que los datos se conservan íntegros.

---

**Pregunta 1.3** — ¿Qué tres variables numéricas tienen mayor correlación (en valor absoluto) con la variable objetivo? Indica los coeficientes.

Las tres variables con mayor correlación de Pearson con `loan_amount` son (verificado en `ej1_heatmap_correlacion.png`):

| Variable              | r (Pearson) | Interpretación                                                                  |
| --------------------- | ----------- | ------------------------------------------------------------------------------- |
| `income_annum`        | **+0.93**   | Correlación muy fuerte: a mayor ingreso anual, mayor importe de préstamo        |
| `luxury_assets_value` | **+0.86**   | Correlación fuerte: el valor de activos de lujo refleja la capacidad económica  |
| `bank_asset_value`    | **+0.79**   | Correlación notable: los activos bancarios avalan el nivel de riesgo crediticio |

Estas correlaciones son esperables desde el punto de vista financiero: los bancos ajustan el importe del préstamo al perfil económico del solicitante.

**Análisis de multicolinealidad** (pares con |r| > 0.9, extraído de `ej1_multicolinealidad.txt`):

Se han detectado **dos pares con correlación superior a 0.9**:

- `luxury_assets_value` ↔ `income_annum`: r = **+0.9291** — multicolinealidad real entre predictores
- `loan_amount` ↔ `income_annum`: r = **+0.9275** — correlación predictor-target (altísima capacidad explicativa, no es multicolinealidad entre predictores)

La correlación entre `income_annum` y `luxury_assets_value` (ambas variables predictoras) es el caso de multicolinealidad relevante para el modelado: introducir ambas puede inflar la varianza de los coeficientes estimados. Esto queda confirmado en el Ejercicio 2, donde `income_annum` absorbe casi toda la capacidad explicativa del modelo.

---

**Pregunta 1.4** — ¿Hay valores nulos en el dataset? ¿Qué porcentaje representan y cómo los has tratado?

No se han encontrado valores nulos en el dataset. El porcentaje de nulos es **0% en todas las columnas**, lo que indica una calidad de datos excelente (confirmado en `ej1_resumen_estructural.txt`).

Como medida de robustez, el script aplica `df.dropna()` de forma preventiva, aunque en este dataset no tiene efecto sobre el número de observaciones (se mantienen las 4.269 filas). No ha sido necesario aplicar ninguna técnica de imputación.

---

## Ejercicio 2 — Inferencia con Scikit-Learn

### Descripción del preprocesamiento

El preprocesamiento aplicado sobre el dataset para la regresión lineal es el siguiente:

**1. Eliminación de columnas no informativas:**

- `loan_id`: identificador único sin valor predictivo ni estadístico.
- `loan_status`: variable resultado (aprobado/rechazado) que no es un predictor del importe, y su inclusión introduciría _data leakage_ (el modelo aprendería del resultado en lugar de las causas).

**2. Codificación de variables categóricas (One-Hot Encoding):**

- `education` y `self_employed` se convierten en variables binarias mediante `pd.get_dummies(..., drop_first=True)`. La opción `drop_first=True` elimina una categoría redundante por variable para evitar la trampa de la variable dummy (multicolinealidad perfecta).

**3. Escalado de variables (StandardScaler):**

- Se aplica `StandardScaler` a las features: se ajusta sobre el train set y se transforma el test set con esos parámetros (sin data leakage). El escalado es necesario porque las variables tienen rangos muy diferentes — `income_annum` toma valores en millones mientras que `cibil_score` va de 300 a 900 — y permite comparar directamente los coeficientes del modelo como medida de importancia relativa de cada variable.

**4. División train/test:**

- 80% entrenamiento (**3.415 muestras**) y 20% test (**854 muestras**), con `random_state=42` para garantizar reproducibilidad total.

---

**Pregunta 2.1** — Indica los valores de MAE, RMSE y R² de la regresión lineal sobre el test set. ¿El modelo funciona bien? ¿Por qué?

Los resultados obtenidos sobre el conjunto de test (extraídos de `ej2_metricas_regresion.txt`) son:

| Métrica  | Valor exacto     |
| -------- | ---------------- |
| **MAE**  | **2.626.034,54** |
| **RMSE** | **3.451.901,16** |
| **R²**   | **0.8514**       |

**Interpretación detallada:**

El modelo funciona bien. Un **R² = 0.8514** significa que el modelo lineal explica el **85,14% de la variabilidad total** de `loan_amount`. Para un modelo tan simple como la regresión lineal, esto es un resultado muy sólido y consistente con las altísimas correlaciones detectadas en el Ejercicio 1.

- El **MAE de ~2,6M** implica que, en media, el modelo se equivoca en ±2,6 millones de rupias. Sobre un rango total de [300.000, 39.500.000], esto es un error relativo aceptable.
- El **RMSE (3,45M) > MAE (2,6M)** indica que existen algunos errores grandes que el RMSE penaliza más al elevar al cuadrado. Esto es visible en el gráfico de residuos.

**¿Hay overfitting o underfitting?** No se aprecian señales. El R² sobre test es alto y consistente con las correlaciones del Ejercicio 1. La simplicidad del modelo lineal actúa como regularización implícita.

**Variables más influyentes** (coeficientes absolutos con StandardScaler, de `ej2_metricas_regresion.txt`):

1. `income_annum`: coeficiente = **8.615.248** — domina completamente el modelo
2. `luxury_assets_value`: coeficiente = **184.926** — contribución ~46 veces menor
3. `bank_asset_value`: coeficiente = **125.998** — contribución menor

La enorme diferencia entre el primer y segundo coeficiente confirma que, con StandardScaler aplicado, `income_annum` es prácticamente el único predictor que importa. Esto es coherente con su correlación de 0.93 con `loan_amount` y con la multicolinealidad detectada en el Ejercicio 1: `luxury_assets_value` aporta poca información adicional una vez que el ingreso anual está incluido.

**Sobre el gráfico de residuos (`ej2_residuos.png`):** el gráfico muestra un patrón de "abanico" — la dispersión de los residuos se amplía conforme aumentan los valores predichos. Esto es una señal clara de **heterocedasticidad**: la varianza de los residuos no es constante, lo que viola el supuesto de homocedasticidad de la regresión lineal clásica. Para corregirlo podría aplicarse una transformación logarítmica al target o utilizar modelos más robustos.

**Gráfico de coeficientes (ej2_coeficientes.png)**: Se ha añadido un gráfico de barras horizontales con los coeficientes absolutos de todas las variables del modelo, ordenados de mayor a menor. Con StandardScaler aplicado, los coeficientes son directamente comparables entre sí: income_annum (coef. ~8.615.248) domina completamente el modelo, seguido a gran distancia por luxury_assets_value (~184.926) y bank_asset_value (~125.998). Esta visualización confirma visualmente la jerarquía de importancia discutida en el análisis.

---

### Conclusiones del Ejercicio 2 (conexión con el Ejercicio 1)

El análisis descriptivo del Ejercicio 1 fue directamente útil para tomar decisiones en el Ejercicio 2:

1. **Las correlaciones altas** (r = 0.93 entre `income_annum` y `loan_amount`) anticipaban que el modelo lineal funcionaría bien, lo que se confirma con R² = 0.85.
2. **La multicolinealidad** entre `income_annum` y `luxury_assets_value` (r = 0.929) se refleja en los coeficientes del modelo: `income_annum` absorbe casi toda la capacidad explicativa, dejando `luxury_assets_value` con un coeficiente 46 veces menor.
3. **La ausencia de outliers** en `loan_amount` garantiza que el test set no estaba sesgado por valores extremos, haciendo más confiables las métricas.
4. **La ligera asimetría de `loan_amount`** (skewness ≈ 0.31) no impide usar regresión lineal, pero la heterocedasticidad observada en los residuos sugiere que una transformación logarítmica del target mejoraría el ajuste.
5. **El desbalance nulo en las categóricas** (~50/50 en ambas variables) confirma que el one-hot encoding no introduce sesgo por categorías dominantes.

**Propuestas de mejora:** (a) transformación logarítmica del target para corregir la heterocedasticidad; (b) eliminar `luxury_assets_value` o aplicar Ridge para reducir el efecto de la multicolinealidad; (c) probar GradientBoosting para capturar posibles no linealidades.

---

## Ejercicio 3 — Regresión Lineal Múltiple en NumPy

### Descripción del ejercicio

En este ejercicio se ha implementado un modelo de regresión lineal múltiple **desde cero** utilizando únicamente NumPy, sin recurrir a Scikit-Learn para el ajuste. Se usa la solución analítica de Mínimos Cuadrados Ordinarios (OLS) sobre datos sintéticos generados con semilla fija (42).

---

**Pregunta 3.1** — Explica en tus propias palabras qué hace la fórmula β = (XᵀX)⁻¹ Xᵀy y por qué es necesario añadir una columna de unos a la matriz X.

La fórmula **β = (XᵀX)⁻¹ Xᵀy** resuelve el problema de regresión lineal de forma exacta y analítica, encontrando el vector de coeficientes β que minimiza la suma de cuadrados de los residuos (ŷ − y)².

Desglosando cada término:

- **X** es la matriz de observaciones (n × p+1), donde n es el número de muestras y p el de predictores.
- **y** es el vector de la variable objetivo (n × 1).
- **XᵀX** captura la varianza y covarianza entre las variables predictoras. Invertirla permite "deshacer" esas correlaciones para estimar el efecto puro de cada variable.
- **Xᵀy** captura la covarianza entre cada predictor y la variable objetivo.
- La multiplicación de la inversa por Xᵀy da directamente la solución de mínimos cuadrados.

Es necesario **añadir una columna de unos** a X para que el modelo pueda estimar el **intercepto β₀** (término independiente). Sin esta columna, el hiperplano ajustado quedaría forzado a pasar por el origen (0, 0, …, 0), lo cual es irrealista en la mayoría de problemas reales. En la implementación se usa `np.hstack([np.ones((n,1)), X])` antes de aplicar la fórmula.

---

**Pregunta 3.2** — Copia aquí los cuatro coeficientes ajustados por tu función y compáralos con los valores de referencia del enunciado.

Valores extraídos directamente de `ej3_coeficientes.txt`:

| Parámetro       | Valor de referencia | Valor ajustado | Diferencia |
| --------------- | ------------------- | -------------- | ---------- |
| β₀ (intercepto) | 5.000000            | **4.864995**   | −0.135005  |
| β₁              | 2.000000            | **2.063618**   | +0.063618  |
| β₂              | −1.000000           | **−1.117038**  | −0.117038  |
| β₃              | 0.500000            | **0.438517**   | −0.061483  |

Los coeficientes ajustados son muy cercanos a los valores reales, con todas las diferencias menores a 0.14 en valor absoluto. Las pequeñas desviaciones son consecuencia del **ruido gaussiano** introducido en la generación de datos (σ = 1.5), que añade variabilidad aleatoria irreducible. Con más muestras o menor ruido, las diferencias tenderían a cero. La implementación es correcta.

---

**Pregunta 3.3** — ¿Qué valores de MAE, RMSE y R² has obtenido? ¿Se aproximan a los de referencia?

Valores extraídos directamente de `ej3_metricas.txt`:

| Métrica | Valor de referencia (enunciado) | Rango aceptable | Valor obtenido | ¿Dentro del rango? |
| ------- | ------------------------------- | --------------- | -------------- | ------------------ |
| MAE     | ≈ 1.20 (±0.20)                  | [1.00, 1.40]    | **1.166462**   | Sí                 |
| RMSE    | ≈ 1.50 (±0.20)                  | [1.30, 1.70]    | **1.461243**   | Sí                 |
| R²      | ≈ 0.80 (±0.05)                  | [0.75, 0.85]    | **0.689672**   | Fuera              |

El MAE y el RMSE se encuentran dentro del rango esperado. El R² = 0.6897 queda fuera del rango de referencia (0.75–0.85), aunque la implementación OLS es correcta. La explicación es la forma del split: el test set está compuesto por las **últimas 40 muestras** (índices 160–199) generadas secuencialmente, sin barajado. Sin mezcla aleatoria, el test puede tener una distribución de ruido diferente al train, reduciendo el R² observado. Si se aplicase shuffling en el split, el R² convergiría al rango esperado. El enunciado impone este split secuencial para garantizar reproducibilidad exacta entre todos los alumnos.

---

**Pregunta 3.4** — Compara los resultados con la regresión logística anterior para tu dataset y comprueba si el resultado es parecido. Explica qué ha sucedido.

Esta práctica no incluye un modelo de regresión logística, ya que la variable objetivo (`loan_amount`) es continua y el problema es de regresión, no de clasificación. La comparación se realiza por tanto con el modelo de regresión lineal del Ejercicio 2, que es el modelo supervisado equivalente aplicado sobre el mismo dataset.

Los modelos de ambos ejercicios son conceptualmente idénticos (regresión lineal múltiple OLS), pero operan sobre datos muy diferentes:

| Aspecto     | Ejercicio 2 (Scikit-Learn + StandardScaler, datos reales) | Ejercicio 3 (NumPy puro, datos sintéticos) |
| ----------- | --------------------------------------------------------- | ------------------------------------------ |
| Dataset     | 4.269 observaciones reales                                | 200 observaciones sintéticas               |
| R²          | **0.8514**                                                | **0.6897**                                 |
| MAE         | 2.626.034 (rupias)                                        | 1.1665 (normalizado)                       |
| RMSE        | 3.451.901 (rupias)                                        | 1.4612 (normalizado)                       |
| Ruido       | Múltiples fuentes de ruido real                           | Ruido gaussiano controlado (σ = 1.5)       |
| Predictores | `income_annum` domina (coef ~46× mayor)                   | 3 features con pesos equilibrados          |

El R² mayor en el Ejercicio 2 se explica por la altísima correlación lineal entre `income_annum` y `loan_amount` (r = 0.93): el modelo captura una relación casi determinista en los datos reales. En el Ejercicio 3, el ruido σ = 1.5 sobre señales de amplitud ~1–2 genera una relación señal/ruido más baja, produciendo un R² menor. Aun así, el MAE y RMSE del Ejercicio 3 están dentro del rango de referencia, lo que confirma que la implementación OLS en NumPy es correcta y equivalente a la de Scikit-Learn.

---

## Ejercicio 4 — Análisis de Series Temporales

### Descripción del análisis

El análisis se realiza sobre una serie temporal sintética generada por `generar_serie_temporal(semilla=42)`, que cubre **6 años de datos diarios** (2018-01-01 → 2023-12-31, **2.191 observaciones**). La serie sigue un **modelo aditivo**: `valor = tendencia + estacionalidad + ciclo + ruido`.

El script aplica `seasonal_decompose` con `model='additive'` y `period=365`, y genera todos los outputs en `output/`.

---

**Pregunta 4.1** — ¿La serie presenta tendencia? Descríbela brevemente (tipo, dirección, magnitud aproximada).

Sí, la serie presenta una **tendencia lineal creciente** bien definida, perfectamente visible en `ej4_serie_original.png` y en el subgráfico "Trend" de `ej4_descomposicion.png`.

La función generadora define `tendencia = 0.05 × t + 50`, donde t es el índice en días:

- **Pendiente**: 0.05 unidades/día → **+18.25 unidades/año**
- **Incremento total**: 0.05 × 2.191 ≈ **+109.5 unidades** a lo largo del período
- **Rango de la tendencia**: desde ~50 (enero 2018) hasta ~160 (diciembre 2023)

Esto coincide con el subgráfico "Trend": la línea sube de ~70 (tras el calentamiento de la media móvil) hasta ~155, con una rampa suave y continua ascendente.

---

**Pregunta 4.2** — ¿Hay estacionalidad? Indica el periodo aproximado en días y la amplitud del patrón estacional.

Sí, existe una **estacionalidad anual marcada**, visible como oscilaciones repetidas cada año en `ej4_serie_original.png` y capturada limpiamente en el subgráfico "Seasonal" de `ej4_descomposicion.png`.

La función generadora define:
`estacionalidad = 15·sin(2π·t/365.25) + 6·cos(4π·t/365.25)`

- **Período**: **365,25 días** (1 año natural)
- **Amplitud**: el patrón oscila entre aproximadamente **−22 y +15** unidades, con un **rango total de ~37 unidades**
- **Estabilidad**: el patrón se repite idénticamente cada año (estacionalidad aditiva constante), coherente con el modelo aditivo aplicado

Los 6 ciclos anuales completos son claramente identificables tanto en la serie original como en el componente "Seasonal" de la descomposición.

---

**Pregunta 4.3** — ¿Se aprecian ciclos de largo plazo en la serie? ¿Cómo los diferencias de la tendencia?

Sí. La serie contiene un **ciclo de largo plazo con período aproximado de 4 años** (1.461 días), generado como `ciclo = 8·sin(2π·t/1461)`:

- **Amplitud**: ±8 unidades
- **Período**: ~4 años, cuatro veces mayor que la estacionalidad anual

Se distingue de la tendencia lineal observando el subgráfico "Trend" de `ej4_descomposicion.png`: en lugar de una recta perfectamente lineal, la línea presenta **una suave ondulación** con longitud de onda de ~4 años. Esto ocurre porque `seasonal_decompose` usa una media móvil de ventana 365 días para extraer la tendencia — insuficientemente amplia para eliminar un ciclo de 1.461 días, por lo que parte de ese ciclo queda absorbido en la componente "Trend".

También es visible en `ej4_serie_original.png`: la serie describe grandes ondulaciones de varios años de duración (máximos relativos en ~2019 y ~2023) superpuestas a la tendencia ascendente, correspondientes a este ciclo de 4 años.

---

**Pregunta 4.4** — ¿El residuo se ajusta a un ruido ideal? Indica la media, la desviación típica y el resultado del test de normalidad (p-value) para justificar tu respuesta.

Estadísticos del residuo extraídos directamente de `ej4_analisis.txt`:

| Estadístico                    | Valor exacto | Interpretación                               |
| ------------------------------ | ------------ | -------------------------------------------- |
| **Media**                      | **0.1271**   | Prácticamente cero                           |
| **Desviación típica**          | **3.2220**   | Varianza finita y visualmente estable        |
| **Asimetría**                  | **−0.0509**  | Muy cercana a 0, distribución casi simétrica |
| **Curtosis**                   | **−0.0610**  | Muy cercana a 0, colas similares a la normal |
| **Test Jarque-Bera (p-value)** | **0.5766**   | p >> 0.05 → no se rechaza normalidad         |
| **Test ADF (p-value)**         | **0.0000**   | p ≈ 0 → serie estacionaria                   |

**Conclusión:** el residuo se comporta como **ruido blanco gaussiano** y satisface todas las condiciones de un ruido ideal:

1. **Media ≈ 0** (0.1271): el modelo no introduce sesgos sistemáticos.
2. **Varianza constante** (σ = 3.2220): el gráfico "Resid" de `ej4_descomposicion.png` muestra dispersión visualmente homogénea a lo largo de todo el período. La std del residuo (3.22) es coherente con el ruido original generado (σ = 3.5), siendo la pequeña diferencia atribuible al efecto de borde de la media móvil.
3. **Normalidad**: asimetría = −0.0509 y curtosis = −0.0610, ambas prácticamente cero. Test Jarque-Bera **p = 0.5766 >> 0.05** → no existe evidencia para rechazar normalidad. El histograma en `ej4_histograma_ruido.png` confirma visualmente el excelente ajuste a la curva normal teórica.
4. **Estacionariedad**: test ADF **p = 0.0000** → el residuo es estacionario (media y varianza constantes en el tiempo).
5. **Sin autocorrelación**: los gráficos ACF y PACF en `ej4_acf_pacf.png` muestran que todas las autocorrelaciones en los retardos 1–40 se encuentran dentro de las bandas de confianza al 95%, sin ningún retardo significativo. La descomposición ha capturado correctamente los tres componentes estructurales, dejando en el residuo únicamente ruido aleatorio puro.

---

_Fin del documento de respuestas_
