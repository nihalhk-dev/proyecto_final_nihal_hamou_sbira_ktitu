"""
=============================================================================
PRÁCTICA FINAL — EJERCICIO 2 
Inferencia con Scikit-Learn (Regresión Lineal)
=============================================================================

Salidas generadas en /output:
- ej2_metricas_regresion.txt   → MAE, RMSE, R² y top variables influyentes
- ej2_residuos.png             → Gráfico de residuos (diagnóstico heterocedasticidad)
- ej2_coeficientes.png         → Gráfico de barras de los coeficientes del modelo
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# =============================================================================
# 1. LOAD DATA
# =============================================================================

def load_data(path):
    """
    Carga el dataset desde un fichero CSV y normaliza los nombres de columna.

    Parámetros
    ----------
    path : str — Ruta al fichero CSV

    Retorna
    -------
    df : pd.DataFrame — Dataset cargado con columnas en minúsculas
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    return df


# =============================================================================
# 2. PREPROCESSING
# =============================================================================

def preprocess_data(df):
    """
    Aplica el preprocesamiento necesario para la regresión lineal:
      - Eliminación de filas con valores nulos (no hay ninguno en este dataset,
        pero se aplica como medida de robustez).
      - One-Hot Encoding de las variables categóricas 'education' y 'self_employed',
        con drop_first=True para evitar la trampa de la variable dummy.

    Parámetros
    ----------
    df : pd.DataFrame — Dataset original

    Retorna
    -------
    df : pd.DataFrame — Dataset preprocesado listo para modelado
    """
    df = df.dropna()

    # One-hot encoding: convierte las categóricas en columnas binarias (0/1).
    # drop_first=True elimina una categoría redundante por variable para evitar
    # multicolinealidad perfecta entre las dummies.
    df = pd.get_dummies(df, columns=["education", "self_employed"], drop_first=True)

    return df


# =============================================================================
# 3. TRAIN MODEL
# =============================================================================

def train_model(X_train, y_train):
    """
    Entrena un modelo de Regresión Lineal con Scikit-Learn.

    Parámetros
    ----------
    X_train : pd.DataFrame — Features de entrenamiento (ya escaladas)
    y_train : pd.Series    — Variable objetivo de entrenamiento

    Retorna
    -------
    model : LinearRegression — Modelo entrenado
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


# =============================================================================
# 4. EVALUATION
# =============================================================================

def evaluate_model(y_test, y_pred):
    """
    Calcula las métricas de evaluación del modelo sobre el test set.

    Parámetros
    ----------
    y_test : pd.Series  — Valores reales
    y_pred : np.ndarray — Valores predichos por el modelo

    Retorna
    -------
    mae  : float — Mean Absolute Error
    rmse : float — Root Mean Squared Error
    r2   : float — Coeficiente de determinación R²
    """
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    return mae, rmse, r2


# =============================================================================
# 5. RESIDUAL PLOT
# =============================================================================

def plot_residuals(y_test, y_pred):
    """
    Genera el gráfico de residuos (valores predichos en X, residuos en Y).

    Un buen modelo debería mostrar residuos distribuidos aleatoriamente
    alrededor de cero sin patrones (homocedasticidad). Un patrón de abanico
    indica heterocedasticidad.

    Salida: output/ej2_residuos.png

    Parámetros
    ----------
    y_test : pd.Series  — Valores reales del test set
    y_pred : np.ndarray — Predicciones del modelo
    """
    residuals = y_test - y_pred

    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuals, alpha=0.6, color='royalblue')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel("Valores Predichos")
    plt.ylabel("Residuos")
    # NOTA: Se utiliza el Gráfico de Residuos como herramienta diagnóstica primaria 
    # en lugar de una Matriz de Confusión (exclusiva de clasificación).
    plt.title("Gráfico de Residuos (Análisis de Heterocedasticidad)")
    plt.tight_layout()
    plt.savefig("output/ej2_residuos.png", dpi=150)
    plt.close()


# =============================================================================
# 6. COEFFICIENT BAR CHART
# =============================================================================

def plot_coefficients(importance_df):
    """
    Genera un gráfico de barras horizontales con los coeficientes absolutos
    del modelo ordenados de mayor a menor importancia.

    El valor absoluto del coeficiente (con StandardScaler aplicado) es una
    medida directa de la importancia relativa de cada variable: cuanto mayor,
    más influye esa variable en la predicción del importe del préstamo.

    Salida: output/ej2_coeficientes.png

    Parámetros
    ----------
    importance_df : pd.DataFrame — DataFrame con columnas 'Feature' e 'Importance'
                                   ordenado de mayor a menor importancia
    """
    plt.figure(figsize=(9, 6))

    # Invertimos el orden para que la barra más larga quede arriba
    bars = plt.barh(
        importance_df['Feature'][::-1],
        importance_df['Importance'][::-1],
        color='steelblue',
        edgecolor='white'
    )

    # Añadimos el valor numérico al final de cada barra para facilitar la lectura
    for bar, val in zip(bars, importance_df['Importance'][::-1]):
        plt.text(
            bar.get_width() * 1.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,.0f}",
            va='center', fontsize=8
        )

    plt.xlabel("Coeficiente Absoluto (StandardScaler)")
    plt.title("Importancia de Variables — Regresión Lineal\n(coeficientes absolutos con variables estandarizadas)")
    plt.tight_layout()
    plt.savefig("output/ej2_coeficientes.png", dpi=150)
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs("output", exist_ok=True)

    print("=" * 55)
    print("EJERCICIO 2 — Regresión Lineal")
    print("=" * 55)

    # 1. Cargar datos
    print("\n[1/7] Cargando datos...")
    df = load_data("data/loan_dataset.csv")

    # 2. Preprocesar
    print("[2/7] Preprocesando datos...")
    df = preprocess_data(df)

    # 3. Definir target y features
    # Se eliminan loan_id (no informativo) y loan_status (data leakage: es el resultado
    # del préstamo, no un predictor del importe).
    target = "loan_amount"
    X = df.drop(columns=[target, "loan_id", "loan_status"])
    y = df[target]

    # 4. División Train/Test (80/20) con semilla fija para reproducibilidad
    print("[3/7] Dividiendo dataset (80% train / 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5. Escalado de variables con StandardScaler
    # JUSTIFICACIÓN: Las variables tienen rangos muy distintos (income_annum en millones,
    # cibil_score entre 300 y 900). El escalado permite comparar los coeficientes
    # directamente como medida de importancia relativa de cada variable.
    # IMPORTANTE: se ajusta sobre train y se transforma test (sin data leakage).
    print("[4/7] Aplicando StandardScaler (fit en train, transform en test)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # 6. Entrenar modelo
    print("[5/7] Entrenando modelo de Regresión Lineal...")
    model = train_model(X_train_scaled, y_train)

    # 7. Predecir y evaluar
    print("[6/7] Evaluando modelo en el test set...")
    y_pred = model.predict(X_test_scaled)
    mae, rmse, r2 = evaluate_model(y_test, y_pred)

    # 8. Calcular importancia de variables (coeficientes absolutos)
    # Con StandardScaler aplicado, los coeficientes son directamente comparables:
    # reflejan cuántas unidades cambia loan_amount por cada desviación típica de la feature.
    importance = pd.DataFrame({
        'Feature':    X.columns,
        'Importance': np.abs(model.coef_)
    }).sort_values(by='Importance', ascending=False)

    top_3 = importance.head(3)

    # 9. Guardar métricas y variables influyentes en fichero de texto
    with open("output/ej2_metricas_regresion.txt", "w", encoding="utf-8") as f:
        f.write("MÉTRICAS — REGRESIÓN LINEAL\n")
        f.write("=" * 40 + "\n")
        f.write(f"MAE  : {mae:.4f}\n")
        f.write(f"RMSE : {rmse:.4f}\n")
        f.write(f"R²   : {r2:.4f}\n\n")
        f.write("VARIABLES MÁS INFLUYENTES (Coeficientes Absolutos):\n")
        f.write("-" * 40 + "\n")
        for _, row in top_3.iterrows():
            f.write(f"{row['Feature']}: {row['Importance']:.4f}\n")

    # 10. Generar gráfico de residuos
    print("[7/7] Generando gráficos (residuos y coeficientes)...")
    plot_residuals(y_test, y_pred)

    # 11. Generar gráfico de coeficientes (todas las variables, ordenadas)
    plot_coefficients(importance)

    # Resultados en consola
    print("\nResultados Finales:")
    print(f"  MAE  = {mae:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  R²   = {r2:.4f}")
    print("\nTop 3 Variables Influyentes:")
    print(top_3.to_string(index=False))

    print("\n✅ Outputs generados en /output:")
    print("   → ej2_metricas_regresion.txt")
    print("   → ej2_residuos.png")
    print("   → ej2_coeficientes.png")


if __name__ == "__main__":
    main()