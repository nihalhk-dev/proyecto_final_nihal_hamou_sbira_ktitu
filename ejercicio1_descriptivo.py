"""
=============================================================================
PRÁCTICA FINAL — EJERCICIO 1 
Análisis Estadístico Descriptivo
=============================================================================

Descripción:
Este script realiza un análisis descriptivo completo sobre un dataset de préstamos,
incluyendo estadísticas, distribuciones, variables categóricas, correlaciones y
detección de outliers.

Salidas generadas en /output:
- ej1_resumen_estructural.txt
- ej1_descriptivo.csv
- ej1_histogramas.png
- ej1_boxplots.png
- ej1_categoricas.png
- ej1_categoricas.txt
- ej1_heatmap_correlacion.png
- ej1_multicolinealidad.txt 
- ej1_outliers.txt
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# =============================================================================
# 1. LOAD DATA & STRUCTURAL SUMMARY
# =============================================================================

def load_data(path):
    df = pd.read_csv(path)
    # Normalizamos nombres de columna: sin espacios y en minúsculas
    df.columns = df.columns.str.strip().str.lower()
    return df

def structural_summary(df):
    """
    Genera un resumen estructural del DataFrame (filas, columnas, tipos de dato y memoria).
    """
    with open("output/ej1_resumen_estructural.txt", "w", encoding="utf-8") as f:
        f.write("RESUMEN ESTRUCTURAL DEL DATASET\n")
        f.write("=" * 40 + "\n")
        f.write(f"Filas (Observaciones): {df.shape[0]}\n")
        f.write(f"Columnas (Variables): {df.shape[1]}\n\n")
        f.write("Tipos de datos:\n")
        f.write(df.dtypes.to_string() + "\n\n")
        f.write("Uso de memoria (bytes):\n")
        f.write(df.memory_usage(deep=True).to_string() + "\n")


# =============================================================================
# 2. PREPROCESSING
# =============================================================================

def preprocess_data(df):
    categoricas = ["education", "self_employed"]

    # Convertimos a tipo category para un manejo correcto en gráficos y estadísticos
    for col in categoricas:
        df[col] = df[col].astype("category")

    # Eliminación preventiva de nulos (0 filas afectadas en este dataset)
    df = df.dropna()
    return df, categoricas


# =============================================================================
# 3. DESCRIPTIVE STATS
# =============================================================================

def descriptive_stats(df, numericas):
    desc = df.describe()
    # Añadimos skewness y kurtosis al resumen estándar de describe()
    # Skewness > 0: cola larga a la derecha. Kurtosis < 0: distribución más plana que la normal
    desc.loc["skew"] = df[numericas].skew()
    desc.loc["kurtosis"] = df[numericas].kurtosis()
    desc.to_csv("output/ej1_descriptivo.csv")


# =============================================================================
# 4. HISTOGRAMS
# =============================================================================

def plot_histograms(df, numericas):
    plt.figure(figsize=(12, 10))
    n_cols = 2
    n_rows = int(np.ceil(len(numericas) / n_cols))

    for i, col in enumerate(numericas, 1):
        plt.subplot(n_rows, n_cols, i)
        # kde=True superpone la curva de densidad estimada para ver la forma de la distribución
        sns.histplot(df[col], bins=30, kde=True)

    plt.suptitle("Histogramas con KDE")
    plt.tight_layout()
    plt.savefig("output/ej1_histogramas.png", dpi=150)
    plt.close()


# =============================================================================
# 5. BOXPLOTS
# =============================================================================

def plot_boxplots(df, categoricas, target):
    plt.figure(figsize=(10, 5))

    for i, col in enumerate(categoricas, 1):
        plt.subplot(1, len(categoricas), i)
        sns.boxplot(x=df[col], y=df[target])

    plt.tight_layout()
    plt.savefig("output/ej1_boxplots.png", dpi=150)
    plt.close()


# =============================================================================
# 6. CATEGORICAL VARIABLES
# =============================================================================

def plot_categoricals(df, categoricas):
    plt.figure(figsize=(10, 5))

    with open("output/ej1_categoricas.txt", "w", encoding="utf-8") as f:
        f.write("ANÁLISIS DE VARIABLES CATEGÓRICAS\n")
        f.write("=" * 40 + "\n")

        for i, col in enumerate(categoricas, 1):
            plt.subplot(1, len(categoricas), i)
            counts = df[col].value_counts()
            proportions = df[col].value_counts(normalize=True)

            counts.plot(kind="bar")

            f.write(f"\nVariable: {col}\n")
            f.write(str(proportions) + "\n")

            # Alerta de desbalance: si una categoría supera el 70% puede sesgar el modelo
            if proportions.max() > 0.7:
                f.write(" Posible desbalance (una categoría domina)\n")

    plt.tight_layout()
    plt.savefig("output/ej1_categoricas.png", dpi=150)
    plt.close()


# =============================================================================
# 7. CORRELATION HEATMAP & MULTICOLLINEARITY
# =============================================================================

def plot_correlation(df):
    corr = df.corr(numeric_only=True)

    # Iteramos el triángulo inferior para evitar duplicados en los pares
    high_corr = []
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > 0.9:
                high_corr.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))

    with open("output/ej1_multicolinealidad.txt", "w", encoding="utf-8") as f:
        f.write("ANÁLISIS DE MULTICOLINEALIDAD (|r| > 0.9)\n")
        f.write("=" * 40 + "\n")
        if high_corr:
            for var1, var2, r in high_corr:
                f.write(f"Alerta: {var1} y {var2} tienen una correlación de r = {r:.4f}\n")
        else:
            f.write("No se encontraron pares de variables con correlación absoluta > 0.9.\n")

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.savefig("output/ej1_heatmap_correlacion.png", dpi=150)
    plt.close()

    return corr

# =============================================================================
# 8. OUTLIERS DETECTION (IQR)
# =============================================================================

def detect_outliers(df, target):
    Q1 = df[target].quantile(0.25)
    Q3 = df[target].quantile(0.75)
    IQR = Q3 - Q1

    # Regla de Tukey: outliers fuera de [Q1 - 1.5·IQR, Q3 + 1.5·IQR]
    # Se usa IQR en lugar de Z-score por ser más robusto ante distribuciones asimétricas
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[target] < lower) | (df[target] > upper)]

    with open("output/ej1_outliers.txt", "w", encoding="utf-8") as f:
        f.write("DETECCIÓN DE OUTLIERS (IQR)\n")
        f.write("=" * 40 + "\n")
        f.write(f"IQR: {IQR}\n")
        f.write(f"Límite inferior (Lower bound): {lower}\n")
        f.write(f"Límite superior (Upper bound): {upper}\n")
        f.write(f"Número total de outliers: {len(outliers)}\n")
        f.write(f"Variable analizada: {target}\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs("output", exist_ok=True)

    print("=" * 55)
    print("EJERCICIO 1 — Análisis Estadístico Descriptivo")
    print("=" * 55)

    print("\n[1/8] Cargando datos...")
    df = load_data("data/loan_dataset.csv")

    print("[2/8] Generando resumen estructural...")
    structural_summary(df)

    print("[3/8] Preprocesando datos...")
    df, categoricas = preprocess_data(df)

    target = "loan_amount"
    numericas = df.select_dtypes(include=np.number).columns

    print("[4/8] Estadísticos descriptivos...")
    descriptive_stats(df, numericas)

    print("[5/8] Generando histogramas y boxplots...")
    plot_histograms(df, numericas)
    plot_boxplots(df, categoricas, target)

    print("[6/8] Variables categóricas...")
    plot_categoricals(df, categoricas)

    print("[7/8] Correlaciones y Multicolinealidad...")
    plot_correlation(df)

    print("[8/8] Detección de outliers...")
    detect_outliers(df, target)

    print("\n Todos los outputs generados en la carpeta /output")


if __name__ == "__main__":
    main()
