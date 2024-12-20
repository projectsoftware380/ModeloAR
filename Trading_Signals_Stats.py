import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Configuración de la base de datos
with open("config4h.json", "r") as config_file:
    config = json.load(config_file)

DB_CONFIG = {
    "dbname": config["database"]["name"],
    "user": config["database"]["user"],
    "password": config["database"]["password"],
    "host": config["database"]["host"],
    "port": config["database"]["port"],
}

TABLA_SEÑALES = "trading_signals"

# Conectar a la base de datos y obtener los datos de la tabla trading_signals
def cargar_datos():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        query = f"SELECT * FROM {TABLA_SEÑALES}"
        datos = pd.read_sql(query, conn)
        conn.close()
        print("Datos cargados correctamente.")
        return datos
    except Exception as e:
        print(f"Error al cargar datos: {e}")
        return pd.DataFrame()

# Realizar análisis estadístico sobre los datos
def analizar_datos(datos):
    # Filtrar resultados ganados y perdidos
    resultados_validos = datos[datos['result'].isin(['win', 'loss'])]

    # Resumen general
    print("--- Resumen General ---")
    print(resultados_validos['result'].value_counts())
    print("\nPromedio de Pips por Resultado:")
    print(resultados_validos.groupby('result')['result_pips'].mean())

    # Análisis de la relación entre RSI, ATR y resultados
    print("\n--- Análisis de la relación entre RSI, ATR y Resultados ---")
    resultados_validos['rsi_atr_combination'] = resultados_validos['rsi'] * resultados_validos['atr']
    print(resultados_validos.groupby(['rsi_atr_combination'])['result_pips'].mean())

    # Mejor combinación de sl_atr_multiple y tp_sl_ratio
    print("\n--- Mejores Combinaciones de sl_atr_multiple y tp_sl_ratio ---")
    combinaciones = resultados_validos.groupby(['sl_atr_multiple', 'tp_sl_ratio'])['result_pips'].mean().reset_index()
    mejores_combinaciones = combinaciones.sort_values(by='result_pips', ascending=False)
    print(mejores_combinaciones.head(10))

    return mejores_combinaciones

# Generar gráficos
def graficar_estadisticas(datos, combinaciones):
    # Gráfico de barras: distribución de resultados
    plt.figure(figsize=(8, 5))
    sns.countplot(x='result', data=datos)
    plt.title("Distribución de Resultados")
    plt.xlabel("Resultado")
    plt.ylabel("Cantidad")
    plt.show()

    # Gráfico de calor: combinaciones de sl_atr_multiple y tp_sl_ratio
    pivot = combinaciones.pivot("sl_atr_multiple", "tp_sl_ratio", "result_pips")
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt=".1f")
    plt.title("Mejores Combinaciones de sl_atr_multiple y tp_sl_ratio")
    plt.xlabel("tp_sl_ratio")
    plt.ylabel("sl_atr_multiple")
    plt.show()

    # Boxplot: distribución de pips por resultado
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='result', y='result_pips', data=datos)
    plt.title("Distribución de Pips por Resultado")
    plt.xlabel("Resultado")
    plt.ylabel("Pips")
    plt.show()

    # Gráfico de dispersión: RSI vs Pips obtenidos
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='rsi', y='result_pips', data=datos, hue='result', palette="coolwarm", alpha=0.7)
    plt.title("Relación entre RSI y Pips Obtenidos")
    plt.xlabel("RSI")
    plt.ylabel("Pips")
    plt.show()

    # Gráfico de dispersión: ATR vs Pips obtenidos
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='atr', y='result_pips', data=datos, hue='result', palette="coolwarm", alpha=0.7)
    plt.title("Relación entre ATR y Pips Obtenidos")
    plt.xlabel("ATR")
    plt.ylabel("Pips")
    plt.show()

# Función principal
def main():
    datos = cargar_datos()
    if not datos.empty:
        combinaciones = analizar_datos(datos)
        graficar_estadisticas(datos, combinaciones)
    else:
        print("No se pudieron cargar los datos.")

if __name__ == "__main__":
    main()
