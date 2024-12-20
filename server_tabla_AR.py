import time
import threading
import psycopg2
import json
import numpy as np
import pandas as pd  # Asegúrate de importar pandas
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sklearn.preprocessing import MinMaxScaler
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Función de normalización
def normalizar_minmax(data, feature_range=(0, 1)):
    scaler = MinMaxScaler(feature_range=feature_range)
    return scaler.fit_transform(data.reshape(-1, 1)).flatten()

# Leer configuración desde config4h.json
with open('config4h.json', 'r') as config_file:
    config = json.load(config_file)

# Configuración de la base de datos
db_config = config['database']

# Conexión a PostgreSQL
def get_connection():
    try:
        conn = psycopg2.connect(
            database=db_config['name'],
            user=db_config['user'],
            password=db_config['password'],
            host=db_config['host'],
            port=db_config['port']
        )
        logging.info("Conexión a la base de datos exitosa.")
        return conn
    except Exception as e:
        logging.error(f"Error al conectar a la base de datos: {e}")
        return None

# Crear la tabla ar_input_data si no existe
def crear_tabla():
    conn = get_connection()
    if not conn:
        return
    cursor = conn.cursor()
    try:
        create_table_query = """
        CREATE TABLE IF NOT EXISTS ar_input_data (
            timestamp TIMESTAMP NOT NULL,
            ticker VARCHAR(10) NOT NULL,
            open NUMERIC,
            high NUMERIC,
            low NUMERIC,
            close NUMERIC,
            volume INTEGER,
            distancia_1 DOUBLE PRECISION,
            distancia_2 DOUBLE PRECISION,
            magnitud DOUBLE PRECISION,
            session_market BYTEA,
            day_of_week BYTEA,
            month_of_year BYTEA,
            rsi NUMERIC(10, 5), -- Añadido RSI
            atr NUMERIC(10, 5), -- Añadido ATR
            matriz_indicadores BYTEA,
            PRIMARY KEY (timestamp, ticker)
        );
        """
        cursor.execute(create_table_query)
        conn.commit()
        logging.info("Tabla ar_input_data verificada/creada exitosamente.")
    except Exception as e:
        logging.error(f"Error al crear la tabla ar_input_data: {e}")
    finally:
        cursor.close()
        conn.close()

# Función para calcular el RSI (Relative Strength Index)
def calcular_rsi(datos, periodos=14):
    delta = datos['close'].diff()
    ganancia = (delta.where(delta > 0, 0)).rolling(window=periodos).mean()
    perdida = (-delta.where(delta < 0, 0)).rolling(window=periodos).mean()
    rs = ganancia / perdida
    rsi = 100 - (100 / (1 + rs))
    datos["rsi"] = rsi
    return datos

# Función para calcular el ATR (Average True Range)
def calcular_atr(datos, periodos=14):
    datos["tr1"] = datos["high"] - datos["low"]
    datos["tr2"] = abs(datos["high"] - datos["close"].shift(1))
    datos["tr3"] = abs(datos["low"] - datos["close"].shift(1))
    datos["tr"] = datos[["tr1", "tr2", "tr3"]].max(axis=1)
    datos["atr"] = datos["tr"].rolling(window=periodos).mean()
    return datos

# Función para codificar las sesiones de mercado, días de la semana y meses del año
def codificar_categorias(timestamp):
    # Sesiones del mercado
    hour = timestamp.hour
    if 0 <= hour < 8:
        session_market = [1, 0, 0, 0]  # Asia
    elif 8 <= hour < 17:
        session_market = [0, 1, 0, 0]  # Londres
    elif 13 <= hour < 22:
        session_market = [0, 0, 1, 0]  # Nueva York
    else:
        session_market = [0, 0, 0, 1]  # Pacífico

    # Días de la semana
    day_of_week = [0] * 7
    day_of_week[timestamp.weekday()] = 1

    # Meses del año
    month_of_year = [0] * 12
    month_of_year[timestamp.month - 1] = 1

    return session_market, day_of_week, month_of_year

# Función principal para actualizar la tabla basada en predicciones
def actualizar_tabla():
    conn = get_connection()
    if not conn:
        return

    cursor = conn.cursor()
    try:
        # Consulta para obtener los registros más recientes de `predicciones`
        query = """
            SELECT timestamp, symbol, distancia_1, distancia_2
            FROM predicciones
            ORDER BY timestamp ASC;
        """
        logging.info("Ejecutando consulta SQL para obtener registros de predicciones...")
        cursor.execute(query)
        predicciones = cursor.fetchall()

        if not predicciones:
            logging.warning("No se encontraron registros en la tabla predicciones.")
            return

        datos_procesados = []
        for timestamp, symbol, distancia_1, distancia_2 in predicciones:
            # Obtener los registros de las otras dos tablas usando el timestamp y ticker/symbol
            cursor.execute("""
                SELECT open, high, low, close, volume
                FROM precios_forex_4h
                WHERE timestamp = %s AND ticker = %s;
            """, (timestamp, f"C:{symbol}"))
            precio = cursor.fetchone()

            cursor.execute("""
                SELECT matriz
                FROM indicadores_matrices
                WHERE timestamp = %s AND ticker = %s;
            """, (timestamp, f"C:{symbol}"))
            matriz = cursor.fetchone()

            # Verificar si las otras tablas tienen datos
            if not precio or not matriz:
                logging.info(f"Datos faltantes para timestamp {timestamp} y symbol {symbol}. Omitiendo.")
                continue

            # Procesar los datos
            open_, high, low, close, volume = [x or 0 for x in precio]
            matriz_binaria = matriz[0]
            distancia_1 = distancia_1 or 0
            distancia_2 = distancia_2 or 0
            magnitud = abs(distancia_1 - distancia_2)

            # Calcular RSI y ATR
            datos = pd.DataFrame({
                'timestamp': [timestamp],
                'open': [open_],
                'high': [high],
                'low': [low],
                'close': [close],
                'volume': [volume]
            })
            datos = calcular_rsi(datos)
            datos = calcular_atr(datos)

            # Codificar las categorías
            session_market, day_of_week, month_of_year = codificar_categorias(timestamp)

            datos_procesados.append((timestamp, symbol, open_, high, low, close, volume,
                                     distancia_1, distancia_2, magnitud, 
                                     bytes(session_market), bytes(day_of_week), bytes(month_of_year),
                                     datos['rsi'].iloc[0], datos['atr'].iloc[0],  # Agregar RSI y ATR
                                     matriz_binaria))

        # Insertar datos procesados en `ar_input_data`
        insert_query = """
            INSERT INTO ar_input_data (timestamp, ticker, open, high, low, close, volume, 
                                       distancia_1, distancia_2, magnitud, 
                                       session_market, day_of_week, month_of_year, rsi, atr, matriz_indicadores)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (timestamp, ticker) DO NOTHING;
        """
        if datos_procesados:
            logging.info(f"Insertando {len(datos_procesados)} registros en la tabla ar_input_data...")
            cursor.executemany(insert_query, datos_procesados)
            conn.commit()
            logging.info("Datos insertados exitosamente.")
        else:
            logging.info("No hay datos procesados para insertar.")

    except Exception as e:
        logging.error(f"Error durante la actualización: {e}")
    finally:
        cursor.close()
        conn.close()

# Función periódica para actualizar la tabla cada 10 segundos
def proceso_periodico():
    while True:
        actualizar_tabla()
        time.sleep(10)

# Crear la aplicación FastAPI
app = FastAPI()

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "Server is running"}

@app.get("/status")
def status():
    return {"message": "Table update task is running"}

@app.on_event("startup")
def iniciar_proceso():
    crear_tabla()
    thread = threading.Thread(target=proceso_periodico, daemon=True)
    thread.start()
    logging.info("Proceso periódico iniciado.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
