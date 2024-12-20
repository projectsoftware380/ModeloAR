import psycopg2
import pandas as pd
import json
import logging
from psycopg2.extras import execute_values
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

def manejar_nan(matriz_indicadores, estrategia="reemplazar_por_0"):
    try:
        if estrategia == "reemplazar_por_0":
            matriz_indicadores = np.nan_to_num(matriz_indicadores, nan=0)
        elif estrategia == "reemplazar_por_media":
            media_columnas = np.nanmean(matriz_indicadores, axis=0)
            inds = np.where(np.isnan(matriz_indicadores))
            matriz_indicadores[inds] = np.take(media_columnas, inds[1])
        else:
            raise ValueError(f"Estrategia '{estrategia}' no reconocida.")
        return matriz_indicadores
    except Exception as e:
        logging.error(f"Error al manejar NaN: {e}")
        raise

def procesar_matriz_indicadores(matriz_indicadores, estrategia_nan="reemplazar_por_0"):
    try:
        matriz_indicadores = np.asarray(matriz_indicadores, dtype=np.float64)
        if matriz_indicadores.size == 0:
            logging.warning("La matriz de indicadores está vacía. No se puede procesar.")
            return None

        # Ajustar la forma según el pipeline original
        if matriz_indicadores.ndim == 3:
            matriz_indicadores = matriz_indicadores.reshape(matriz_indicadores.shape[0], -1)
        elif matriz_indicadores.ndim == 1:
            matriz_indicadores = matriz_indicadores.reshape(1, -1)
        elif matriz_indicadores.ndim != 2:
            logging.error(f"La matriz de indicadores no es 1D ni 2D después de ajustar. Dimensiones: {matriz_indicadores.ndim}")
            return None

        # Manejar NaN
        matriz_indicadores = manejar_nan(matriz_indicadores, estrategia=estrategia_nan)
        if matriz_indicadores.size == 0:
            logging.warning("Después de manejar NaN, la matriz de indicadores está vacía.")
            return None

        # Aplanar a 1D si así se requirió en el entrenamiento
        vector = matriz_indicadores.flatten()
        return vector
    except Exception as e:
        logging.error(f"Error al procesar la matriz de indicadores: {e}")
        return None

# Configuración de logging
log_filename = "generate_trading_signals.log"
logging.basicConfig(
    filename=log_filename,
    filemode="w",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.info("Inicio del script de generación de señales de trading.")

# Cargar configuración
with open("config4h.json", "r") as config_file:
    config = json.load(config_file)

DB_CONFIG = {
    "dbname": config["database"]["name"],
    "user": config["database"]["user"],
    "password": config["database"]["password"],
    "host": config["database"]["host"],
    "port": config["database"]["port"],
}

TABLA_DATOS = "test_data"
TABLA_SEÑALES = "trading_signals"  # Asegurarnos de usar esta tabla para guardar señales.

currency_pair = "EURUSD"
model_path = f"D:/TradingIA/venv/ModeloAR/{currency_pair}/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)
        return mean, log_std

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q_value = nn.Linear(hidden_dim, 1)
    def forward(self, state, action):
        action = action.view(action.size(0), -1)
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q_value(x)

STATE_DIM = 159
ACTION_DIM = 1
HIDDEN_DIM = 32

actor = Actor(STATE_DIM, ACTION_DIM, HIDDEN_DIM).to(DEVICE)
critic1 = Critic(STATE_DIM, ACTION_DIM, HIDDEN_DIM).to(DEVICE)
critic2 = Critic(STATE_DIM, ACTION_DIM, HIDDEN_DIM).to(DEVICE)

def cargar_estado_modelo(modelo, ruta_modelo, dispositivo):
    try:
        estado = torch.load(ruta_modelo, map_location=dispositivo, weights_only=True)
        logging.debug(f"Modelo cargado con weights_only=True desde {ruta_modelo}.")
    except TypeError:
        import warnings
        warnings.warn("El parámetro 'weights_only' no es soportado. Cargando sin él.")
        estado = torch.load(ruta_modelo, map_location=dispositivo)
        logging.debug(f"Modelo cargado sin weights_only desde {ruta_modelo}.")
    except Exception as e:
        logging.critical(f"Error al cargar el modelo desde {ruta_modelo}: {e}")
        raise e
    modelo.load_state_dict(estado)
    logging.debug(f"State dict cargado en el modelo desde {ruta_modelo}.")

# Cargar modelos entrenados
cargar_estado_modelo(actor, f"{model_path}actor_model_final.pth", DEVICE)
cargar_estado_modelo(critic1, f"{model_path}critic1_model_final.pth", DEVICE)
cargar_estado_modelo(critic2, f"{model_path}critic2_model_final.pth", DEVICE)

actor.eval()
critic1.eval()
critic2.eval()
logging.info("Modelos cargados correctamente.")

# Cargar scaler y encoder
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

logging.info("Scaler y Encoder cargados correctamente.")

TABLA_SEÑALES_ESTRUCTURA = f"""
    CREATE TABLE IF NOT EXISTS {TABLA_SEÑALES} (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMPTZ NOT NULL,
        symbol VARCHAR(10) NOT NULL,
        type VARCHAR(10) NOT NULL,
        entry_price NUMERIC(10, 5) NOT NULL,
        sl_price NUMERIC(10, 5) NOT NULL,
        tp_price NUMERIC(10, 5) NOT NULL,
        result_pips NUMERIC(10, 2),
        confidence NUMERIC(5, 2),
        result VARCHAR(10),
        status VARCHAR(20) NOT NULL,
        candles_to_result INTEGER,
        atr_sl_size NUMERIC(10, 5),
        sl_atr_multiple NUMERIC(10, 2),
        tp_sl_ratio NUMERIC(10, 2),
        matriz_indicadores BYTEA
    )
"""

def deserializar_matriz(matriz_bytes):
    try:
        return pickle.loads(matriz_bytes)
    except Exception as e:
        logging.error(f"Error al deserializar 'matriz_indicadores': {e}")
        return None

def generar_señales(datos):
    try:
        columnas_numericas = [
            'open', 'high', 'low', 'close', 'volume',
            'distancia_1', 'distancia_2', 'magnitud', 'atr'
        ]

        # Ajustar según tu pipeline
        columnas_categoricas = ["session_market", "day_of_week", "month_of_year"]

        datos[columnas_numericas] = datos[columnas_numericas].apply(pd.to_numeric, errors='coerce').fillna(0)
        datos[columnas_numericas] = scaler.transform(datos[columnas_numericas])

        columnas_norm_final = [col + "_norm" for col in columnas_numericas]
        for orig, norm_col in zip(columnas_numericas, columnas_norm_final):
            datos[norm_col] = datos[orig]

        columnas_finales_numericas_norm = columnas_norm_final
        columnas_necesarias = columnas_finales_numericas_norm + ['matriz_indicadores']

        missing_columns = [col for col in columnas_necesarias if col not in datos.columns]
        if missing_columns:
            logging.error(f"Faltan columnas necesarias para generar señales: {missing_columns}")
            return pd.DataFrame()

        datos['matriz_indicadores_deserialized'] = datos['matriz_indicadores'].apply(deserializar_matriz)
        if datos['matriz_indicadores_deserialized'].isnull().any():
            logging.error("Algunas 'matriz_indicadores' no se pudieron deserializar correctamente.")
            datos = datos.dropna(subset=['matriz_indicadores_deserialized'])
            logging.info(f"Filas con 'matriz_indicadores' inválidas eliminadas. Total restante: {len(datos)}.")
            if len(datos) == 0:
                return pd.DataFrame()

        vectors = []
        for m in datos['matriz_indicadores_deserialized'].values:
            v = procesar_matriz_indicadores(m, estrategia_nan="reemplazar_por_0")
            if v is None:
                logging.error("No se pudo procesar la matriz de indicadores en un vector válido.")
                return pd.DataFrame()
            vectors.append(v)

        matriz = np.vstack(vectors)
        matriz_indicadores_dim = STATE_DIM - len(columnas_finales_numericas_norm)
        if matriz.shape[1] != matriz_indicadores_dim:
            logging.error(
                f"La dimensión de 'matriz_indicadores' no coincide con lo esperado. "
                f"Esperado: {matriz_indicadores_dim}, Obtenido: {matriz.shape[1]}"
            )
            return pd.DataFrame()

        estados = np.hstack((datos[columnas_finales_numericas_norm].values, matriz))
        if estados.shape[1] != STATE_DIM:
            logging.error(
                f"La dimensión total del estado no coincide con STATE_DIM. "
                f"Esperado: {STATE_DIM}, Obtenido: {estados.shape[1]}"
            )
            return pd.DataFrame()

        if not all(col in datos.columns for col in ['close', 'atr_sl_size', 'sl_atr_multiple', 'tp_sl_ratio']):
            logging.error("Faltan columnas 'close', 'atr_sl_size', 'sl_atr_multiple', o 'tp_sl_ratio'.")
            return pd.DataFrame()

        estados_tensor = torch.tensor(estados, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            medias, log_vars = actor(estados_tensor)
            acciones = medias

        acciones_np = acciones.cpu().numpy().flatten()

        señales = pd.DataFrame({
            'timestamp': datos['timestamp'],
            'symbol': datos['symbol'],
            'type': ['BUY' if x > 0 else 'SELL' for x in acciones_np],
            'entry_price': datos['close'],
            'sl_price': datos['close'] - (datos['atr_sl_size'] * datos['atr_norm']),
            'tp_price': datos['close'] + (datos['sl_atr_multiple'] * (datos['close'] - (datos['close'] - (datos['atr_sl_size'] * datos['atr_norm'])))),
            'result_pips': np.nan,
            'confidence': np.random.uniform(0.5, 1.0, size=len(datos)),
            'result': np.nan,
            'status': 'OPEN',
            'candles_to_result': np.nan,
            'atr_sl_size': datos['atr_sl_size'],
            'sl_atr_multiple': datos['sl_atr_multiple'],
            'tp_sl_ratio': datos['tp_sl_ratio'],
            'matriz_indicadores': datos['matriz_indicadores']
        })

        logging.info(f"{len(señales)} señales de trading generadas.")
        return señales

    except Exception as e:
        logging.error(f"Error al generar señales de trading: {e}")
        return pd.DataFrame()

def main():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        logging.info("Conexión a la base de datos establecida.")
    except Exception as e:
        logging.critical(f"Error al conectar a la base de datos: {e}")
        print("Error crítico: No se pudo conectar a la base de datos.")
        return

    try:
        with conn.cursor() as cursor:
            cursor.execute(TABLA_SEÑALES_ESTRUCTURA)
            conn.commit()
            logging.info(f"Estructura de tabla '{TABLA_SEÑALES}' asegurada.")
    except Exception as e:
        logging.critical(f"Error al crear la tabla '{TABLA_SEÑALES}': {e}")
        conn.rollback()
        print(f"Error crítico: No se pudo crear la tabla '{TABLA_SEÑALES}'.")
        conn.close()
        return

    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = %s
                );
            """, (TABLA_DATOS,))
            existe = cursor.fetchone()[0]
            if not existe:
                logging.critical(f"La tabla '{TABLA_DATOS}' no existe en la base de datos.")
                print(f"Error crítico: La tabla '{TABLA_DATOS}' no existe en la base de datos.")
                return
            else:
                logging.info(f"La tabla '{TABLA_DATOS}' existe. Procediendo a extraer datos.")

        datos = pd.read_sql(f"SELECT * FROM {TABLA_DATOS} ORDER BY timestamp ASC", conn)
        logging.info(f"Se han leído {len(datos)} registros de la tabla '{TABLA_DATOS}'.")

        señales = generar_señales(datos)
        if not señales.empty:
            with conn.cursor() as cursor:
                columnas = ', '.join(señales.columns)
                registros = [tuple(row) for row in señales.itertuples(index=False)]
                # Aquí nos aseguramos de usar TABLA_SEÑALES para guardar las señales
                query = f"INSERT INTO {TABLA_SEÑALES} ({columnas}) VALUES %s"
                execute_values(cursor, query, registros)
                conn.commit()
                logging.info(f"{len(señales)} señales de trading insertadas en la base de datos (tabla '{TABLA_SEÑALES}').")
        else:
            logging.info("No se generaron señales de trading para insertar.")

    except Exception as e:
        logging.critical(f"Error durante el procesamiento de datos: {e}")
        conn.rollback()
    finally:
        conn.close()
        logging.info("Conexión a la base de datos cerrada.")

if __name__ == "__main__":
    main()
