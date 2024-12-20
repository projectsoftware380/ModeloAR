import psycopg2
import json
import numpy as np
import logging
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import pickle

# Configuración de logging
log_filename = "state_builder.log"
logging.basicConfig(
    filename=log_filename,
    filemode="w",
    level=logging.DEBUG,  # Cambiar a DEBUG para mayor detalle durante el desarrollo
    format="%(asctime)s - %(levelname)s - %(message)s",
)
print(f"Logging configurado. Archivo de logs: {log_filename}")

# Función para aplanar listas recursivamente
def flatten_list(nested_list):
    """Aplana una lista anidada recursivamente."""
    flat = []
    for item in nested_list:
        if isinstance(item, (list, tuple, np.ndarray)):
            flat.extend(flatten_list(item))
        else:
            flat.append(item)
    return flat

# Función para deserializar 'matriz_indicadores'
def deserializar_matriz(matriz_indicadores):
    try:
        if isinstance(matriz_indicadores, memoryview):
            bytes_data = bytes(matriz_indicadores)
            matriz_indicadores = pickle.loads(bytes_data)
            logging.debug("Matriz de indicadores deserializada correctamente desde memoryview.")
        elif isinstance(matriz_indicadores, bytes):
            matriz_indicadores = pickle.loads(matriz_indicadores)
            logging.debug("Matriz de indicadores deserializada correctamente desde bytes.")
        elif isinstance(matriz_indicadores, str):
            matriz_indicadores = pickle.loads(matriz_indicadores.encode('utf-8'))
            logging.debug("Matriz de indicadores deserializada correctamente desde string.")
        else:
            logging.error(f"Tipo de 'matriz_indicadores' no soportado: {type(matriz_indicadores)}")
            return None
    except Exception as e:
        logging.error(f"Error al deserializar 'matriz_indicadores': {e}")
        return None
    
    # Verificar que sea un array NumPy
    if not isinstance(matriz_indicadores, np.ndarray):
        logging.error(f"'matriz_indicadores' no es un numpy array después de deserializar: {type(matriz_indicadores)}")
        return None
    
    # Convertir cualquier dimensionalidad a 1D
    if matriz_indicadores.ndim > 2:
        logging.info(f"La matriz tiene {matriz_indicadores.ndim} dimensiones. Se flatten a 1D.")
        matriz_indicadores = matriz_indicadores.flatten()
    elif matriz_indicadores.ndim == 1:
        logging.info("La matriz es 1D y se convertirá a 2D con una sola fila.")
        matriz_indicadores = matriz_indicadores.reshape(1, -1)
    
    # Asegurar que todos los datos sean numéricos
    try:
        matriz_indicadores = matriz_indicadores.astype(float)
    except ValueError as ve:
        logging.error(f"Error al convertir 'matriz_indicadores' a float: {ve}")
        return None
    
    return matriz_indicadores

# Función para cargar la configuración
def cargar_configuracion(config_path="config4h.json"):
    try:
        with open(config_path, "r") as file:
            return json.load(file)
    except Exception as e:
        logging.error(f"Error al cargar configuración: {e}")
        print(f"Error: No se pudo cargar la configuración desde '{config_path}'.")
        return None

# Función para conectar a la base de datos
def conectar_bd(db_config):
    try:
        conn = psycopg2.connect(
            database=db_config["name"],
            user=db_config["user"],
            password=db_config["password"],
            host=db_config["host"],
            port=db_config["port"],
        )
        logging.info("Conexión a la base de datos exitosa.")
        return conn
    except Exception as e:
        logging.error(f"Error al conectar a la base de datos: {e}")
        print("Error: No se pudo conectar a la base de datos.")
        return None

# Función para obtener datos desde la tabla
def obtener_datos_tabla(conn, table_name):
    try:
        with conn.cursor() as cursor:
            query = f"""
                SELECT timestamp, open, high, low, close, volume, distancia_1, distancia_2, matriz_indicadores 
                FROM {table_name}
                ORDER BY timestamp ASC
            """
            cursor.execute(query)
            rows = cursor.fetchall()
            logging.info(f"Se obtuvieron {len(rows)} registros.")
            return rows
    except Exception as e:
        logging.error(f"Error al consultar la tabla: {e}")
        return []

# Función para calcular RSI
def calcular_rsi(df, periodos=14):
    try:
        delta = df['close'].diff()
        ganancia = delta.where(delta > 0, 0)
        perdida = -delta.where(delta < 0, 0)
        ganancia_media = ganancia.rolling(window=periodos).mean()
        perdida_media = perdida.rolling(window=periodos).mean()
        rs = ganancia_media / perdida_media
        rsi = 100 - (100 / (1 + rs))
        df["rsi"] = rsi
        
        # Loggear después de calcular RSI
        logging.info("Después de calcular RSI:")
        logging.info(f"dtypes:\n{df.dtypes}")
        logging.info(f"head:\n{df[['rsi']].head()}")
        
        return df
    except Exception as e:
        logging.error(f"Error al calcular RSI: {e}")
        df["rsi"] = np.nan
        return df

# Función para calcular ATR
def calcular_atr(df, periodos=14):
    try:
        df["tr1"] = df["high"] - df["low"]
        df["tr2"] = (df["high"] - df["close"].shift(1)).abs()
        df["tr3"] = (df["low"] - df["close"].shift(1)).abs()
        df["tr"] = df[["tr1", "tr2", "tr3"]].max(axis=1)
        df["atr"] = df["tr"].rolling(window=periodos).mean()
        
        # Loggear después de calcular ATR
        logging.info("Después de calcular ATR:")
        logging.info(f"dtypes:\n{df.dtypes}")
        logging.info(f"head:\n{df[['atr']].head()}")
        
        return df
    except Exception as e:
        logging.error(f"Error al calcular ATR: {e}")
        df["atr"] = np.nan
        return df

# Función para convertir columnas a numéricas con logs detallados
def convertir_columnas_a_numericas(df, columnas):
    """
    Convierte las columnas especificadas a tipos numéricos.
    Si la conversión falla, los valores se reemplazan por NaN.
    Además, registra cuántos valores fueron convertidos a NaN.
    """
    for columna in columnas:
        if columna in df.columns:
            antes_na = df[columna].isna().sum()
            df[columna] = pd.to_numeric(df[columna], errors='coerce')
            despues_na = df[columna].isna().sum()
            cantidad_convertida = despues_na - antes_na
            logging.debug(f"Columna '{columna}' convertida a numérico. Valores convertidos a NaN: {cantidad_convertida}.")
    return df

# Función para normalizar datos con logs detallados
def normalizar_datos(df, columnas, feature_range=(0, 1)):
    scaler = MinMaxScaler(feature_range=feature_range)
    for columna in columnas:
        if columna in df.columns:
            datos = df[columna].values.reshape(-1, 1)
            # Logear el dtype y tipos únicos de los datos
            tipos_unicos = set(map(type, datos.flatten()))
            logging.debug(f"Normalizando columna '{columna}' con dtype: {datos.dtype}, tipos únicos: {tipos_unicos}")
            # Reemplazar NaN temporalmente para evitar errores en la normalización
            nan_mask = np.isnan(datos)
            datos[nan_mask] = 0
            try:
                datos_normalizados = scaler.fit_transform(datos).flatten()
            except Exception as e:
                logging.error(f"Error al normalizar la columna '{columna}': {e}")
                datos_normalizados = np.full_like(datos.flatten(), np.nan)
            # Restaurar NaN
            datos_normalizados[nan_mask.flatten()] = np.nan
            df[f"{columna}_norm"] = datos_normalizados
            logging.debug(f"Columna '{columna}' normalizada. Tipo: {df[f'{columna}_norm'].dtype}")
    
    # Loggear después de normalizar
    logging.info("Después de normalizar los datos:")
    logging.info(f"dtypes:\n{df.dtypes}")
    logging.info(f"head:\n{df[[f'{columna}_norm' for columna in columnas]].head()}")
    
    return df

# Función para codificar categorías temporales
def codificar_categorias(df):
    session_market = []
    day_of_week = []
    month_of_year = []
    
    for timestamp in df['timestamp']:
        hour = timestamp.hour
        if 0 <= hour < 8:
            session_market.append([1, 0, 0, 0])  # Asia
        elif 8 <= hour < 17:
            session_market.append([0, 1, 0, 0])  # Londres
        elif 17 <= hour < 22:
            session_market.append([0, 0, 1, 0])  # Nueva York
        else:
            session_market.append([0, 0, 0, 1])  # Pacífico
        
        dow = [0] * 7
        dow[timestamp.weekday()] = 1
        day_of_week.append(dow)
        
        moy = [0] * 12
        moy[timestamp.month - 1] = 1
        month_of_year.append(moy)
    
    df['session_market'] = session_market
    df['day_of_week'] = day_of_week
    df['month_of_year'] = month_of_year
    logging.debug("Categorías temporales codificadas.")
    
    # Loggear después de codificar categorías
    logging.info("Después de codificar categorías temporales:")
    logging.info(f"dtypes:\n{df.dtypes}")
    logging.info(f"head:\n{df[['session_market', 'day_of_week', 'month_of_year']].head()}")
    
    return df

# Función para deserializar todas las matrices de indicadores
def deserializar_matrices(df):
    matrices = []
    for idx, matriz in enumerate(df['matriz_indicadores']):
        deserialized = deserializar_matriz(matriz)
        if deserialized is not None:
            # Aplanar completamente la lista
            flat_list = flatten_list(deserialized.flatten().tolist())
            # Verificar que todos los elementos sean numéricos
            if all(isinstance(x, (int, float)) for x in flat_list):
                matrices.append(flat_list)
            else:
                matrices.append(np.nan)
                logging.warning(f"Fila {idx + 1}: 'matriz_indicadores_proc' contiene tipos no numéricos después del aplanado.")
        else:
            matrices.append(np.nan)  # Marcar como NaN si falla la deserialización
            logging.warning(f"Fila {idx + 1}: Deserialización de 'matriz_indicadores' fallida.")
    df['matriz_indicadores_proc'] = matrices
    
    # Loggear dtypes y una muestra de los datos después de deserializar
    logging.info("Después de deserializar matrices:")
    logging.info(f"dtypes:\n{df.dtypes}")
    logging.info(f"head:\n{df.head()}")
    
    return df

# Función para construir el vector de estado
def construir_vector_estado(df):
    vectores_estado = []
    for idx, row in df.iterrows():
        # Verificar si 'matriz_indicadores_proc' es una lista de flotantes
        if isinstance(row['matriz_indicadores_proc'], float) and np.isnan(row['matriz_indicadores_proc']):
            logging.warning(f"Fila {idx + 1} omitida por 'matriz_indicadores' inválida.")
            continue
        if not isinstance(row['matriz_indicadores_proc'], list):
            logging.warning(f"Fila {idx + 1} omitida porque 'matriz_indicadores_proc' no es una lista.")
            continue
        if not all(isinstance(x, (int, float)) for x in row['matriz_indicadores_proc']):
            logging.warning(f"Fila {idx + 1} omitida porque 'matriz_indicadores_proc' contiene tipos no numéricos.")
            continue
        
        # Verificar y convertir cada parte a float
        try:
            open_norm = float(row['open_norm'])
            high_norm = float(row['high_norm'])
            low_norm = float(row['low_norm'])
            close_norm = float(row['close_norm'])
            volume_norm = float(row['volume_norm'])
            distancia_1_norm = float(row['distancia_1_norm'])
            distancia_2_norm = float(row['distancia_2_norm'])
            magnitud_norm = float(row['magnitud_norm'])
            
            # session_market, day_of_week, month_of_year son listas de enteros
            session_market = [float(x) for x in row['session_market']]
            day_of_week = [float(x) for x in row['day_of_week']]
            month_of_year = [float(x) for x in row['month_of_year']]
            
            rsi_norm = float(row['rsi_norm'])
            atr_norm = float(row['atr_norm'])
            
            matriz_indicadores_proc = [float(x) for x in row['matriz_indicadores_proc']]
        except ValueError as ve:
            logging.error(f"Fila {idx + 1}: Error al convertir a float: {ve}")
            continue
        except TypeError as te:
            logging.error(f"Fila {idx + 1}: Error de tipo al convertir a float: {te}")
            continue
        
        # Concatenar todas las partes del vector de estado
        try:
            partes = [
                open_norm, high_norm, low_norm, close_norm,
                volume_norm, distancia_1_norm, distancia_2_norm, magnitud_norm,
                *session_market, *day_of_week, *month_of_year,
                rsi_norm, atr_norm,
                *matriz_indicadores_proc
            ]
            vector = np.array(partes, dtype=float)  # Forzar dtype=float
            
            # Registrar el tipo de datos y una muestra del vector
            logging.debug(f"Fila {idx + 1}: Tipo del vector_estado: {vector.dtype}")
            logging.debug(f"Fila {idx + 1}: Forma del vector_estado: {vector.shape}")
            logging.debug(f"Fila {idx + 1}: Contenido del vector_estado (primeros 10 elementos): {vector[:10]}...")
            
            # Verificar si hay NaN en el vector_estado
            if np.isnan(vector).any():
                logging.warning(f"Fila {idx + 1} omitida debido a NaN en el vector de estado.")
                continue
            vectores_estado.append(vector)
        except Exception as e:
            logging.critical(f"Fila {idx + 1}: Error al construir el vector de estado: {e}", exc_info=True)
            # Registrar el contenido completo de 'partes' para diagnóstico
            logging.critical(f"Fila {idx + 1}: Contenido de 'partes': {partes}")
            continue
    return np.array(vectores_estado)

# Función principal para procesar todos los datos
def procesar_datos_completo(rows):
    # Convertir a DataFrame
    columnas = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'distancia_1', 'distancia_2', 'matriz_indicadores']
    df = pd.DataFrame(rows, columns=columnas)
    
    # Deserializar 'matriz_indicadores'
    df = deserializar_matrices(df)
    
    # Eliminar filas donde la deserialización falló
    df = df.dropna(subset=['matriz_indicadores_proc'])
    
    # Convertir columnas a numéricas antes de calcular 'magnitud'
    columnas_a_convertir = ['open', 'high', 'low', 'close']
    df = convertir_columnas_a_numericas(df, columnas_a_convertir)
    
    # Logear dtypes después de la conversión
    logging.info(f"dtypes después de convertir columnas a numéricas:\n{df.dtypes}")
    
    # Eliminar filas con NaN en las columnas convertidas
    filas_invalidas = df[df[columnas_a_convertir].isna().any(axis=1)]
    if not filas_invalidas.empty:
        logging.warning(f"Total de filas con valores no numéricos en columnas convertidas: {len(filas_invalidas)}")
        df = df.dropna(subset=columnas_a_convertir)
        logging.info(f"Filas eliminadas debido a valores no numéricos en columnas convertidas: {len(filas_invalidas)}")
    
    # Calcular magnitud
    df['magnitud'] = (df['distancia_1'] - df['distancia_2']).abs()
    
    # Calcular RSI y ATR sobre todo el DataFrame
    df = calcular_rsi(df)
    df = calcular_atr(df)
    
    # Normalizar los campos
    campos_normalizar = ['open', 'high', 'low', 'close', 'volume', 'distancia_1', 'distancia_2', 'magnitud', 'rsi', 'atr']
    df = normalizar_datos(df, campos_normalizar)
    
    # Codificar categorías temporales
    df = codificar_categorias(df)
    
    # Descartar filas donde RSI o ATR no pudieron ser calculados
    df = df.dropna(subset=['rsi_norm', 'atr_norm'])
    
    # Construir el vector de estado
    vectores_estado = construir_vector_estado(df)
    
    logging.info(f"Total de vectores generados: {len(vectores_estado)}")
    return vectores_estado

# Función para guardar los vectores de estado
def guardar_vectores(vectores_estado, output_path="vectores_estado.npy"):
    try:
        # Verificar el dtype del array
        if vectores_estado.dtype.kind not in {'f', 'i'}:
            logging.error(f"vectores_estado tiene dtype no numérico: {vectores_estado.dtype}")
            print(f"Error: vectores_estado tiene dtype no numérico: {vectores_estado.dtype}")
            return
        np.save(output_path, vectores_estado)
        logging.info(f"Vectores de estado guardados en '{output_path}'.")
    except Exception as e:
        logging.error(f"Error al guardar los vectores de estado: {e}")

# Función de prueba para crear un vector simple
def test_vector_creation():
    partes = [
        0.5, 0.6, 0.7, 0.8,
        0.9, 1.0, 1.1, 1.2,
        1.3, 1.4, 1.5, 1.6,
        1.7, 1.8, 1.9, 2.0,
        2.1, 2.2, 2.3, 2.4,
        2.5, 2.6, 2.7, 2.8,
        2.9, 3.0, 3.1, 3.2,
        3.3, 3.4, 3.5, 3.6,
        3.7, 3.8, 3.9, 4.0
    ]
    try:
        vector = np.array(partes, dtype=float)
        logging.debug(f"Vector de prueba: dtype={vector.dtype}, shape={vector.shape}, contenido={vector[:10]}...")
        if np.isnan(vector).any():
            logging.warning("Vector de prueba contiene NaN.")
        else:
            logging.info("Vector de prueba no contiene NaN.")
    except Exception as e:
        logging.error(f"Error en la creación del vector de prueba: {e}")

# Script principal
if __name__ == "__main__":
    print("Inicio de ejecución del script.")
    try:
        config = cargar_configuracion()
        if not config:
            logging.error("Error en la configuración. Verifica el archivo 'config4h.json'.")
            exit()
        
        db_config = config["database"]
        table_name = "ar_input_data"
        
        conn = conectar_bd(db_config)
        if not conn:
            logging.error("No se pudo conectar. Terminando ejecución.")
            exit()
        
        datos = obtener_datos_tabla(conn, table_name)
        conn.close()
        logging.info("Conexión a la base de datos cerrada.")
        
        if datos:
            vectores_estado = procesar_datos_completo(datos)
            if vectores_estado.size > 0:
                guardar_vectores(vectores_estado)
                print("Script ejecutado correctamente. Consulta 'state_builder.log' para más detalles.")
            else:
                logging.warning("No se generaron vectores de estado debido a datos inválidos.")
                print("No se generaron vectores de estado debido a datos inválidos.")
        else:
            logging.warning("No hay datos para procesar.")
            print("No hay datos disponibles para procesar.")
        
        # Ejecutar la prueba de creación de vector
        test_vector_creation()
    except Exception as e:
        logging.exception("Error inesperado:")
        print(f"Error crítico: {e}")
