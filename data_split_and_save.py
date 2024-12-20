import psycopg2
import pandas as pd
import logging
from psycopg2.extras import execute_values

# ------------------------------------------------------------
# Configuración de logging
# ------------------------------------------------------------
LOG_FILENAME = "data_split.log"
logging.basicConfig(
    filename=LOG_FILENAME,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.info("Inicio del script de división de datos.")

# ------------------------------------------------------------
# Parámetros de conexión a la base de datos
# ------------------------------------------------------------
DB_CONFIG = {
    "dbname": "nombre_base_datos",
    "user": "usuario",
    "password": "contraseña",
    "host": "localhost",
    "port": "5432",
}

TABLA_ORIGINAL = "ar_input_data"
TABLA_ENTRENAMIENTO = "training_data"
TABLA_PRUEBA = "test_data"

FECHA_CORTE = "2023-01-01"

# ------------------------------------------------------------
# Funciones auxiliares
# ------------------------------------------------------------
def conectar_db(config):
    """Establece la conexión a la base de datos."""
    try:
        conn = psycopg2.connect(**config)
        logging.info("Conexión a la base de datos exitosa.")
        return conn
    except Exception as e:
        logging.error(f"Error al conectar a la base de datos: {e}")
        raise

def cargar_datos(conn, tabla):
    """Carga los datos de la tabla original."""
    query = f"SELECT * FROM {tabla} ORDER BY timestamp ASC"
    datos = pd.read_sql(query, conn)
    logging.info(f"Datos cargados desde la tabla '{tabla}'. Total registros: {len(datos)}")
    return datos

def dividir_datos(datos, fecha_corte):
    """Divide los datos en dos conjuntos según la fecha de corte."""
    entrenamiento = datos[datos["timestamp"] < fecha_corte].copy()
    prueba = datos[datos["timestamp"] >= fecha_corte].copy()
    logging.info(f"División completada: Entrenamiento={len(entrenamiento)}, Prueba={len(prueba)}")
    return entrenamiento, prueba

def crear_tabla_si_no_existe(conn, tabla):
    """Crea una tabla con la misma estructura que la original si no existe."""
    with conn.cursor() as cursor:
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {tabla} (
                LIKE {TABLA_ORIGINAL} INCLUDING ALL
            );
        """)
        conn.commit()
        logging.info(f"Tabla '{tabla}' creada o verificada.")

def guardar_datos(conn, datos, tabla):
    """Guarda los datos en una tabla de PostgreSQL."""
    if datos.empty:
        logging.warning(f"No hay datos para guardar en '{tabla}'.")
        return

    columnas = ", ".join(datos.columns)
    valores = [tuple(row) for row in datos.itertuples(index=False)]
    query = f"INSERT INTO {tabla} ({columnas}) VALUES %s"

    with conn.cursor() as cursor:
        execute_values(cursor, query, valores)
        conn.commit()
        logging.info(f"{len(datos)} registros insertados en '{tabla}'.")

# ------------------------------------------------------------
# Script principal
# ------------------------------------------------------------
def main():
    conn = conectar_db(DB_CONFIG)
    try:
        # Crear tablas de destino si no existen
        crear_tabla_si_no_existe(conn, TABLA_ENTRENAMIENTO)
        crear_tabla_si_no_existe(conn, TABLA_PRUEBA)

        # Cargar y dividir los datos
        datos = cargar_datos(conn, TABLA_ORIGINAL)
        entrenamiento, prueba = dividir_datos(datos, FECHA_CORTE)

        # Guardar los datos en las tablas correspondientes
        guardar_datos(conn, entrenamiento, TABLA_ENTRENAMIENTO)
        guardar_datos(conn, prueba, TABLA_PRUEBA)
    except Exception as e:
        logging.error(f"Error durante la ejecución: {e}")
        raise
    finally:
        conn.close()
        logging.info("Conexión a la base de datos cerrada.")

if __name__ == "__main__":
    main()
