import numpy as np
import logging

# Configuración de logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

def manejar_nan(matriz_indicadores, estrategia="reemplazar_por_0"):
    """
    Maneja los valores NaN en la matriz de indicadores según la estrategia seleccionada.
    Estrategias disponibles:
    - "reemplazar_por_0": Reemplaza NaN con 0.
    - "reemplazar_por_media": Reemplaza NaN con la media de la columna.
    
    Ejemplo:
        >>> manejar_nan(np.array([[1, np.nan], [3, 4]]), "reemplazar_por_media")
        array([[1. , 3.5],
               [3. , 4. ]])
    """
    try:
        logging.debug(f"Matriz antes de manejar NaN: \n{matriz_indicadores}")

        if estrategia == "reemplazar_por_0":
            # Reemplazar NaN por 0
            matriz_indicadores = np.nan_to_num(matriz_indicadores, nan=0)
            logging.info("Se reemplazaron los NaN con 0.")
        elif estrategia == "reemplazar_por_media":
            # Reemplazar NaN por la media de la columna
            # Calcular la media de cada columna ignorando NaN
            media_columnas = np.nanmean(matriz_indicadores, axis=0)
            # Encontrar los índices donde hay NaN
            inds = np.where(np.isnan(matriz_indicadores))
            if media_columnas.size == 0:
                logging.error("No se pudo calcular la media de las columnas debido a que todas las entradas son NaN.")
                return matriz_indicadores
            # Reemplazar NaN por la media de la columna correspondiente
            matriz_indicadores[inds] = np.take(media_columnas, inds[1])
            logging.info("Se reemplazaron los NaN con la media de la columna.")
        else:
            logging.error(f"Estrategia '{estrategia}' no reconocida.")
            raise ValueError(f"Estrategia '{estrategia}' no reconocida.")

        logging.debug(f"Matriz después de manejar NaN: \n{matriz_indicadores}")

        return matriz_indicadores
    except Exception as e:
        logging.error(f"Error al manejar NaN: {e}")
        raise

def procesar_matriz_indicadores(matriz_indicadores, estrategia_nan="reemplazar_por_0"):
    """
    Procesa la matriz de indicadores y la convierte en un vector 1D.
    Admite tanto matrices 1D como 2D.
    
    Ejemplo:
        >>> procesar_matriz_indicadores(np.array([[1, 2], [3, 4]]))
        array([1, 2, 3, 4])
    """
    try:
        # Convertir la entrada a un array de NumPy si no lo es, y asegurarse de que es float
        matriz_indicadores = np.asarray(matriz_indicadores, dtype=np.float64)

        # Verificar si la matriz está vacía
        if matriz_indicadores.size == 0:
            logging.warning("La matriz de indicadores está vacía. No se puede procesar.")
            return None

        # Manejar matrices 1D convirtiéndolas a 2D
        if matriz_indicadores.ndim == 1:
            logging.info("La matriz es 1D y se convertirá a 2D con una sola fila.")
            matriz_indicadores = matriz_indicadores.reshape(1, -1)
        elif matriz_indicadores.ndim != 2:
            logging.error(f"La matriz de indicadores no es 1D ni 2D. Dimensiones actuales: {matriz_indicadores.ndim}")
            return None

        logging.debug(f"Matriz después de la verificación de dimensiones: \n{matriz_indicadores}")

        # Manejar NaN según la estrategia
        matriz_indicadores = manejar_nan(matriz_indicadores, estrategia=estrategia_nan)

        # Verificar nuevamente si la matriz está vacía después de manejar NaN
        if matriz_indicadores.size == 0:
            logging.warning("Después de manejar NaN, la matriz de indicadores está vacía. No se puede procesar.")
            return None

        # Convertir la matriz en un vector 1D
        vector = matriz_indicadores.flatten()  # Aplanar la matriz a un vector
        logging.debug(f"Vector procesado: \n{vector}")

        return vector  # Devolver el vector para su uso posterior

    except Exception as e:
        logging.error(f"Error al procesar la matriz de indicadores: {e}")
        return None

# Función auxiliar para procesar múltiples matrices y etiquetarlas en los logs
def procesar_y_logear(matriz, descripcion, estrategia_nan="reemplazar_por_0"):
    logging.info(f"Procesando matriz: {descripcion}")
    try:
        vector = procesar_matriz_indicadores(matriz, estrategia_nan)
        if vector is not None:
            logging.info(f"Vector procesado correctamente: \n{vector}\n")
        else:
            logging.info("No se pudo procesar el vector.\n")
    except Exception as e:
        logging.error(f"Error al procesar y logear la matriz '{descripcion}': {e}")

# Ejemplo de cómo usar la función de validación y conversión
if __name__ == "__main__":
    # Ejemplo de una matriz de indicadores válida (17x7)
    matriz_validada = np.random.rand(17, 7)  # Una matriz 17x7 de números aleatorios (17 periodos, 7 indicadores)
    procesar_y_logear(matriz_validada, "Matriz Valida (17x7)")

    # Ejemplo de una matriz de indicadores con NaN
    matriz_con_nan = np.array([[1, 2, 3, np.nan], [5, 6, 7, 8]])
    procesar_y_logear(matriz_con_nan, "Matriz con NaN (2x4)", estrategia_nan="reemplazar_por_0")
    
    # Ejemplo de una matriz de indicadores con NaN y estrategia de reemplazo por media
    matriz_con_nan_media = np.array([[1, 2, np.nan, 4], [5, np.nan, 7, 8]])
    procesar_y_logear(matriz_con_nan_media, "Matriz con NaN y Reemplazo por Media (2x4)", estrategia_nan="reemplazar_por_media")

    # Ejemplo de una matriz de indicadores vacía
    matriz_vacia = np.array([]).reshape(0, 4)  # Matriz vacía con 0 filas y 4 columnas
    procesar_y_logear(matriz_vacia, "Matriz Vacia (0x4)")

    # Ejemplo de una matriz de indicadores no 2D (1D)
    matriz_no_2d = np.array([1, 2, 3, 4])
    procesar_y_logear(matriz_no_2d, "Matriz No 2D (1D)")
