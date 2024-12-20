import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging

# Configuración del archivo de log
log_filename = "sac_model_config.log"
logging.basicConfig(
    filename=log_filename,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.info("Inicio de la configuración del modelo SAC.")

# Configuración básica
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 3e-4
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.2  # Coeficiente de entropía

logging.info(f"Configuración: DEVICE={DEVICE}, LEARNING_RATE={LEARNING_RATE}, GAMMA={GAMMA}, TAU={TAU}, ALPHA={ALPHA}")

# Cargar vectores de estado
try:
    vectores_estado = np.load("vectores_estado.npy")
    logging.info(f"Vectores de estado cargados correctamente. Forma: {vectores_estado.shape}")
    print(f"Vectores de estado cargados: {vectores_estado.shape}")
except Exception as e:
    logging.critical(f"Error al cargar vectores de estado: {e}")
    raise e

# Ajustar la dimensión del estado para incluir la matriz de indicadores
STATE_DIM = vectores_estado.shape[1]  # El tamaño del vector de estado ahora incluye la matriz de indicadores
ACTION_DIM = 1  # Supongamos una acción continua (e.g., tamaño de la operación)
HIDDEN_DIM = 256  # Tamaño de las capas ocultas
logging.info(f"Dimensiones configuradas: STATE_DIM={STATE_DIM}, ACTION_DIM={ACTION_DIM}, HIDDEN_DIM={HIDDEN_DIM}")

# Definir la red Actor
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
        log_std = torch.clamp(self.log_std(x), -20, 2)  # Restringir log_std
        return mean, log_std

# Definir la red Crítico
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q_value = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q_value(x)

# Instanciar las redes
try:
    actor = Actor(STATE_DIM, ACTION_DIM, HIDDEN_DIM).to(DEVICE)
    critic1 = Critic(STATE_DIM, ACTION_DIM, HIDDEN_DIM).to(DEVICE)
    critic2 = Critic(STATE_DIM, ACTION_DIM, HIDDEN_DIM).to(DEVICE)
    logging.info("Redes instanciadas correctamente.")
except Exception as e:
    logging.critical(f"Error al instanciar las redes: {e}")
    raise e

# Optimizadores
try:
    actor_optimizer = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
    critic1_optimizer = optim.Adam(critic1.parameters(), lr=LEARNING_RATE)
    critic2_optimizer = optim.Adam(critic2.parameters(), lr=LEARNING_RATE)
    logging.info("Optimizadores configurados correctamente.")
except Exception as e:
    logging.critical(f"Error al configurar los optimizadores: {e}")
    raise e

print("Modelos configurados correctamente.")
logging.info("Configuración de los modelos completada con éxito.")
