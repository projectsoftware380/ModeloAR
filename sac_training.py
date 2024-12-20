import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging

# Configuración de logging
log_filename = "sac_training.log"
logging.basicConfig(
    filename=log_filename,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.info("Inicio del entrenamiento SAC.")

# Configuración básica
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-5  # Reducir la tasa de aprendizaje aún más
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.1  # Reducimos la entropía para mejorar la estabilidad
BATCH_SIZE = 64
NUM_EPOCHS = 1000
BUFFER_SIZE = 100000

# Definir el par de divisas (puedes modificar esto según el par con el que estés trabajando)
currency_pair = "EURUSD"  # Ejemplo: EUR/USD

# Cargar vectores de estado
vectores_estado = np.load("vectores_estado.npy")
logging.info(f"Vectores de estado cargados. Forma: {vectores_estado.shape}")

# Dimensiones
STATE_DIM = vectores_estado.shape[1]  # Incluye RSI, ATR y la matriz de indicadores
ACTION_DIM = 1  # Supongamos una acción continua (e.g., tamaño de la operación)
HIDDEN_DIM = 32  # Reducimos aún más el tamaño de las capas ocultas

logging.info(f"Dimensiones configuradas: STATE_DIM={STATE_DIM}, ACTION_DIM={ACTION_DIM}, HIDDEN_DIM={HIDDEN_DIM}")

# Inicialización de pesos (He)
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')  # Inicialización He
        nn.init.constant_(m.bias, 0)

# Redes neuronales para SAC
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.apply(weights_init)  # Inicialización de pesos

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)  # Restringir log_std
        return mean, log_std

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q_value = nn.Linear(hidden_dim, 1)
        self.apply(weights_init)  # Inicialización de pesos

    def forward(self, state, action):
        action = action.view(action.size(0), -1)  # Asegurarse de que `action` tenga 2 dimensiones
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q_value(x)

# Inicializar redes y optimizadores
actor = Actor(STATE_DIM, ACTION_DIM, HIDDEN_DIM).to(DEVICE)
critic1 = Critic(STATE_DIM, ACTION_DIM, HIDDEN_DIM).to(DEVICE)
critic2 = Critic(STATE_DIM, ACTION_DIM, HIDDEN_DIM).to(DEVICE)
critic1_target = Critic(STATE_DIM, ACTION_DIM, HIDDEN_DIM).to(DEVICE)
critic2_target = Critic(STATE_DIM, ACTION_DIM, HIDDEN_DIM).to(DEVICE)

critic1_target.load_state_dict(critic1.state_dict())
critic2_target.load_state_dict(critic2.state_dict())

actor_optimizer = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
critic1_optimizer = optim.Adam(critic1.parameters(), lr=LEARNING_RATE)
critic2_optimizer = optim.Adam(critic2.parameters(), lr=LEARNING_RATE)

logging.info("Modelos y optimizadores inicializados.")

# Buffer de experiencia simulada (ejemplo para entrenamiento)
class ReplayBuffer:
    def __init__(self, size):
        self.buffer = []
        self.max_size = size

    def add(self, transition):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[idx] for idx in indices]

buffer = ReplayBuffer(BUFFER_SIZE)
logging.info("Replay buffer inicializado.")

# Simular llenado del buffer
for i in range(10000):  # Agregar datos de ejemplo
    state = torch.FloatTensor(vectores_estado[np.random.randint(0, len(vectores_estado))]).to(DEVICE)
    action = torch.FloatTensor([[np.random.uniform(-1, 1)]]).to(DEVICE)
    reward = np.random.uniform(-1, 1)  # Recompensa simulada
    next_state = state + torch.FloatTensor(np.random.normal(0, 0.1, state.shape)).to(DEVICE)
    done = np.random.choice([0, 1], p=[0.95, 0.05])
    buffer.add((state, action, reward, next_state, done))
logging.info("Replay buffer llenado con datos simulados.")

# Función de recompensa
def calcular_recompensa(signal, result):
    """
    Ajusta las recompensas basadas en el resultado de las señales de compra/venta.
    - Si el resultado es 'win', se da una recompensa positiva.
    - Si el resultado es 'loss', se da una recompensa negativa.
    """
    if result == "win":
        return 1  # Recompensa positiva por ganar la operación
    elif result == "loss":
        return -1  # Penalización por perder la operación
    return 0  # Sin recompensa si está "open" (todavía no se ha cerrado la operación)

# Función para guardar el modelo **solo al final del entrenamiento**
def save_model():
    model_save_path = f"D:/TradingIA/venv/ModeloAR/{currency_pair}/"  # Ruta donde se guardarán los modelos, incluye el par de divisas
    # Crear el directorio si no existe
    import os
    os.makedirs(model_save_path, exist_ok=True)
    
    # Guardar los modelos
    torch.save(actor.state_dict(), f"{model_save_path}actor_model_final.pth")
    torch.save(critic1.state_dict(), f"{model_save_path}critic1_model_final.pth")
    torch.save(critic2.state_dict(), f"{model_save_path}critic2_model_final.pth")
    logging.info(f"Modelos guardados en {model_save_path}.")

# Función de entrenamiento con clipping de gradientes
def train(batch_size):
    transitions = buffer.sample(batch_size)
    states, actions, rewards, next_states, dones = zip(*transitions)

    states = torch.stack(states)
    actions = torch.stack(actions)
    rewards = torch.FloatTensor(rewards).to(DEVICE).unsqueeze(1)
    next_states = torch.stack(next_states)
    dones = torch.FloatTensor(dones).to(DEVICE).unsqueeze(1)

    # Actualizar críticos
    with torch.no_grad():
        next_actions, next_log_pis = actor(next_states)
        target_q1 = critic1_target(next_states, next_actions)
        target_q2 = critic2_target(next_states, next_actions)
        target_q = rewards + GAMMA * (1 - dones) * (torch.min(target_q1, target_q2) - ALPHA * next_log_pis)

    q1 = critic1(states, actions)
    q2 = critic2(states, actions)
    critic1_loss = F.mse_loss(q1, target_q)
    critic2_loss = F.mse_loss(q2, target_q)

    critic1_optimizer.zero_grad()
    critic1_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic1.parameters(), 0.5)  # Clipping de gradientes
    critic1_optimizer.step()

    critic2_optimizer.zero_grad()
    critic2_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic2.parameters(), 0.5)  # Clipping de gradientes
    critic2_optimizer.step()

    # Actualizar actor
    actions, log_pis = actor(states)
    q1_pi = critic1(states, actions)
    actor_loss = (ALPHA * log_pis - q1_pi).mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)  # Clipping de gradientes
    actor_optimizer.step()

    # Actualizar críticos objetivo
    for target_param, param in zip(critic1_target.parameters(), critic1.parameters()):
        target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

    for target_param, param in zip(critic2_target.parameters(), critic2.parameters()):
        target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

    logging.info(f"Entrenamiento completado: Actor Loss={actor_loss.item()}, Critic Loss={critic1_loss.item()}")

# Entrenamiento principal
for epoch in range(NUM_EPOCHS):
    train(BATCH_SIZE)
    logging.info(f"Época {epoch + 1}/{NUM_EPOCHS} completada.")

# Guardar el modelo al finalizar todo el entrenamiento
save_model()

logging.info("Entrenamiento completado y modelos guardados.")
