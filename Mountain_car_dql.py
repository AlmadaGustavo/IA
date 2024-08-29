import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F

# Define model
class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()

        # Define as camadas da rede
        self.fc1 = nn.Linear(in_states, h1_nodes)   # Primeira camada totalmente conectada
        self.out = nn.Linear(h1_nodes, out_actions) # Camada de saída

    def forward(self, x):
        # Aplicação da função de ativação ReLU e cálculo da saída
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return x

class ReplayMemory():
    def __init__(self, maxlen):
        # Inicializa uma memória de replay com tamanho máximo 'maxlen'
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition):
        # Adiciona uma transição (estado, ação, novo estado, recompensa, terminado) à memória
        self.memory.append(transition)

    def sample(self, sample_size):
        # Retorna uma amostra aleatória da memória
        return random.sample(self.memory, sample_size)

    def __len__(self):
        # Retorna o tamanho atual da memória
        return len(self.memory)


class MountainCarDQL():
    # Hiperparâmetros ajustáveis
    learning_rate_a = 0.01         # Taxa de aprendizado (alpha)
    discount_factor_g = 0.9        # Fator de desconto (gamma)    
    network_sync_rate = 50000      # Número de passos antes de sincronizar a rede de treino e a rede alvo
    replay_memory_size = 100000    # Tamanho da memória de replay
    mini_batch_size = 32           # Tamanho do lote de treinamento amostrado da memória de replay
    
    num_divisions = 20             # Divisões de espaço para discretização de estados (posição e velocidade)

    # Rede Neural
    loss_fn = nn.MSELoss()         # Função de perda da rede neural. MSE = Erro Quadrático Médio.
    optimizer = None               # Otimizador da rede neural, inicializado mais tarde.


    def train(self, episodes, render=False):
        # Cria uma instância do ambiente MountainCar-v0. Se 'render' for True, o ambiente será visualmente renderizado.
        env = gym.make('MountainCar-v0', render_mode='human' if render else None)
        
        # Obtém o número de estados (dimensão da observação) e o número de ações possíveis no ambiente.
        num_states = env.observation_space.shape[0] # Espera-se 2: posição e velocidade
        num_actions = env.action_space.n

        # Discretiza o espaço de posição e velocidade em segmentos. Isto transforma o espaço contínuo em um espaço discreto.
        self.pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], self.num_divisions)    # Entre -1.2 e 0.6
        self.vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], self.num_divisions)    # Entre -0.07 e 0.07

        # Inicializa epsilon para a política epsilon-greedy. Epsilon começa em 1 (ações totalmente aleatórias).
        epsilon = 1
        memory = ReplayMemory(self.replay_memory_size)  # Cria a memória de replay com o tamanho especificado.

        # Cria as redes neurais de política e de destino. Ambas redes têm a mesma estrutura.
        policy_dqn = DQN(in_states=num_states, h1_nodes=10, out_actions=num_actions)
        target_dqn = DQN(in_states=num_states, h1_nodes=10, out_actions=num_actions)

        # Sincroniza a rede de destino com a rede de política inicialmente.
        target_dqn.load_state_dict(policy_dqn.state_dict())

        # Define o otimizador Adam para a rede de política com a taxa de aprendizado especificada.
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        # Lista para armazenar as recompensas obtidas em cada episódio.
        rewards_per_episode = []

        # Lista para armazenar a evolução de epsilon ao longo do treinamento.
        epsilon_history = []

        # Contador de passos tomados e flags para verificar se o objetivo foi alcançado.
        step_count = 0
        goal_reached = False
        best_rewards = -200  # Inicializa a melhor recompensa obtida.

        # Loop principal de treinamento por episódio.
        for i in range(episodes):
            state = env.reset()[0]  # Reseta o ambiente e obtém o estado inicial.
            terminated = False      # Flag que indica se o episódio terminou.
            rewards = 0             # Acumulador de recompensas do episódio.

            # Loop para cada passo no episódio.
            while not terminated and rewards > -1000:
                # Seleciona a ação com base na política epsilon-greedy.
                if random.random() < epsilon:
                    # Ação aleatória
                    action = env.action_space.sample()
                else:
                    # Ação com base na política (máximo valor Q estimado pela rede de política).
                    with torch.no_grad():
                        action = policy_dqn(self.state_to_dqn_input(state)).argmax().item()

                # Executa a ação e obtém o novo estado e a recompensa.
                new_state, reward, terminated, truncated, _ = env.step(action)

                # Acumula a recompensa do episódio.
                rewards += reward

                # Armazena a transição de experiência na memória de replay.
                memory.append((state, action, new_state, reward, terminated)) 

                # Atualiza o estado atual para o próximo passo.
                state = new_state

                # Incrementa o contador de passos.
                step_count += 1

            # Adiciona as recompensas obtidas neste episódio à lista.
            rewards_per_episode.append(rewards)
            if terminated:
                goal_reached = True

            # A cada 1000 episódios, imprime o progresso e plota gráficos.
            if i != 0 and i % 1000 == 0:
                print(f'Episode {i} Epsilon {epsilon}')
                self.plot_progress(rewards_per_episode, epsilon_history)
            
            # Atualiza a melhor recompensa e salva a política se a recompensa atual for melhor.
            if rewards > best_rewards:
                best_rewards = rewards
                print(f'Best rewards so far: {best_rewards}')
                # Salva o estado atual da rede de política.
                torch.save(policy_dqn.state_dict(), f"mountaincar_dql_{i}.pt")

            # Se houver experiência suficiente na memória e o objetivo foi alcançado, otimiza a rede de treino.
            if len(memory) > self.mini_batch_size and goal_reached:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)        

                # Decai o valor de epsilon.
                epsilon = max(epsilon - 1/episodes, 0)
                epsilon_history.append(epsilon)

                # Sincroniza a rede de destino com a rede de política a cada 'network_sync_rate' passos.
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0                    

        # Fecha o ambiente após o treinamento.
        env.close()

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        current_q_list = []  # Lista para armazenar os valores Q atuais calculados pela rede de política.
        target_q_list = []   # Lista para armazenar os valores Q-alvo calculados pela rede de destino.

        # Itera sobre cada transição no mini-batch.
        for state, action, new_state, reward, terminated in mini_batch:
            # Se o estado é terminal (o agente alcançou o objetivo ou falhou), o valor Q-alvo é simplesmente a recompensa recebida.
            if terminated: 
                target = torch.FloatTensor([reward])
            else:
                # Caso contrário, calcula o valor Q-alvo usando a fórmula de Bellman.
                with torch.no_grad():
                    # Calcula o valor Q-alvo como a recompensa recebida mais o valor Q máximo do novo estado, descontado pelo fator gamma.
                    target = torch.FloatTensor(
                        reward + self.discount_factor_g * target_dqn(self.state_to_dqn_input(new_state)).max()
                    )

            # Obtém os valores Q atuais da rede de política para o estado atual.
            current_q = policy_dqn(self.state_to_dqn_input(state))
            current_q_list.append(current_q)

            # Obtém os valores Q da rede de destino para o estado atual.
            target_q = target_dqn(self.state_to_dqn_input(state)) 
            
            # Ajusta o valor Q para a ação tomada no estado atual para o valor Q-alvo calculado.
            target_q[action] = target
            target_q_list.append(target_q)
                
        # Calcula a perda (loss) entre os valores Q atuais e os valores Q-alvo para todo o mini-batch.
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        # Otimiza a rede de política com base na perda calculada.
        self.optimizer.zero_grad()   # Zera os gradientes acumulados do otimizador.
        loss.backward()             # Calcula os gradientes com base na perda.
        self.optimizer.step()       # Atualiza os pesos da rede de política com base nos gradientes calculados.


    def state_to_dqn_input(self, state) -> torch.Tensor:
        # Converte o estado (posição, velocidade) em índices discretos usando bins.
        state_p = np.digitize(state[0], self.pos_space)
        state_v = np.digitize(state[1], self.vel_space)
        
        # Retorna o estado convertido como um tensor do PyTorch.
        return torch.FloatTensor([state_p, state_v])
    
    def test(self, episodes, model_filepath):
        # Cria uma instância do ambiente MountainCar.
        env = gym.make('MountainCar-v0', render_mode='human')
        num_states = env.observation_space.shape[0]  # Número de estados (posição e velocidade).
        num_actions = env.action_space.n  # Número de ações possíveis (esquerda, parada, direita).

        # Recria os bins para a discretização dos estados, como foi feito no treinamento.
        self.pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], self.num_divisions)
        self.vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], self.num_divisions)

        # Carrega o modelo treinado.
        policy_dqn = DQN(in_states=num_states, h1_nodes=10, out_actions=num_actions) 
        policy_dqn.load_state_dict(torch.load(model_filepath))
        policy_dqn.eval()  # Coloca o modelo em modo de avaliação (desativa dropout, batch norm, etc.)

        # Executa o teste por um número especificado de episódios.
        for i in range(episodes):
            state = env.reset()[0]  # Inicializa o estado do ambiente.
            terminated = False  # Flag para verificar se o episódio foi terminado.
            truncated = False  # Flag para verificar se o episódio foi truncado (máximo de ações atingido).

            # O agente navega pelo ambiente até o episódio ser terminado ou truncado.
            while not terminated and not truncated:
                # Seleciona a melhor ação usando a política aprendida.
                with torch.no_grad():
                    action = policy_dqn(self.state_to_dqn_input(state)).argmax().item()

                # Executa a ação no ambiente e obtém o novo estado e recompensa.
                state, reward, terminated, truncated, _ = env.step(action)

        # Fecha o ambiente após o teste.
        env.close()

if __name__ == '__main__':

    mountaincar = MountainCarDQL()
   # mountaincar.train(20000, False)
    mountaincar.test(10, "mountaincar_dql_12853.pt")