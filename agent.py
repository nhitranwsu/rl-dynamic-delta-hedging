import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# 1. The Neural Network Architecture
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        # A lightweight feed-forward network to process the 5 market variables
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Outputs the expected future reward for each possible action

# 2. The Memory Buffer (Experience Replay)
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action_idx, reward, next_state, done):
        self.buffer.append((state, action_idx, reward, next_state, done))

    def sample(self, batch_size):
        # Randomly sample past experiences to break correlation in sequential trading data
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)

# 3. The Reinforcement Learning Agent
class DQNAgent:
    def __init__(self, state_dim=5, action_choices=[-100, -50, -10, 0, 10, 50, 100]):
        self.state_dim = state_dim
        
        # The exact quantities of stock the AI is allowed to buy/sell at any given tick
        self.action_choices = action_choices
        self.action_dim = len(action_choices)
        
        # RL Hyperparameters
        self.gamma = 0.99           # Discount factor for future rewards
        self.lr = 1e-3              # Learning rate
        self.batch_size = 64
        self.epsilon = 1.0          # 100% exploration at start
        self.epsilon_decay = 0.995  # Gradually shift from exploration to exploitation
        self.epsilon_min = 0.05
        
        # Q-Network and Target Network (Target network stabilizes training)
        self.q_net = QNetwork(state_dim, self.action_dim)
        self.target_net = QNetwork(state_dim, self.action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.memory = ReplayBuffer()

    def act(self, state, evaluate=False):
        """Chooses an action based on the current market state."""
        # Explore: Take a random action to learn the market
        if not evaluate and random.random() < self.epsilon:
            action_idx = random.randrange(self.action_dim)
        # Exploit: Trust the neural network's current understanding
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
            action_idx = torch.argmax(q_values).item()
            
        return self.action_choices[action_idx], action_idx

    def train(self):
        """Trains the neural network using a batch of past trades."""
        if len(self.memory) < self.batch_size:
            return
            
        # Pull a batch of data from memory
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        # Current Q Values
        q_values = self.q_net(states).gather(1, actions)
        
        # Next Q Values (calculated by the stable Target Network)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
        # Calculate loss (Mean Squared Error) and run backpropagation
        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon so the AI relies more on its brain and less on random guesses over time
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        """Syncs the target network with the main network periodically."""
        self.target_net.load_state_dict(self.q_net.state_dict())