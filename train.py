import numpy as np
import torch
from environment import DeltaHedgingEnv
from agent import DQNAgent

def train_agent(episodes=500):
    env = DeltaHedgingEnv()
    agent = DQNAgent(state_dim=5)
    
    print("Starting Deep Q-Network Training...")
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # 1. AI observes the state and decides how many shares to hedge
            action_shares, action_idx = agent.act(state)
            
            # 2. The environment processes the trade and moves the market forward
            next_state, reward, done = env.step(action_shares)
            
            # 3. Save the result to memory
            agent.memory.push(state, action_idx, reward, next_state, done)
            
            # 4. Train the neural network on past experiences
            agent.train()
            
            state = next_state
            total_reward += reward
            
        # Sync the target network every 10 episodes for stability
        if episode % 10 == 0:
            agent.update_target_network()
            print(f"Episode: {episode:3d} | Total Reward: {total_reward:10.2f} | Epsilon: {agent.epsilon:.3f}")

    print("Training Complete. Saving model weights...")
    torch.save(agent.q_net.state_dict(), "dqn_hedger.pth")

if __name__ == "__main__":
    train_agent(episodes=200)