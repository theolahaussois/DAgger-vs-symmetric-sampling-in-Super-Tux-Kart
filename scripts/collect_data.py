import numpy as np
from pathlib import Path
from src.envs.stk_wrapper import STKEnv
from src.agents.team1.agent1 import Agent1

def collect_expert_demonstrations(track_name, num_episodes, max_steps, save_path):
    env = STKEnv(track_name, max_steps)
    expert = Agent1(env)    
    dataset_states = []
    dataset_actions = []    
    for episode in range(num_episodes):
        obs, info = env.reset(seed=42 + episode, options={})
        expert.reset()        
        terminated = False
        truncated = False
        episode_reward = 0.0        
        while not (terminated or truncated):
            action = expert.choose_action(obs)            
            dataset_states.append(obs)
            dataset_actions.append(action)            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward            
        print(f"Épisode {episode + 1}/{num_episodes} terminé. Récompense : {episode_reward:.2f}")        
    env.close()    
    states_array = np.array(dataset_states, dtype=np.float32)
    actions_array = np.array(dataset_actions, dtype=np.float32)    
    Path(save_path).parent.mkdir(parents_exist=True, exist_ok=True)
    np.savez_compressed(save_path, states=states_array, actions=actions_array)
    print(f"Dataset expert sauvegardé dans {save_path} (Shape states: {states_array.shape})")

if __name__ == "__main__":
    collect_expert_demonstrations(
        track_name="cocoa_temple",
        num_episodes=5,
        max_steps=1000,
        save_path="runs/symmetric-sampling/expert_dataset.npz"
    )