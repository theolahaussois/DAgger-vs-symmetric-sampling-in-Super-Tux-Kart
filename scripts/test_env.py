from src.envs.stk_wrapper import STKEnv

def main():
    env = STKEnv("cocoa_temple", 500)   
    obs, info = env.reset(seed=42, options={})
    print(f"Observation initiale brute : shape={obs.shape}")    
    terminated = False
    truncated = False
    total_reward = 0.0    
    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward        
    print(f"Épisode terminé. Récompense totale : {total_reward:.2f}")
    print(f"Terminé : {terminated}, Tronqué : {truncated}")    
    env.close()

if __name__ == "__main__":
    main()