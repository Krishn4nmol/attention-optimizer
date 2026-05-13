from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from attention_env import AttentionEnv
import os

print("Setting up environment...")
env = AttentionEnv()

# Verify environment is correct
check_env(env)
print("Environment OK!")

# Create PPO agent
print("Creating PPO agent...")
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
)

# Train it
print("Training started... (takes 1-2 minutes)")
model.learn(total_timesteps=50000)

# Save the trained model
os.makedirs("models", exist_ok=True)
model.save("models/attention_ppo")
print("Model saved to models/attention_ppo")

# Quick test
print("\nTesting trained agent...")
obs, _ = env.reset()
for i in range(10):
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)
    action_names = ["Continue","Pause","Slow Down","Simplify","Add Break"]
    print(f"Attention: {obs[0]:.2f} → Action: {action_names[action]} | Reward: {reward:.1f}")
    if done:
        break