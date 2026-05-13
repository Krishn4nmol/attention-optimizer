import multiprocessing
multiprocessing.freeze_support()

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from attention_env import AttentionEnv
import numpy as np
import os

if __name__ == "__main__":

    os.makedirs("models",      exist_ok=True)
    os.makedirs("logs",        exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    print("="*50)
    print("Attention PPO Trainer v2.0")
    print("Safe mode — no overheating")
    print("="*50)

    print("\nChecking environment...")
    check_env(AttentionEnv())
    print("Environment OK!")

    N_ENVS = 4
    print(f"\nSpinning up {N_ENVS} environments (laptop safe)...")

    vec_env = DummyVecEnv([lambda: Monitor(AttentionEnv()) for _ in range(N_ENVS)])
    vec_env = VecMonitor(vec_env)

    eval_env = Monitor(AttentionEnv())
    print("Environments ready!")

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path="./checkpoints/",
        name_prefix="attention_ppo"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best/",
        log_path="./logs/",
        eval_freq=10_000,
        n_eval_episodes=20,
        deterministic=True,
        verbose=1
    )

    print("\nBuilding PPO agent...")
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=15,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.02,
        policy_kwargs=dict(
            net_arch=[256, 256, 128],
        ),
        tensorboard_log="./logs/",
    )

    print("Agent ready!")
    print("\nTraining started...")
    print("Takes about 5 minutes")
    print("-"*50)

    model.learn(
        total_timesteps=300_000,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )

    model.save("models/attention_ppo_final")
    print("\nModel saved!")

    # ── Test with FORCED attention scenarios ───────────────────
    print("\n" + "="*50)
    print("TESTING ALL ATTENTION RANGES")
    print("="*50)

    test_env = AttentionEnv()
    action_counts = {name: 0 for name in test_env.action_names}
    rewards = []

    test_scenarios = [
        (0.85, "Very High"),
        (0.75, "High"),
        (0.65, "Medium-High"),
        (0.55, "Medium"),
        (0.45, "Medium-Low"),
        (0.35, "Low"),
        (0.25, "Very Low"),
        (0.15, "Critical"),
        (0.10, "Extremely Low"),
    ]

    print(f"\n{'Attention':<12} {'Range':<15} {'Action':<22} {'Reward'}")
    print("-"*60)

    for attn_val, label in test_scenarios:
        obs, _ = test_env.reset()
        test_env.current_attention = attn_val
        test_env.low_attention_streak = 6 if attn_val < 0.3 else 0
        obs = test_env._get_obs()

        action, _ = model.predict(obs, deterministic=True)
        _, reward, _, _, _ = test_env.step(action)
        name = test_env.action_names[action]
        action_counts[name] += 1
        rewards.append(float(reward))

        print(f"{attn_val:<12.2f} {label:<15} {name:<22} {reward:+.1f}")

    print("\n" + "="*50)
    print("ACTION DISTRIBUTION:")
    print("="*50)
    for action, count in action_counts.items():
        bar = "█" * (count * 3)
        print(f"{action:<20} {bar} ({count})")

    print(f"\nAverage Reward : {np.mean(rewards):.2f}")
    print("="*50)
    print("\nDone! Ready for Module 3.")