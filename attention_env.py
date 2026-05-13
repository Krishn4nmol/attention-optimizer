import gymnasium as gym
import numpy as np
from gymnasium import spaces

class AttentionEnv(gym.Env):
    """
    Custom RL Environment for attention-based content adaptation.
    
    State:  attention score (0.0 to 1.0)
    Actions:
        0 = Continue normally
        1 = Pause video
        2 = Slow down delivery
        3 = Simplify content
        4 = Add engagement break
    """

    def __init__(self):
        super(AttentionEnv, self).__init__()

        # One value: current attention score
        self.observation_space = spaces.Box(
            low=np.array([0.0]),
            high=np.array([1.0]),
            dtype=np.float32
        )

        # 5 possible actions
        self.action_space = spaces.Discrete(5)

        self.action_names = [
            "Continue",
            "Pause Video",
            "Slow Down",
            "Simplify Content",
            "Add Break"
        ]

        self.current_attention = 1.0
        self.step_count = 0
        self.max_steps = 200

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Start with random attention score
        self.current_attention = np.random.uniform(0.3, 1.0)
        self.step_count = 0
        return np.array([self.current_attention], dtype=np.float32), {}

    def step(self, action):
        self.step_count += 1
        reward = 0.0

        # ── Reward Logic ──────────────────────────────────────
        # High attention + Continue = good
        if action == 0:  # Continue
            if self.current_attention >= 0.65:
                reward = 1.0   # correct decision
            else:
                reward = -1.0  # wrong — should have intervened

        # Low attention + Pause = good
        elif action == 1:  # Pause
            if self.current_attention < 0.40:
                reward = 1.5   # correct intervention
                self.current_attention = min(1.0, self.current_attention + 0.2)
            else:
                reward = -0.5  # unnecessary pause

        # Moderate attention + Slow Down = good
        elif action == 2:  # Slow Down
            if 0.40 <= self.current_attention < 0.65:
                reward = 1.0
                self.current_attention = min(1.0, self.current_attention + 0.1)
            else:
                reward = -0.3

        # Very low attention + Simplify = good
        elif action == 3:  # Simplify
            if self.current_attention < 0.35:
                reward = 1.5
                self.current_attention = min(1.0, self.current_attention + 0.25)
            else:
                reward = -0.3

        # Any low attention + Break = good
        elif action == 4:  # Add Break
            if self.current_attention < 0.50:
                reward = 1.0
                self.current_attention = min(1.0, self.current_attention + 0.15)
            else:
                reward = -0.5

        # Attention naturally drifts down over time
        self.current_attention = max(0.0,
            self.current_attention - np.random.uniform(0.01, 0.05)
        )

        obs = np.array([self.current_attention], dtype=np.float32)
        terminated = self.step_count >= self.max_steps
        truncated = False

        return obs, reward, terminated, truncated, {}

    def render(self):
        action_taken = self.action_names[0]
        print(f"Attention: {self.current_attention:.2f} | Step: {self.step_count}")