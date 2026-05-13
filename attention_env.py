import gymnasium as gym
import numpy as np
from gymnasium import spaces

class AttentionEnv(gym.Env):
    """
    Advanced Attention Environment v3.0
    
    State Space (6 features):
        - Current attention score (0-1)
        - Attention trend (going up or down)
        - Time since last intervention
        - Consecutive low attention count
        - Session duration (normalized)
        - Blink rate (normalized)
    
    Actions:
        0 = Continue normally
        1 = Pause video
        2 = Slow down delivery
        3 = Simplify content
        4 = Add engagement break
    """

    def __init__(self):
        super(AttentionEnv, self).__init__()

        self.observation_space = spaces.Box(
            low=np.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0,  1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(5)

        self.action_names = [
            "Continue",
            "Pause Video",
            "Slow Down",
            "Simplify Content",
            "Add Break"
        ]

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_attention    = np.random.uniform(0.1, 1.0)
        self.prev_attention       = self.current_attention
        self.step_count           = 0
        self.max_steps            = 300
        self.time_since_intervene = 0
        self.low_attention_streak = 0
        self.session_progress     = 0.0
        self.blink_rate           = np.random.uniform(0.2, 0.8)
        self.cumulative_reward    = 0.0
        self.last_action          = 0

        return self._get_obs(), {}

    def _get_obs(self):
        trend = float(np.clip(
            (self.current_attention - self.prev_attention) * 5, -1.0, 1.0
        ))
        return np.array([
            self.current_attention,
            trend,
            min(self.time_since_intervene / 50.0, 1.0),
            min(self.low_attention_streak / 10.0, 1.0),
            self.session_progress,
            self.blink_rate,
        ], dtype=np.float32)

    def step(self, action):
        self.step_count          += 1
        self.session_progress     = self.step_count / self.max_steps
        self.prev_attention       = self.current_attention
        self.time_since_intervene += 1
        reward                    = 0.0

        # ── Track low attention streak ─────────────────────────
        if self.current_attention < 0.45:
            self.low_attention_streak += 1
        else:
            self.low_attention_streak = max(0, self.low_attention_streak - 1)

        # ── Penalty for repeating same action ─────────────────
        repetition_penalty = -0.2 if action == self.last_action else 0.0

        # ══ ACTION REWARD LOGIC ════════════════════════════════

        # Action 0: Continue
        if action == 0:
            if self.current_attention >= 0.70:
                reward = 2.0
            elif self.current_attention >= 0.55:
                reward = 0.5
            elif self.current_attention >= 0.40:
                reward = -0.5
            else:
                reward = -3.0   # heavy penalty for ignoring low attention

        # Action 1: Pause Video
        elif action == 1:
            if self.current_attention < 0.30:
                reward = 4.0    # very high — best action at critical level
                self.current_attention = min(1.0, self.current_attention + 0.30)
            elif self.current_attention < 0.45:
                reward = 3.0
                self.current_attention = min(1.0, self.current_attention + 0.20)
            elif self.current_attention < 0.60:
                reward = -1.5   # punish wrong use
            else:
                reward = -3.0   # heavy punishment when not needed
            self.time_since_intervene = 0

        # Action 2: Slow Down
        elif action == 2:
            if 0.40 <= self.current_attention < 0.65:
                reward = 2.0    # perfect range for slow down
                self.current_attention = min(1.0, self.current_attention + 0.12)
            elif self.current_attention < 0.40:
                reward = 0.3    # reduced — should use stronger action
                self.current_attention = min(1.0, self.current_attention + 0.08)
            else:
                reward = -0.8
            self.time_since_intervene = 0

        # Action 3: Simplify Content
        elif action == 3:
            if self.current_attention < 0.25:
                reward = 4.0    # best action at very low attention
                self.current_attention = min(1.0, self.current_attention + 0.30)
            elif self.current_attention < 0.40:
                reward = 3.0
                self.current_attention = min(1.0, self.current_attention + 0.18)
            elif self.current_attention < 0.65:
                reward = 0.0    # neutral
            else:
                reward = -2.0   # penalty for dumbing down when focused
            self.time_since_intervene = 0

        # Action 4: Add Break
        elif action == 4:
            if self.low_attention_streak >= 5:
                reward = 5.0    # highest reward in whole system
                self.current_attention = min(1.0, self.current_attention + 0.35)
            elif self.current_attention < 0.35:
                reward = 3.5
                self.current_attention = min(1.0, self.current_attention + 0.25)
            elif self.current_attention < 0.60:
                reward = -0.5
            else:
                reward = -3.0   # heavy penalty — break when focused
            self.time_since_intervene = 0

        # ── Bonus rewards ──────────────────────────────────────
        new_trend = self.current_attention - self.prev_attention
        if new_trend > 0.05:
            reward += 0.5   # attention improved

        if self.current_attention >= 0.75:
            reward += 0.3   # maintaining high attention

        if self.current_attention < 0.25 and action == 0:
            reward -= 2.0   # critical penalty for ignoring crisis

        reward += repetition_penalty

        # ── Natural attention drift ────────────────────────────
        if self.current_attention > 0.7:
            drift = np.random.uniform(0.005, 0.025)
        elif self.current_attention > 0.4:
            drift = np.random.uniform(0.01, 0.04)
        else:
            drift = np.random.uniform(0.02, 0.06)

        # Random natural refocus
        if np.random.random() < 0.05:
            drift = -np.random.uniform(0.05, 0.15)

        self.current_attention = float(np.clip(
            self.current_attention - drift, 0.0, 1.0
        ))

        self.blink_rate = float(np.clip(
            self.blink_rate + np.random.uniform(-0.05, 0.05), 0.0, 1.0
        ))

        self.last_action       = action
        self.cumulative_reward += reward

        obs        = self._get_obs()
        terminated = self.step_count >= self.max_steps
        truncated  = False

        return obs, float(reward), terminated, truncated, {}

    def render(self):
        print(f"Step {self.step_count:3d} | "
              f"Attention: {self.current_attention:.2f} | "
              f"Streak: {self.low_attention_streak} | "
              f"Total Reward: {self.cumulative_reward:.1f}")