# 🧠 Attention Span Optimizer for Online Learning

An AI system that monitors student attention during online classes
and automatically adapts content delivery in real time using
computer vision and reinforcement learning.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.9-green)
![Streamlit](https://img.shields.io/badge/Streamlit-latest-red)
![PPO](https://img.shields.io/badge/RL-PPO-orange)

---

## 🎯 What It Does

```
Webcam monitors your eyes
        ↓
AI computes attention score (0-100%)
        ↓
PPO agent decides: Continue / Slow Down / Pause / Simplify / Break
        ↓
VLC video player responds automatically
        ↓
Session data saved for analysis
```

---

## 🖥️ Requirements

### Hardware
- Laptop with webcam
- 8GB RAM minimum (16GB+ recommended)
- Windows 10/11, macOS, or Linux

### Software
- Python 3.11
- VLC Media Player — download from videolan.org

---

## ⚙️ Installation

### Step 1: Clone the repository
```bash
git clone https://github.com/Krishn4nmol/attention-optimizer.git
cd attention-optimizer
```

### Step 2: Create virtual environment
```bash
python -m venv venv
```

### Step 3: Activate virtual environment

Windows:
```bash
venv\Scripts\activate
```

macOS/Linux:
```bash
source venv/bin/activate
```

### Step 4: Install all libraries
```bash
pip install mediapipe==0.10.9
pip install opencv-python==4.8.1.78
pip install stable-baselines3
pip install gymnasium
pip install streamlit
pip install plotly
pip install python-vlc
pip install tensorboard
pip install numpy pandas
pip install kaleido
```

### Step 5: Install VLC Media Player
Download from: https://www.videolan.org/vlc/

---

## 🚀 How to Run

### Test your webcam first
```bash
python test_webcam.py
```

### Test face detection
```bash
python test_face.py
```

### Train the PPO agent
```bash
python train_agent.py
```
This takes 5-10 minutes. Model saved to models/attention_ppo_final

### Run live attention dashboard
```bash
streamlit run dashboard.py
```
Opens at http://localhost:8501

### Run adaptive video player
```bash
streamlit run vlc_player.py
```
Enter your video path and click Start Session.
VLC opens automatically and plays with AI control.

### Run experiment module
```bash
streamlit run experiment.py
```
Compare adaptive vs baseline attention sessions.

### Analyze results
```bash
python analyze_results.py
```
Generates comparison charts saved to results/ folder.

---

## 📁 Project Structure

```
attention-optimizer/
│
├── test_webcam.py          # Test webcam works
├── test_face.py            # Test MediaPipe face detection
├── attention_monitor.py    # Real-time eye tracking + attention score
├── attention_env.py        # Custom PPO RL environment
├── train_agent.py          # Train PPO agent (300k steps)
├── dashboard.py            # Live Streamlit attention dashboard
├── vlc_player.py           # VLC player with AI auto-pause/speed control
├── experiment.py           # Adaptive vs baseline experiment module
├── analyze_results.py      # Results analysis + chart generation
│
├── models/
│   ├── attention_ppo_final.zip    # Trained PPO model
│   └── best/                      # Best checkpoint during training
│
├── results/                       # Experiment CSV data + charts
├── checkpoints/                   # Training checkpoints
└── logs/                          # TensorBoard training logs
```

---

## 🧠 How Attention Score Works

Three components combined:

| Component | Weight | What it measures |
|---|---|---|
| Eye Aspect Ratio (EAR) | 40% | How open your eyes are |
| Gaze Score | 40% | Whether eyes are centered on screen |
| Head Pose | 20% | Whether head is facing forward |

```
Attention = 0.40 x EAR + 0.40 x Gaze + 0.20 x Head Pose
```

---

## 🤖 PPO Agent Actions

| Score Range | Action | What Happens |
|---|---|---|
| 0.70 - 1.00 | Continue | Video plays normally |
| 0.45 - 0.65 | Slow Down | Video slows to 0.6x speed |
| 0.30 - 0.45 | Pause Video | Video pauses automatically |
| 0.10 - 0.30 | Simplify | Video rewinds 15 seconds |
| 0.00 - 0.25 | Add Break | Video pauses, break recommended |

---

## 📊 Results

Experiment comparing adaptive vs baseline (2 participants, multiple sessions):

| Metric | Adaptive | Baseline | Improvement |
|---|---|---|---|
| Avg Attention | 79.11% | 78.46% | +0.65% |
| Time Focused | 90.87% | 89.07% | +1.80% |
| Time Distracted | 3.40% | 4.43% | -1.03% |
| Attention Stability (std) | 16.23% | 18.32% | -2.09% |

Key finding: Adaptive system reduced attention variability by 2.09%,
indicating more stable focus during learning sessions.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.11 | Core language |
| MediaPipe | Face mesh + eye landmark detection |
| OpenCV | Webcam capture + frame processing |
| Stable Baselines3 | PPO reinforcement learning |
| Gymnasium | Custom RL environment |
| Streamlit | Web dashboard UI |
| VLC (python-vlc) | Video playback control |
| Plotly | Interactive charts |
| TensorBoard | Training visualization |

---

## 📈 Training the Agent

The PPO agent was trained for 300,000 timesteps with:
- 4 parallel environments
- Network architecture: [256, 256, 128]
- Learning rate: 3e-4
- Entropy coefficient: 0.02 (for exploration)
- Batch size: 256

---

## 👤 Author

Krishn4nmol
- GitHub: https://github.com/Krishn4nmol/attention-optimizer

---

## 📄 License

MIT License - free to use for research and education.
