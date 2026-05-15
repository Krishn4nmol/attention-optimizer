import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import glob

# Load all results
results_dir = "results"
all_files   = glob.glob(f"{results_dir}/*.csv")

adaptive_dfs = []
baseline_dfs = []

for f in all_files:
    df = pd.read_csv(f)
    if "adaptive" in df["mode"].values:
        adaptive_dfs.append(df[df["mode"] == "adaptive"])
    if "baseline" in df["mode"].values:
        baseline_dfs.append(df[df["mode"] == "baseline"])

adaptive_df = pd.concat(adaptive_dfs) if adaptive_dfs else pd.DataFrame()
baseline_df = pd.concat(baseline_dfs) if baseline_dfs else pd.DataFrame()

def metrics(df):
    if df.empty:
        return {}
    scores = df["attention_score"].values
    return {
        "avg":        round(np.mean(scores) * 100, 2),
        "std":        round(np.std(scores) * 100, 2),
        "focused":    round(np.mean(scores > 0.65) * 100, 2),
        "distracted": round(np.mean(scores < 0.40) * 100, 2),
        "drops":      int(sum(1 for i in range(1, len(scores))
                          if scores[i] < 0.45 and scores[i-1] >= 0.45)),
        "min":        round(np.min(scores) * 100, 2),
        "max":        round(np.max(scores) * 100, 2),
    }

am = metrics(adaptive_df)
bm = metrics(baseline_df)

print("="*50)
print("RESULTS SUMMARY")
print("="*50)
print(f"{'Metric':<25} {'Adaptive':>12} {'Baseline':>12} {'Improvement':>12}")
print("-"*60)
print(f"{'Avg Attention %':<25} {am['avg']:>12} {bm['avg']:>12} "
      f"{am['avg']-bm['avg']:>+12.2f}")
print(f"{'Time Focused %':<25} {am['focused']:>12} {bm['focused']:>12} "
      f"{am['focused']-bm['focused']:>+12.2f}")
print(f"{'Time Distracted %':<25} {am['distracted']:>12} {bm['distracted']:>12} "
      f"{am['distracted']-bm['distracted']:>+12.2f}")
print(f"{'Attention Drops':<25} {am['drops']:>12} {bm['drops']:>12} "
      f"{am['drops']-bm['drops']:>+12}")
print(f"{'Std Deviation %':<25} {am['std']:>12} {bm['std']:>12} "
      f"{am['std']-bm['std']:>+12.2f}")
print("="*50)

# Save summary
os.makedirs("results", exist_ok=True)
summary = pd.DataFrame({
    "Metric": ["Avg Attention %", "Time Focused %",
               "Time Distracted %", "Attention Drops", "Std Dev %"],
    "Adaptive": [am["avg"], am["focused"], am["distracted"],
                 am["drops"], am["std"]],
    "Baseline": [bm["avg"], bm["focused"], bm["distracted"],
                 bm["drops"], bm["std"]],
    "Improvement": [
        am["avg"] - bm["avg"],
        am["focused"] - bm["focused"],
        bm["distracted"] - am["distracted"],
        bm["drops"] - am["drops"],
        bm["std"] - am["std"]
    ]
})
summary.to_csv("results/summary.csv", index=False)
print("\nSummary saved to results/summary.csv")

# Plot 1 — Attention over time comparison
fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=adaptive_df["timestamp"],
    y=adaptive_df["attention_score"],
    mode="lines",
    name="Adaptive (AI)",
    line=dict(color="#00ff88", width=1.5),
    fill="tozeroy",
    fillcolor="rgba(0,255,136,0.05)"
))
fig1.add_trace(go.Scatter(
    x=baseline_df["timestamp"],
    y=baseline_df["attention_score"],
    mode="lines",
    name="Baseline (No AI)",
    line=dict(color="#ffaa00", width=1.5),
    fill="tozeroy",
    fillcolor="rgba(255,170,0,0.05)"
))
fig1.add_hline(y=0.65, line_dash="dash",
               line_color="white", opacity=0.5,
               annotation_text="Focus Threshold")
fig1.add_hline(y=0.40, line_dash="dash",
               line_color="red", opacity=0.5,
               annotation_text="Critical Threshold")
fig1.update_layout(
    title="Attention Score Over Time: Adaptive vs Baseline",
    xaxis_title="Time (seconds)",
    yaxis_title="Attention Score",
    yaxis=dict(range=[0, 1]),
    height=400,
    template="plotly_dark",
    legend=dict(x=0.01, y=0.99)
)
fig1.write_html("results/attention_comparison.html")
fig1.write_image("results/attention_comparison.png")
print("Chart saved to results/attention_comparison.png")

# Plot 2 — Bar chart
categories = ["Avg Attention %", "Time Focused %", "Time Distracted %"]
fig2 = go.Figure()
fig2.add_trace(go.Bar(
    name="Adaptive (AI)",
    x=categories,
    y=[am["avg"], am["focused"], am["distracted"]],
    marker_color="#00ff88",
    text=[f"{am['avg']}%", f"{am['focused']}%", f"{am['distracted']}%"],
    textposition="auto"
))
fig2.add_trace(go.Bar(
    name="Baseline (No AI)",
    x=categories,
    y=[bm["avg"], bm["focused"], bm["distracted"]],
    marker_color="#ffaa00",
    text=[f"{bm['avg']}%", f"{bm['focused']}%", f"{bm['distracted']}%"],
    textposition="auto"
))
fig2.update_layout(
    title="Performance Metrics: Adaptive vs Baseline",
    barmode="group",
    yaxis_title="Percentage (%)",
    height=400,
    template="plotly_dark"
)
fig2.write_html("results/metrics_comparison.html")
fig2.write_image("results/metrics_comparison.png")
print("Chart saved to results/metrics_comparison.png")

print("\n✅ Analysis complete!")
print("Open results/attention_comparison.html in browser to see charts")