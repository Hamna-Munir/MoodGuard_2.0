import csv, os
from datetime import datetime

LOG_FILE = "data/sessions.csv"
HEADERS  = ["timestamp", "emotion", "confidence", "focus_score", "state", "alert", "tip"]

def init_log():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(HEADERS)

def log_entry(emotion, confidence, focus_score, state, alert, tip):
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            emotion, round(confidence, 1),
            focus_score, state, alert, tip
        ])

def read_log():
    import pandas as pd
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame(columns=HEADERS)
    return pd.read_csv(LOG_FILE, encoding='utf-8', encoding_errors='replace')

def get_session_summary(df):
    """Build the dict that Gemini needs."""
    if df.empty:
        return None
    emo_counts = df["emotion"].value_counts(normalize=True).mul(100).round(1).to_dict()
    alerts     = df[df["alert"]==True][["timestamp","state"]].tail(5)
    alert_list = [f"{row.state} at {row.timestamp[-5:]}" for _, row in alerts.iterrows()]
    return {
        "emotion_breakdown": emo_counts,
        "avg_focus":         round(df["focus_score"].mean(), 1),
        "dominant_state":    df["state"].mode()[0],
        "top_alerts":        alert_list,
        "duration_min":      len(df) * 5 // 60
    }