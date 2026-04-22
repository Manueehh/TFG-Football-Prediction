import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
import glob
import shap
import matplotlib.pyplot as plt
from datetime import date

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from feature_engineering import generate_features, get_season
from calculate_market_values import (
    normalize_text, match_player_value, add_various_features
)

FEATURES = [
    "elo_home", "elo_away", "elo_diff",
    "home_avg_goals_scored_7", "away_avg_goals_scored_7",
    "home_avg_goals_conceded_7", "away_avg_goals_conceded_7",
    "goal_diff_form_home", "goal_diff_form_away",
    "total_avg_goals",
    "home_avg_shots_7", "away_avg_shots_7",
    "home_avg_shots_on_target_7", "away_avg_shots_on_target_7",
    "shots_on_target_ratio_home", "shots_on_target_ratio_away",
    "attack_strength_home", "attack_strength_away",
    "defense_strength_home", "defense_strength_away",
    "discipline_index_home", "discipline_index_away",
    "attack_vs_defense_home", "attack_vs_defense_away",
    "home_team_value", "away_team_value",
    "team_value_diff", "team_value_ratio",
    "B365H_prob", "B365D_prob", "B365A_prob",
    "prob_diff_home_away", "prob_fav_margin",
]


@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "laliga_features.csv"))
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)

    players = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "players_with_market_values.csv"))
    players["team_norm"] = players["team"].map(normalize_text)
    players["name_norm"] = players["name"].map(normalize_text)
    players["market_value"] = pd.to_numeric(players["market_value"], errors="coerce").fillna(0)

    teams = sorted(pd.concat([df["HomeTeam"], df["AwayTeam"]]).unique())
    return df, players, teams


def compute_features(df_hist, players_df, match_date, home_team, away_team,
                     home_lineup, away_lineup, b365h, b365d, b365a):
    df_before = df_hist[df_hist["Date"] < pd.Timestamp(match_date)].copy()

    new_row = pd.DataFrame([{
        "Div": "SP1", "Date": pd.Timestamp(match_date),
        "HomeTeam": home_team, "AwayTeam": away_team,
        "FTHG": 0, "FTAG": 0, "FTR": "D",
        "HTHG": 0, "HTAG": 0, "HTR": "D",
        "HS": 0, "AS": 0, "HST": 0, "AST": 0,
        "HF": 0, "AF": 0, "HC": 0, "AC": 0,
        "HY": 0, "AY": 0, "HR": 0, "AR": 0,
        "B365H": b365h, "B365D": b365d, "B365A": b365a,
    }])

    df_full = pd.concat([df_before, new_row], ignore_index=True)
    df_full = generate_features(df_full)

    result = df_full.iloc[[-1]].copy()

    season = get_season(pd.Timestamp(match_date))
    home_tn = normalize_text(home_team)
    away_tn = normalize_text(away_team)

    result["home_team_value"] = sum(
        match_player_value(p.strip(), season, home_tn, players_df) for p in home_lineup
    )
    result["away_team_value"] = sum(
        match_player_value(p.strip(), season, away_tn, players_df) for p in away_lineup
    )

    result = add_various_features(result)
    result = result.fillna(0)

    return result[FEATURES]

def get_explainer(path):
    with open(path, 'rb') as f:
        explainer = pickle.load(f)
    return explainer

def get_shap_values(explainer, X):
    shap_values = explainer(X)
    return shap_values
    


# ── UI ──

st.set_page_config(page_title="Match Prediction - LaLiga", page_icon="⚽")
st.title("Match Prediction - LaLiga")
explainer = get_explainer(os.path.join(BASE_DIR, "explainer_model_A.pkl"))

df, players, teams = load_data()

pkl_files = glob.glob(os.path.join(BASE_DIR, "*.pkl"))
if not pkl_files:
    st.error("No .pkl on directory.")
    st.stop()

model_name = st.selectbox("Model", [os.path.basename(f) for f in pkl_files])

col1, col2, col3 = st.columns(3)
with col1:
    match_date = st.date_input("Date", value=date.today())
with col2:
    home_team = st.selectbox("Home Team", teams)
with col3:
    away_team = st.selectbox("Away Team", teams, index=min(1, len(teams) - 1))

c1, c2 = st.columns(2)
with c1:
    home_lineup_str = st.text_area(f"Lineup {home_team}", placeholder="Player1, Player2, ...")
with c2:
    away_lineup_str = st.text_area(f"Lineup {away_team}", placeholder="Player1, Player2, ...")

st.subheader("Odds Bet365")
b1, b2, b3 = st.columns(3)
with b1:
    b365h = st.number_input("Home", min_value=1.01, value=2.00, step=0.01)
with b2:
    b365d = st.number_input("Draw", min_value=1.01, value=3.25, step=0.01)
with b3:
    b365a = st.number_input("Away", min_value=1.01, value=3.50, step=0.01)

if st.button("Predict", type="primary"):
    home_lineup = [p.strip() for p in home_lineup_str.split(",") if p.strip()]
    away_lineup = [p.strip() for p in away_lineup_str.split(",") if p.strip()]

    if not home_lineup or not away_lineup:
        st.warning("Introduce lineups.")
    elif home_team == away_team:
        st.warning("Teams must be different.")
    else:
        with st.spinner("Calculating..."):
            X = compute_features(df, players, match_date, home_team, away_team,
                                 home_lineup, away_lineup, b365h, b365d, b365a)

            with open(os.path.join(BASE_DIR, model_name), "rb") as f:
                model = pickle.load(f)

            pred = model.predict(X)[0]
            proba = model.predict_proba(X)[0]
            classes = model.classes_

        st.divider()
        st.subheader("Result")
        shap_values = get_shap_values(explainer, X)

        if str(pred) == "1":
            st.success(f"**Prediction: {home_team} win (1)**")
            cols = st.columns(len(classes))
            for i, (cls, prob) in enumerate(zip(classes, proba)):
                label = "1 (Home)" if str(cls) == "1" else "X2 (Draw/Away)"
                cols[i].metric(label, f"{prob * 100:.1f}%")
            with st.expander("SHAP Values"):
                fig = plt.figure()
                shap.plots.waterfall(shap_values[0,:,0], show=False)
                st.pyplot(fig)
        else:
            st.info(f"**Prediction: Draw or {away_team} win (X2)**")
            cols = st.columns(len(classes))
            for i, (cls, prob) in enumerate(zip(classes, proba)):
                label = "1 (Home)" if str(cls) == "1" else "X2 (Draw/Away)"
                cols[i].metric(label, f"{prob * 100:.1f}%")
            with st.expander("SHAP Values"):
                fig = plt.figure()
                shap.plots.waterfall(shap_values[0,:,1], show=False)
                st.pyplot(fig)

        with st.expander("Features"):
            st.dataframe(X.T.rename(columns={X.index[0]: "Value"}))

    
    
