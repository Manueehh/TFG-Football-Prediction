import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import shap
import matplotlib.pyplot as plt
import datetime
import pickle
import requests

from datetime import date

API_KEY= "b1e997403f8e767cb315ed618df6a697"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..")
API_URL = "http://localhost:8001"
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

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
    df = pd.read_csv(os.path.join(ROOT_DIR, "data", "processed", "laliga_features.csv"))
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)

    players = pd.read_csv(os.path.join(ROOT_DIR, "data", "processed", "players_with_market_values.csv"))
    players["team_norm"] = players["team"].map(normalize_text)
    players["name_norm"] = players["name"].map(normalize_text)
    players["market_value"] = pd.to_numeric(players["market_value"], errors="coerce").fillna(0)

    teams = sorted(pd.concat([df["HomeTeam"], df["AwayTeam"]]).unique())
    return df, players, teams

def load_data_players(dataplayers: pd.DataFrame, team: str, season: str):
    team_normalizado = normalize_text(team)
    mask = (
        dataplayers['team_norm'].apply(
            lambda t: t == team_normalizado or t in team_normalizado or team_normalizado in t
        )
        & (dataplayers['Season'] == season)
    )
    list_players = dataplayers[mask]
    return list_players['name'].values



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

def get_laliga_odds():
    url = "https://api.the-odds-api.com/v4/sports/soccer_spain_la_liga/odds"
    params = {
        "apiKey" : API_KEY,
        "regions" : "eu",
        "markets" : "h2h",
        "oddsFormat" : "decimal"
    }
    response = requests.get(url, params=params)
    print(len(response.json()))
    return response.json()

def extract_pinnacle_odds(home_team, away_team, date, odds_data):
    for match in odds_data:
        if match['home_team'] == home_team and match['away_team'] == away_team:
            for bookmaker in match['bookmakers']:
                if bookmaker['key'] == 'pinnacle':
                    for market in bookmaker['markets']:
                        if market['key'] == 'h2h':
                            outcomes = {o['name']: o['price'] for o in market['outcomes']}
                            return {
                                'home_team': home_team,
                                'away_team': away_team,
                                'date': date,
                                'B365H': outcomes.get(home_team),
                                'B365D': outcomes.get('Draw'),
                                'B365A': outcomes.get(away_team)
                            }
    return None

    


# ── UI ──

st.set_page_config(page_title="Match Prediction - LaLiga", page_icon="⚽")
st.title("Match Prediction - LaLiga")
explainer = get_explainer(os.path.join(ROOT_DIR, "explainer_model_A.pkl"))

df, players, teams = load_data()

col1, col2, col3 = st.columns(3)
with col1:
    match_date = st.date_input("Date", value=date.today())
    season = get_season(pd.Timestamp(match_date))
with col2:
    home_team = st.selectbox("Home Team", teams)
    players_home_all = load_data_players(players,home_team,season)
with col3:
    away_team = st.selectbox("Away Team", teams, index=min(1, len(teams) - 1))
    players_away_all = load_data_players(players,away_team,season)

c1, c2 = st.columns(2)
with c1:
    if len(players_home_all) == 0:
        st.info(f"There are not data from {home_team} in {season} season")
    else:
        players_home = st.multiselect(options=players_home_all,label=f"Lineup {home_team}", max_selections = 11)
with c2:
    if len(players_away_all) == 0:
        st.info(f"There are not data from {away_team} in {season} season")
    else:
        players_away = st.multiselect(options=players_away_all,label=f"Lineup {home_team}", max_selections=11)

if "odds_values" not in st.session_state:
    st.session_state.odds_values = None
if "manual_odds" not in st.session_state:
    st.session_state.manual_odds = False

if st.button("Get Odds", type="primary"):
    with st.spinner("Fetching odds..."):
        odds = get_laliga_odds()
        result = extract_pinnacle_odds(home_team, away_team, match_date, odds)
    if result is not None:
        st.session_state.odds_values = result
        st.session_state.manual_odds = False
    else:
        st.session_state.odds_values = None
        st.session_state.manual_odds = True
        st.info(f"Couldn't find odds for {home_team} - {away_team} on {match_date}. Enter them manually.")

# Input manual si no se encontraron odds
if st.session_state.manual_odds:
    st.subheader("Manual Odds Input")
    m1, m2, m3 = st.columns(3)
    with m1:
        manual_h = st.number_input("Home odds", min_value=1.01, value=2.0, step=0.01)
    with m2:
        manual_d = st.number_input("Draw odds", min_value=1.01, value=3.0, step=0.01)
    with m3:
        manual_a = st.number_input("Away odds", min_value=1.01, value=4.0, step=0.01)

    if st.button("Confirm Odds", type="secondary"):
        st.session_state.odds_values = {
            'B365H': manual_h,
            'B365D': manual_d,
            'B365A': manual_a
        }
        st.session_state.manual_odds = False

if st.session_state.odds_values is not None:
    odds_bet = st.session_state.odds_values
    st.subheader("Odds")
    b1, b2, b3 = st.columns(3)
    with b1:
        st.info(f"Home: {odds_bet['B365H']:.2f}")
    with b2:
        st.info(f"Draw: {odds_bet['B365D']:.2f}")
    with b3:
        st.info(f"Away: {odds_bet['B365A']:.2f}")

    b365h = odds_bet['B365H']
    b365d = odds_bet['B365D']
    b365a = odds_bet['B365A']

    if st.button("Predict", type="primary"):
        if len(players_home) == 0 or len(players_away) == 0:
            st.warning("Introduce lineups.")
        elif home_team == away_team:
            st.warning("Teams must be different.")
        else:
            with st.spinner("Calculating..."):
                X = compute_features(df, players, match_date, home_team, away_team,
                             players_home, players_away, b365h, b365d, b365a)

                response = requests.post(
                    f"{API_URL}/predict",
                    json=X.iloc[0].to_dict()
                )
                result = response.json()

                pred = result['prediction']
                proba = list(result['probabilities'].values())
                classes = list(result['probabilities'].keys())

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