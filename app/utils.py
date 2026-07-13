import streamlit as st
import pandas as pd
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..")
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from calculate_market_values import normalize_text


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