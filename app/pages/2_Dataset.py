import streamlit as st
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_data

st.set_page_config(page_title="Dataset - LaLiga", layout="wide")

st.markdown("""
    <style>
    .block-container { max-width: 1200px; padding-left: 3rem; padding-right: 3rem; }
    </style>
""", unsafe_allow_html=True)

st.title("Dataset Overview")
st.caption("Around 7,600 LaLiga matches from 20 seasons (2005–2025)")

df, players, teams = load_data()

st.subheader("At glance")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Matches", f"{len(df)}")
m2.metric("Seasons", df["Season"].nunique())
m3.metric("Teams", len(teams))
m4.metric("Players", f"{players['name'].nunique()}")

st.divider()

st.subheader("Explore matches")
f1, f2, f3 = st.columns(3)
with f1:
    seasons = ["All"] + sorted(df["Season"].unique(), reverse=True)
    sel_season = st.selectbox("Season", seasons)
with f2:
    sel_team = st.selectbox("Team", ["All"] + teams)
with f3:
    results = {"All": None, "Home win (1)": "H", "Draw (X)": "D", "Away win (2)": "A"}
    sel_result = st.selectbox("Result", list(results.keys()))

view = df.copy()
if sel_season != "All":
    view = view[view["Season"] == sel_season]
if sel_team != "All":
    view = view[(view["HomeTeam"] == sel_team) | (view["AwayTeam"] == sel_team)]
if results[sel_result] is not None:
    view = view[view["FTR"] == results[sel_result]]

st.caption(f"{len(view)} matches match the current filters")


st.dataframe(
    view.sort_values("Date", ascending=False).reset_index(drop=True),
    use_container_width=True, height=350,
)

st.divider()


st.subheader("Outcome distribution")
c1, c2 = st.columns(2)

with c1:
    st.caption("Full-time results (1 / X / 2)")
    ftr_counts = view["FTR"].value_counts().reindex(["H", "D", "A"]).fillna(0)
    ftr_counts.index = ["Home win", "Draw", "Away win"]
    st.bar_chart(ftr_counts)

with c2:
    st.caption("Matches per season")
    per_season = view.groupby("Season").size()
    st.bar_chart(per_season)


st.subheader("Draw rate by season")
st.caption("The share of matches ending in a draw: the hardest outcome to predict")
draw_rate = (
    df.assign(is_draw=(df["FTR"] == "D").astype(int))
      .groupby("Season")["is_draw"].mean()
)
st.line_chart(draw_rate)


if {"FTHG", "FTAG"}.issubset(view.columns):
    st.subheader("Goals per match")
    total_goals = (view["FTHG"] + view["FTAG"]).value_counts().sort_index()
    st.bar_chart(total_goals)