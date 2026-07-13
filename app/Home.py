import streamlit as st

st.set_page_config(
    page_title="LaLiga Predictor",
    page_icon="⚽",
    layout="wide",
)

st.markdown("""
    <style>
    .block-container {
        max-width: 1200px;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("⚽ LaLiga Match Outcome Predictor")

st.markdown("""
Welcome to the **LaLiga match outcome prediction system**, developed as part of a
Final Degree Project on *Explainable Machine Learning Models for Forecasting Outcomes
in Professional Football*.

This application predicts the outcome of LaLiga matches using only information available
**before kick-off**, combining a cascaded machine learning architecture with SHAP
explanations to make each prediction transparent.
""")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Prediction")
    st.markdown("""
    Select an upcoming match, enter the starting lineups and retrieve the betting odds
    to obtain a prediction, its class probabilities and a SHAP explanation of the
    features that drove the decision.
    """)
    st.page_link("pages/1_Prediction.py", label="Go to Prediction")

with col2:
    st.subheader("Dataset")
    st.markdown("""
    Explore the dataset behind the models: around 7,600 LaLiga matches from 20 seasons
    (2005–2025), with results, statistics, betting odds, Elo ratings and squad values.
    """)
    st.page_link("pages/2_Dataset.py", label="Explore the Dataset")

st.divider()

st.caption(
    "Manuel Avilés Rodríguez · Escuela Superior de Informática · UCLM · 2026"
)