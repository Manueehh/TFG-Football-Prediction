from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

app = FastAPI(title="LaLiga Prediction API")


mlflow.set_tracking_uri("http://localhost:5000")
model = mlflow.sklearn.load_model("models:/LaLiga_RF_1X2/1")

FEATURE_SELECTION = [
    'elo_home', 'elo_away', 'elo_diff',
    'home_avg_goals_scored_7', 'away_avg_goals_scored_7',
    'home_avg_goals_conceded_7', 'away_avg_goals_conceded_7',
    'goal_diff_form_home', 'goal_diff_form_away', 'total_avg_goals',
    'home_avg_shots_7', 'away_avg_shots_7',
    'home_avg_shots_on_target_7', 'away_avg_shots_on_target_7',
    'shots_on_target_ratio_home', 'shots_on_target_ratio_away',
    'attack_strength_home', 'attack_strength_away',
    'defense_strength_home', 'defense_strength_away',
    'discipline_index_home', 'discipline_index_away',
    'attack_vs_defense_home', 'attack_vs_defense_away',
    'home_team_value', 'away_team_value',
    'team_value_diff', 'team_value_ratio',
    'B365H_prob', 'B365D_prob', 'B365A_prob',
    'prob_diff_home_away', 'prob_fav_margin'
]
# aplicamos pydantic aqui
class MatchFeatures(BaseModel):
    elo_home: float
    elo_away: float
    elo_diff: float
    home_avg_goals_scored_7: float
    away_avg_goals_scored_7: float
    home_avg_goals_conceded_7: float
    away_avg_goals_conceded_7: float
    goal_diff_form_home: float
    goal_diff_form_away: float
    total_avg_goals: float
    home_avg_shots_7: float
    away_avg_shots_7: float
    home_avg_shots_on_target_7: float
    away_avg_shots_on_target_7: float
    shots_on_target_ratio_home: float
    shots_on_target_ratio_away: float
    attack_strength_home: float
    attack_strength_away: float
    defense_strength_home: float
    defense_strength_away: float
    discipline_index_home: float
    discipline_index_away: float
    attack_vs_defense_home: float
    attack_vs_defense_away: float
    home_team_value: float
    away_team_value: float
    team_value_diff: float
    team_value_ratio: float
    B365H_prob: float
    B365D_prob: float
    B365A_prob: float
    prob_diff_home_away: float
    prob_fav_margin: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(features: MatchFeatures):
    try:
        df = pd.DataFrame([features.model_dump()])
        df = df[FEATURE_SELECTION]
        
        prediction = model.predict(df)[0]
        probabilities = model.predict_proba(df)[0]
        classes = model.classes_.tolist()
        
        return {
            "prediction": prediction,
            "probabilities": dict(zip(classes, probabilities.tolist())),
            "confidence": float(max(probabilities))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))