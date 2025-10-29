# ==========================================================
#  feature_engineering.py
#  Football Prediction
#  Author: Manuel Avilés Rodríguez
# ==========================================================

import pandas as pd
import os
import numpy as np

# Global variable for directory
base_dir = os.path.dirname(os.path.abspath(__file__))

def rolling_feature(df : pd.DataFrame, team_col : str, value_col : str, new_col : str, window=7):
    """
    Rolling mean for each team. Shift(1) avoids the actual match and only takes into account the 7 previous matches
    """
    df[new_col] = (
        df.groupby(team_col)[value_col]
          .apply(lambda x: x.shift(1).rolling(window).mean())
          .reset_index(level=0, drop=True)
    )
    return df

def add_elo_features(df : pd.DataFrame, k_factor=20):
    """
    ELO ratings for each team
    """
    teams = pd.concat([df["HomeTeam"], df["AwayTeam"]]).unique()
    elo = {team: 1500 for team in teams}
    
    elo_home, elo_away = [], []
    
    for _, row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        elo_home.append(elo[home])
        elo_away.append(elo[away])
        
        if row["FTHG"] > row["FTAG"]:
            score_home, score_away = 1, 0
        elif row["FTHG"] < row["FTAG"]:
            score_home, score_away = 0, 1
        else:
            score_home = score_away = 0.5
        

        exp_home = 1 / (1 + 10 ** ((elo[away] - elo[home]) / 400))
        exp_away = 1 - exp_home
        

        elo[home] += k_factor * (score_home - exp_home)
        elo[away] += k_factor * (score_away - exp_away)
    
    df["elo_home"] = elo_home
    df["elo_away"] = elo_away
    df["elo_diff"] = df["elo_home"] - df["elo_away"]
    
    return df


def add_form_features(df : pd.DataFrame, window=7):
    """
    Average of goals scored by home and away teams.
    """
    df = rolling_feature(df, "HomeTeam", "FTHG", "home_avg_goals_scored_7", window)
    df = rolling_feature(df, "AwayTeam", "FTAG", "away_avg_goals_scored_7", window)
    df = rolling_feature(df, "HomeTeam", "FTAG", "home_avg_goals_conceded_7", window)
    df = rolling_feature(df, "AwayTeam", "FTHG", "away_avg_goals_conceded_7", window)
    
    df["goal_diff_form_home"] = df["home_avg_goals_scored_7"] - df["home_avg_goals_conceded_7"]
    df["goal_diff_form_away"] = df["away_avg_goals_scored_7"] - df["away_avg_goals_conceded_7"]
    
    return df


def add_stat_features(df : pd.DataFrame, window=7):
    """
    Rolling averages for each team (shots, shots on target, corners, etc...)
    """
    cols = [
        ("HomeTeam", "HS", "home_avg_shots_7"),
        ("AwayTeam", "AS", "away_avg_shots_7"),
        ("HomeTeam", "HST", "home_avg_shots_on_target_7"),
        ("AwayTeam", "AST", "away_avg_shots_on_target_7"),
        ("HomeTeam", "HC", "home_avg_corners_7"),
        ("AwayTeam", "AC", "away_avg_corners_7"),
        ("HomeTeam", "HF", "home_avg_fouls_7"),
        ("AwayTeam", "AF", "away_avg_fouls_7"),
        ("HomeTeam", "HY", "home_avg_yellows_7"),
        ("AwayTeam", "AY", "away_avg_yellows_7"),
        ("HomeTeam", "HR", "home_avg_reds_7"),
        ("AwayTeam", "AR", "away_avg_reds_7"),
    ]
    
    for team_col, value_col, new_col in cols:
        if value_col in df.columns:
            df = rolling_feature(df, team_col, value_col, new_col, window)
    
    return df


def add_index_features(df : pd.DataFrame):
    """
    Various statistics, like attack strength, defense strength, discipline...
    """
    df["attack_strength_home"] = (
        df["home_avg_goals_scored_7"] + 0.1 * df["home_avg_shots_on_target_7"]
    )
    df["attack_strength_away"] = (
        df["away_avg_goals_scored_7"] + 0.1 * df["away_avg_shots_on_target_7"]
    )
    df["defense_strength_home"] = 1 / (
        df["home_avg_goals_conceded_7"] + 0.1 * df["home_avg_yellows_7"] + 1e-3
    )
    df["defense_strength_away"] = 1 / (
        df["away_avg_goals_conceded_7"] + 0.1 * df["away_avg_yellows_7"] + 1e-3
    )
    df["discipline_index_home"] = 1 - (
        (0.5 * df["home_avg_yellows_7"] + 1.5 * df["home_avg_reds_7"]) / 10
    )
    df["discipline_index_away"] = 1 - (
        (0.5 * df["away_avg_yellows_7"] + 1.5 * df["away_avg_reds_7"]) / 10
    )
    
    return df


def add_market_features(df : pd.DataFrame):
    """
    Odd marks into probabilities
    """

    cols = ["B365H", "B365D", "B365A"]
    if not all(c in df.columns for c in cols):
        print("⚠️  No se encontraron todas las columnas de cuotas (B365H/D/A). Se omite esta transformación.")
        return df
    
    for c in cols:
        df[f"{c}_prob"] = 1 / df[c]

    sum_probs = df[[f"{c}_prob" for c in cols]].sum(axis=1)
    for c in cols:
        df[f"{c}_prob"] = df[f"{c}_prob"] / sum_probs

    df["prob_diff_home_away"] = df["B365H_prob"] - df["B365A_prob"] 
    df["prob_fav_margin"] = df[[f"{c}_prob" for c in cols]].max(axis=1) - df[[f"{c}_prob" for c in cols]].min(axis=1)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    return df



def generate_features(df : pd.DataFrame):
    """
    Computes all the features
    """
    
    df = df.sort_values("Date")
    df = add_elo_features(df)
    df = add_form_features(df)
    df = add_stat_features(df)
    df = add_index_features(df)
    if "B365H" in df.columns:
        df = add_market_features(df)
    
    return df

def join_with_matches(data_features : pd.DataFrame):
    path = os.path.join(base_dir, '..', 'data', 'processed', 'matches_final_info.csv')
    path = os.path.normpath(path)
    matches_lineups = pd.read_csv(path)

    df_joined = pd.merge(
        data_features,
        matches_lineups,
        on=['Date','HomeTeam','AwayTeam'],
        how='left'
    )

    return df_joined


if __name__ == "__main__":
    path = os.path.join(base_dir, '..', 'data', 'processed','LaLiga_combined.csv')
    path = os.path.normpath(path)
    df = pd.read_csv(path)
    df = generate_features(df)
    df = join_with_matches(df)
    df.to_csv("data/processed/laliga_features.csv", index=False)
