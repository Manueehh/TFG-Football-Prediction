import ast
import re
import unicodedata
import pandas as pd
import os
from difflib import SequenceMatcher

base_dir = os.path.dirname(os.path.abspath(__file__))


def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def parse_market_values(v):
    if not isinstance(v, str):
        return 0.0

    s = v.lower().strip()
    s = s.replace(",", ".")
    s = (
        s.replace("€", "")
        .replace("millones", "m")
        .replace("millon", "m")
        .replace("mill.", "m")
        .replace("mill", "m")
        .replace(" mil", "k")
    )

    nums = re.findall(r"[\d.]+", s)
    if not nums:
        return 0.0

    try:
        n = float(nums[0])
    except ValueError:
        return 0.0

    if "m" in s:
        n *= 1_000_000
    elif "k" in s:
        n *= 1_000

    return float(int(n))


def parse_list(val):
    if isinstance(val, list):
        return val
    if not isinstance(val, str):
        return []
    try:
        val = ast.literal_eval(val)
        if isinstance(val, list):
            return [str(x) for x in val]
    except Exception:
        pass
    return [x.strip() for x in val.split(",") if x.strip()]

def token_set_score(a_norm: str, b_norm: str) -> float:
    sa, sb = set(a_norm.split()), set(b_norm.split())
    if not sa:
        return 0.0
    return len(sa & sb) / len(sa)


def match_player_value(pname_raw: str, season: str, team_norm: str, players_df: pd.DataFrame) -> float:
    pname = normalize_text(pname_raw)
    if not pname:
        return 0.0

    mask = (players_df["Season"] == season) & (players_df["team_norm"] == team_norm)
    roster = players_df.loc[mask, ["name_norm", "market_value"]]
    if roster.empty:
        return 0.0

    exact = roster.loc[roster["name_norm"] == pname]
    if not exact.empty:
        return float(exact.iloc[0]["market_value"])

    roster["score"] = roster["name_norm"].apply(lambda n: token_set_score(pname, n))
    best_idx = roster["score"].idxmax()
    best_score = roster.loc[best_idx, "score"]

    if best_score < 0.6:
        roster["fuzzy"] = roster["name_norm"].apply(lambda n: SequenceMatcher(None, pname, n).ratio())
        best_idx = roster["fuzzy"].idxmax()
        best_score = roster.loc[best_idx, "fuzzy"]

    if best_score >= 0.55:
        return float(roster.loc[best_idx, "market_value"])

    return 0.0


def team_value(lineup, team_norm, season, players_df: pd.DataFrame) -> float:
    if not isinstance(lineup, list) or not lineup:
        return 0.0

    total = 0.0
    for raw_name in lineup:
        total += match_player_value(raw_name, season, team_norm, players_df)
    return float(total)


def load_player_values():
    path = os.path.join(base_dir, "..", "data", "processed", "transfermarket_values")
    path = os.path.normpath(path)
    player_values = []

    for filename in os.listdir(path):
        if filename.endswith(".csv"):
            season = filename.split("_")[2].replace(".csv", "")
            s1, s2 = season.split("-")
            season_fmt = f"{s1}_{s2[-2:]}"
            file_path = os.path.join(path, filename)

            df = pd.read_csv(file_path)
            df["Season"] = season_fmt
            df = df.filter(["name", "market_value", "team", "Season"])
            df["market_value"] = df["market_value"].apply(parse_market_values)
            player_values.append(df)

    return player_values


def concat_dfs(dataframes: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(dataframes, ignore_index=True)


def write_csv(dataframe: pd.DataFrame, filename: str):
    processed_folder = os.path.join(base_dir, "..", "data", "processed")
    os.makedirs(processed_folder, exist_ok=True)
    output_file = os.path.join(processed_folder, filename)
    dataframe.to_csv(output_file, index=False)
    print(f"Archivo guardado en {output_file}")


def load_matches() -> pd.DataFrame:
    path = os.path.join(base_dir, "..", "data", "processed", "laliga_features.csv")
    path = os.path.normpath(path)
    return pd.read_csv(path)


def add_team_values_to_features(features: pd.DataFrame, players_df: pd.DataFrame):
    players_df["team_norm"] = players_df["team"].map(normalize_text)
    players_df["name_norm"] = players_df["name"].map(normalize_text)
    players_df["market_value"] = pd.to_numeric(players_df["market_value"], errors="coerce").fillna(0)

    features["home_team_norm"] = features["HomeTeam"].map(normalize_text)
    features["away_team_norm"] = features["AwayTeam"].map(normalize_text)
    features["Home_Lineup_List"] = features["Home_Lineup_List"].apply(parse_list)
    features["Away_Lineup_List"] = features["Away_Lineup_List"].apply(parse_list)


    features["home_team_value"] = features.apply(
        lambda row: team_value(row["Home_Lineup_List"], row["home_team_norm"], row["Season"], players_df),
        axis=1,
    )

    features["away_team_value"] = features.apply(
        lambda row: team_value(row["Away_Lineup_List"], row["away_team_norm"], row["Season"], players_df),
        axis=1,
    )

    return features


if __name__ == "__main__":
    players_list = load_player_values()
    players_df = concat_dfs(players_list)
    write_csv(players_df, "players_with_market_values.csv")

    features = load_matches()
    features = add_team_values_to_features(features, players_df)

    write_csv(features, "laliga_features.csv")
