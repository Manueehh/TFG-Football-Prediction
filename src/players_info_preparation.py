import ast
import json 
import os
import pandas as pd

# Global variable for directory
base_dir = os.path.dirname(os.path.abspath(__file__))

def read_season_players():
    path = os.path.join(base_dir,'..','data', 'seasons_info')
    path = os.path.normpath(path)
    
    player_dfs = []

    for filename in os.listdir(path):
        if filename.endswith('.json'):
            file_path = os.path.join(path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                season_data = json.load(file)

            if 'players' in season_data:
                players = season_data['players']
                df = pd.DataFrame(players)
                player_dfs.append(df)

    if not player_dfs:
        return pd.DataFrame()

    all_players = pd.concat(player_dfs, ignore_index=True)
    all_players = all_players.drop_duplicates(subset='href')  # elimina duplicados

    return all_players

def get_matches():
    path = os.path.join(base_dir,'..','data', 'seasons_info')
    path = os.path.normpath(path)
    
    matchess = []
    for filename in os.listdir(path):
        if filename.endswith('.json'):
            file_path = os.path.join(path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                season_data = json.load(file)

            if 'players' in season_data:
                matches = season_data['rounds']
                df = pd.DataFrame(matches)
                matchess.append(df)

    if not matchess:
        return pd.DataFrame()
    
    all_matches = pd.concat(matchess, ignore_index=True)
    return all_matches

def get_info_from_matches():
    path = os.path.join(base_dir, '..', 'data', 'processed', 'matches_info.csv')
    path = os.path.normpath(path)
    df = pd.read_csv(path)

    all_matches = []

    for i, row in df.iterrows():
        try:
            matches = ast.literal_eval(row["matches"])
            for match in matches:
                all_matches.append({
                    "date_time": match.get("date_time"),
                    "home_team": match.get("home_team"),
                    "away_team": match.get("away_team"),
                    "home_lineup": match.get("home_lineup"),
                    "away_lineup": match.get("away_lineup")
                })
        except Exception as e:
            continue

    matches_lineups_df = pd.DataFrame(all_matches)
    return matches_lineups_df

def concat_dfs(players: pd.DataFrame, matches : pd.DataFrame):

    home_lineups = matches['home_lineup']
    home_lineups_names = get_names_lineups(home_lineups,players)
    away_lineups = matches['away_lineup']
    away_lineups_names = get_names_lineups(away_lineups,players)

    matches = matches.filter(['date_time','home_team','away_team'])
    matches['home_lineup_names'] = home_lineups_names
    matches['away_lineup_names'] = away_lineups_names

    return matches

def get_names_lineups(lineups : pd.Series, players : pd.Series):
    all_lineups = []
    for lineup in lineups:
        lineup_names = []
        for player in lineup:
            flag = players[players['href'] == player]
            if not flag.empty:
                name = flag.iloc[0]['name']
                lineup_names.append(name)
        all_lineups.append(lineup_names)
    return all_lineups

def transform_data(matches : pd.DataFrame):
    matches['date_time'] = pd.to_datetime(matches['date_time']).dt.date
    matches.rename(columns={'date_time' : 'Date'}, inplace=True)
    matches.rename(columns={'home_team' : 'HomeTeam'}, inplace=True)
    matches.rename(columns={'away_team' : 'AwayTeam'}, inplace=True)
    matches['HomeTeam'] = matches['HomeTeam'].astype('category')
    matches['AwayTeam'] = matches['AwayTeam'].astype('category')

    return matches
                
def write_csv(dataframe : pd.DataFrame, name : str):
    processed_folder = os.path.join(base_dir, '..' ,'data', 'processed')
    os.makedirs(processed_folder, exist_ok=True)
    output_file = os.path.join(processed_folder, name)
    dataframe.to_csv(output_file, index=False)

if __name__ == "__main__":
    players_data = read_season_players()
    write_csv(players_data,'players_info.csv')
    matches_data = get_matches()
    write_csv(matches_data, 'matches_info.csv')
    matches_lineups = get_info_from_matches()
    write_csv(matches_lineups, 'matches_lineups.csv')
    matches_final = concat_dfs(players_data,matches_lineups)
    matches_final = transform_data(matches_final)
    write_csv(matches_final, 'matches_final_info.csv')
