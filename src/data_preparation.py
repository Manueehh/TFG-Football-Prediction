# ==========================================================
#  data_preparation.py
#  Football Prediction
#  Author: Manuel Avilés Rodríguez
# ==========================================================

import pandas as pd
import os

# Global variable for directory
base_dir = os.path.dirname(os.path.abspath(__file__))


def data_concat_and_selection(dataframes : list[pd.DataFrame]):
    f_df = pd.DataFrame()
    # Data concatenation, this is for Primera Division
    for dataframe in dataframes:
        f_df = pd.concat([f_df,dataframe],ignore_index=True)

    """
    Data cleansing, filtering the useful columns. The keys for the columns are the following:
    Div : Division (in this case SP1)
    Date : Date when the game happened
    HomeTeam : Local Team
    AwayTeam : Visiting Team
    FTHG : Full Time Home Team Goals
    FTAG : Full Time Away Team Goals
    FTR : Full Time Result (H,D,A)
    HTHG : Half-Time Home Team Goals
    HTAG : Half-Time Away Team Goals
    HTR : Half-Time Result
    HS : Home Team Shots
    AS : Away Team Shots
    HST : Home Team Shots on Target
    AST : Away Team Shots on Target
    HF : Home Team Fouls
    AF : Away Team Fouls
    HC : Home Team Corners
    AC : Away Team Corners
    HY : Home Team Yellow Cards
    AY : Away Team Yellow Cards
    HR : Home Team Red Cards
    AR : Away Team Red Cards
    B365H : Bet365 odds for winning Home Team
    B365D : Bet365 odds for draw
    B365A : Bet365 odds for winning Away Team
    """
    f_df_cols = f_df.filter(['Div','Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR','HTHG','HTAG','HTR','HS','AS','HST','AST','HF','AF','HC','AC',
                              'HY','AY','HR','AR','B365H','B365D','B365A'])
    return f_df_cols

def data_transforming(dataframe : pd.DataFrame):
    """
    Transforming the data, using dtypes we can observe that the columns Div, Date, Time, HomeTeam, AwayTeam, FTR, HTR, are objects
    """
    dataframe['Div'] = dataframe['Div'].astype('category')
    dataframe['Date'] = pd.to_datetime(dataframe['Date'], dayfirst=True, errors='coerce')
    dataframe['FTR'] = dataframe['FTR'].astype('category')
    dataframe['HTR'] = dataframe['HTR'].astype('category')

    return dataframe.drop_duplicates()

def read_primera():
    path = os.path.join(base_dir, '..', 'data', 'Primera')
    path = os.path.normpath(path)
    dataframes = []

    for filename in os.listdir(path):
        if filename.endswith('.csv'):
            file_path = os.path.join(path, filename)
            df = pd.read_csv(file_path)

            dataframes.append(df)

    return dataframes

def write_csv(dataframe : pd.DataFrame):
    processed_folder = os.path.join(base_dir, '..', 'data', 'processed')
    os.makedirs(processed_folder, exist_ok=True)
    output_file = os.path.join(processed_folder, 'LaLiga_combined.csv')
    dataframe.to_csv(output_file, index=False)

def mapping_team_names(data_transformed : pd.DataFrame):
    team_name_map = {
        'Alaves' : 'Alavés',
        'Ath Bilbao' : 'Athletic',
        'Valencia' : 'Valencia',
        'Ath Madrid' : 'Atlético',
        'Cadiz' : 'Cádiz',
        'Celta' : 'Celta',
        'Espanol' : 'Espanyol',
        'Mallorca' : 'Mallorca',
        'Osasuna' : 'Osasuna',
        'Sevilla' : 'Sevilla',
        'Betis' : 'Real Betis',
        'La Coruna' : 'RC Deportivo',
        'Real Madrid' : 'Real Madrid',
        'Barcelona' : 'Barcelona',
        'Getafe' : 'Getafe',
        'Malaga' : 'Málaga',
        'Santander' : 'Racing',
        'Sociedad' : 'R. Sociedad',
        'Villarreal' : 'Villarreal',
        'Zaragoza' : 'Real Zaragoza',
        'Recreativo' : 'Recreativo',
        'Gimnastic' : 'Gimnàstic Tarragona',
        'Levante' : 'Levante',
        'Murcia' : 'Real Murcia',
        'Almeria' : 'Almería',
        'Valladolid' : 'Real Valladolid',
        'Numancia' : 'Numancia',
        'Sp Gijon' : 'Real Sporting',
        'Tenerife' : 'Tenerife',
        'Xerez' : 'Xerez CD',
        'Hercules' : 'Hércules',
        'Granada' : 'Granada',
        'Vallecano' : 'Rayo Vallecano',
        'Elche' : 'Elche',
        'Cordoba' : 'Córdoba',
        'Las Palmas' : 'Las Palmas',
        'Leganes' : 'Leganés',
        'Girona' : 'Girona',
        'Huesca' : 'Huesca',
        'Eibar' : 'Eibar'
    }

    data_transformed['HomeTeam'] = data_transformed['HomeTeam'].map(team_name_map).fillna(data_transformed['HomeTeam'])
    data_transformed['AwayTeam'] = data_transformed['AwayTeam'].map(team_name_map).fillna(data_transformed['AwayTeam'])

    return data_transformed

if __name__ == "__main__":
    dataframes_primera = read_primera()
    dataframes_concat = data_concat_and_selection(dataframes=dataframes_primera)
    data_transformed = data_transforming(dataframes_concat)
    data_transformed = mapping_team_names(data_transformed)
    write_csv(data_transformed)
