import pandas as pd 
import os 

# Global variable for directory
base_dir = os.path.dirname(os.path.abspath(__file__))

def load_player_values():
    path = os.path.join(base_dir,'..','data', 'processed', 'transfermarket_values')
    path = os.path.normpath(path)

    player_values = []
    for filename in os.listdir(path):
        if filename.endswith('.csv'):
            season = filename.split('_')[2].replace('.csv','')
            season_splitted = season.split('-')
            season_like_features = season_splitted[0] + '_'+season_splitted[1][-2:]
            file_path = os.path.join(path, filename)
            df = pd.read_csv(file_path)
            df['Season'] = season_like_features

            df = df.filter(['name','market_value','team','Season'])
            player_values.append(df)

    return player_values

def concat_dfs(dataframes : pd.DataFrame):
    f_df = pd.DataFrame()

    for dataframe in dataframes:
        f_df = pd.concat([f_df,dataframe],ignore_index=True)

    return f_df

def write_csv(dataframe : pd.DataFrame, filename : str):
    processed_folder = os.path.join(base_dir, '..', 'data', 'processed')
    os.makedirs(processed_folder, exist_ok=True)
    output_file = os.path.join(processed_folder, filename)
    dataframe.to_csv(output_file, index=False)

dfs = load_player_values()
final_df = concat_dfs(dfs)

write_csv(final_df, 'players_with_market_values.csv')

#NORMALIZAR LUEGO NOMBRES Y YA CALCULAR LOS VALORES DE LAS ALINEACIONES