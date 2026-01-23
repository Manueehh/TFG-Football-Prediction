import time
import random
import os
import json
import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm


base_dir = os.path.dirname(os.path.abspath(__file__))


def scrape_team_squad(team_url):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(team_url, headers=headers)
    if response.status_code != 200:
        print(f"Error accesing {team_url} ({response.status_code})")
        return pd.DataFrame()

    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table", class_="items")
    if not table:
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        for c in comments:
            if "table" in c:
                comment_soup = BeautifulSoup(c, "html.parser")
                table = comment_soup.find("table", class_="items")
                if table:
                    break

    if not table:
        print(f"Couldn't find on {team_url}")
        return pd.DataFrame()

    players_data = []

    def clean_text(text):
        """For formatting the data returned, it has multiple \n and \t"""
        if not text:
            return None
        text = text.replace("\n", " ").replace("\t", " ").replace("\r", " ").replace("\xa0", " ")
        text = " ".join(text.split())
        return text.strip()

    rows = table.find_all("tr", class_=["odd", "even"])
    for row in rows:
        try:
            name_tag = row.find("td", class_="hauptlink")
            if not name_tag:
                continue

            player_name = clean_text(name_tag.text)
            player_link = "https://www.transfermarkt.es" + name_tag.find("a")["href"]

            position_tag = row.find("td", class_="posrela")
            position = clean_text(position_tag.text) if position_tag else None

            cells = row.find_all("td")
            age = clean_text(cells[4].text) if len(cells) > 4 else None

            flag_tag = row.find("img", class_="flaggenrahmen")
            nationality = flag_tag["title"] if flag_tag else None

            value_tag = row.find("td", class_="rechts hauptlink")
            market_value = clean_text(value_tag.text) if value_tag else None

            players_data.append({
                "name": player_name,
                "position": position,
                "age": age,
                "nationality": nationality,
                "market_value": market_value,
                "profile_url": player_link
            })
        except Exception as e:
            print(f"Error processing player: {e}")

    return pd.DataFrame(players_data)


def scrape_la_liga(filename : str, teams : dict):
    path = os.path.join(base_dir,'..','data', 'processed' , 'transfermarket_values', filename)
    path = os.path.normpath(path)

    all_players = []
    for team, url in tqdm(teams.items(), desc="Scraping teams La Liga"):
        df_team = scrape_team_squad(url)
        if not df_team.empty:
            df_team["team"] = team
            all_players.append(df_team)
        else:
            print(f"No players for {team}")
        time.sleep(random.uniform(1, 3))

    if all_players:
        df_all = pd.concat(all_players, ignore_index=True)
        df_all = df_all.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        df_all.to_csv(path, index=False, encoding="utf-8",sep=',', quoting=1)


def scrape_all_seasons():
    seasons_dir = os.path.join(base_dir, "..", "data", "seasons_teams_data")

    season_files = [f for f in os.listdir(seasons_dir) if f.endswith(".json")]
    season_files.sort()

    for season_file in season_files:
        season_path = os.path.join(seasons_dir, season_file)
        with open(season_path, "r", encoding="utf-8") as f:
            teams = json.load(f)

        season_name = os.path.splitext(season_file)[0]
        output_filename = f"LaLiga_transfermarket_{season_name}.csv"
        scrape_la_liga(output_filename, teams)


if __name__ == "__main__":
    df = scrape_all_seasons()
