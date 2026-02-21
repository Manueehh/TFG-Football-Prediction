import csv
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
def main():
    temporada = input("Introduce la temporada (por ejemplo '22-23' o '2022-2023'): ").strip()
    archivo_salida = f"LaLiga_lineups_{temporada}.csv"
    path = os.path.join(base_dir, 'data', 'laliga_lineups', archivo_salida)
    path = os.path.normpath(path)

    # Definimos los nombres de las columnas
    campos = ["MatchNumber", "Date", "HomeTeam", "AwayTeam", "HomeLineup", "AwayLineup"]

    with open(path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=campos)
        writer.writeheader()

        print(f"\nComenzando registro de alineaciones para la temporada {temporada}")
        print("-------------------------------------------------------------")

        for i in range(221, 380):  # 380 partidos
            print(f"\n🕹 Partido {i}/380")

            date = input("Fecha del partido (YYYY-MM-DD): ").strip()
            home_team = input("Equipo local: ").strip()
            away_team = input("Equipo visitante: ").strip()

            print("Introduce la alineación del equipo local (separa por comas):")
            home_lineup = input("> ").strip()

            print("Introduce la alineación del equipo visitante (separa por comas):")
            away_lineup = input("> ").strip()

            writer.writerow({
                "MatchNumber": i,
                "Date": date,
                "HomeTeam": home_team,
                "AwayTeam": away_team,
                "HomeLineup": home_lineup,
                "AwayLineup": away_lineup
            })

            print(f"Partido {i} guardado correctamente.")

        print(f"Todos los partidos registrados. Archivo guardado como '{archivo_salida}'")

if __name__ == "__main__":
    main()
