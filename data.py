# Waiting to do a full data pull until we know what position we want to test based on the draft classes. 
import glob
import os

import pandas as pd
from dotenv import load_dotenv

# Read the .env file into the environment. Does nothing if the file is missing, so secrets on GitHub still work
load_dotenv()

# Stop early with a clear message if the Kaggle token was not found
if not os.getenv("KAGGLE_API_TOKEN"):
    raise RuntimeError("KAGGLE_API_TOKEN not found. Add it to .env or to your GitHub secrets.")

# Import kagglehub after the token is loaded, so it can find the login
import kagglehub

# Competition handle used by kagglehub
COMP = "nfl-big-data-bowl-2027"


# Download one file at a time so the 2.3 GB game tracking files are never pulled up front
def get_file(name):
    folder = kagglehub.competition_download(COMP, path=name)
    if os.path.isfile(folder):
        return folder
    return glob.glob(os.path.join(folder, "**", name), recursive=True)[0]


# Load players and fix types. Int64 lets draft fields stay null for UDFAs
players = pd.read_csv(get_file("players.csv"), parse_dates=["birth_date"])
draft_cols = ["draft_round", "draft_pick_within_round", "draft_overall_pick"]
players[draft_cols] = players[draft_cols].astype("Int64")
players["is_udfa"] = players["draft_overall_pick"].isna()

# Load combine results and drop draft_year, which players already has
combine = pd.read_csv(get_file("combine_results.csv")).drop(columns="draft_year")

# Load career totals, one row per player
career = pd.read_csv(get_file("player_career_successes.csv"))

# Build one player-level table. Left joins keep every player, nulls stay as is
player_table = (
    players
    .merge(combine, on="nfl_id", how="left", validate="1:1")
    .merge(career, on="nfl_id", how="left", validate="1:1")
)

# Load games, since the ids and weeks are already clean
games = pd.read_csv(get_file("games.csv"))

# Load combine tracking and turn the time column into real timestamps
combine_tracking = pd.read_csv(
    get_file("combine_tracking.csv"), parse_dates=["time"]
)

# Add a frame number inside each drill attempt, since the data has none
combine_tracking["frame_id"] = (
    combine_tracking.groupby("event_id")["time"].rank(method="dense").astype(int)
)

# Columns in player_play that are true/false but contain nulls
bool_cols = [
    "in_motion_at_ball_snap", "target", "pressure_allowed", "blitzing",
]

# Columns in player_play with few repeated text values, stored as category to save memory
cat_cols = [
    "team_abbr", "lined_up_position", "possession_team", "yardline_side",
    "play_direction", "offense_formation", "receiver_alignment", "pass_result",
    "play_nullified_by_penalty", "team_coverage_man_zone", "team_coverage_type",
    "route_ran", "coverage_assignment", "coverage_assignment_at_snap",
]

# Load player_play and apply the types. Missing values are kept
player_play = pd.read_csv(get_file("player_play.csv"))
player_play[bool_cols] = player_play[bool_cols].astype("boolean")
player_play[cat_cols] = player_play[cat_cols].astype("category")


# Column types for the game tracking files. Smaller types cut memory use
GAME_TRACKING_TYPES = {
    "game_id": "int64",
    "play_id": "int32",
    "nfl_id": "int32",
    "x": "float32",
    "y": "float32",
    "s": "float32",
    "a": "float32",
    "dis": "float32",
    "o": "float32",
    "dir": "float32",
}


# Find the NFL season from a game_id. Games in January and February belong to the prior season
def season_from_game_id(game_id):
    year = game_id // 1_000_000
    month = (game_id // 10_000) % 100
    return year - 1 if month <= 2 else year


# Pull tracking rows for chosen players and/or games. Reads each file in chunks and keeps only matching rows
def load_game_tracking(nfl_ids=None, game_ids=None, seasons=None, chunk_rows=1_000_000):
    # Pick which season files to open. Use the game_ids if given, otherwise all three
    if seasons is None:
        if game_ids is not None:
            seasons = sorted({season_from_game_id(g) for g in game_ids})
        else:
            seasons = [2023, 2024, 2025]

    # Read each season file in chunks and filter before storing anything
    pieces = []
    for season in seasons:
        file_path = get_file(f"game_tracking_{season}.csv")
        for chunk in pd.read_csv(file_path, dtype=GAME_TRACKING_TYPES, chunksize=chunk_rows):
            if nfl_ids is not None:
                chunk = chunk[chunk["nfl_id"].isin(nfl_ids)]
            if game_ids is not None:
                chunk = chunk[chunk["game_id"].isin(game_ids)]
            pieces.append(chunk)
    tracking = pd.concat(pieces, ignore_index=True)

    # Fix types on the small filtered table. Missing event tags stay null
    tracking["time"] = pd.to_datetime(tracking["time"])
    tracking["event"] = tracking["event"].astype("category")

    # Add a frame number inside each play, since the data has none
    tracking["frame_id"] = (
        tracking.groupby(["game_id", "play_id"])["time"].rank(method="dense").astype(int)
    )

    return tracking.sort_values(
        ["game_id", "play_id", "nfl_id", "frame_id"], ignore_index=True
    )


# Attach play context and game info to tracking rows. Left joins keep every frame
def add_play_context(tracking, player_play, games):
    out = tracking.merge(
        player_play,
        on=["game_id", "play_id", "nfl_id"],
        how="left",
        validate="many_to_one",
    )
    return out.merge(games, on="game_id", how="left", validate="many_to_one")


# Quick check that everything loaded. Only runs when the file is run directly
if __name__ == "__main__":
    for name, df in {
        "player_table": player_table,
        "games": games,
        "combine_tracking": combine_tracking,
        "player_play": player_play,
    }.items():
        print(name, df.shape)
