import pandas as pd
import numpy as np
import json
import glob
import os
from typing import Dict, List, Optional, Tuple, Union

# Define global constants
PRESSURE_LEVEL_ORDER = ["Low", "Medium", "High", "Extreme"]

def load_and_combine_data(data_dir: str) -> pd.DataFrame:
    """
    Load all match JSON files from the specified directory and combine into a single DataFrame.

    Parameters:
    -----------
    data_dir : str
        Directory containing match JSON files

    Returns:
    --------
    pd.DataFrame
        Combined ball-by-ball data from all matches
    """
    match_files = glob.glob(os.path.join(data_dir, "*.json"))

    if not match_files:
        return pd.DataFrame()

    all_dfs = []
    for match_file in match_files:
        df = convert_match_json_to_ball_df(match_file)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)

    # Add batting partnership info to each ball
    df = add_partnership_info(df)

    # Add rolling stats (last 3 overs stats)
    df = add_rolling_stats(df)

    # Calculate run rates and required rates
    df = add_run_rate_metrics(df)

    # Find first innings runs for each match (used for 2nd innings pressure)
    matches = df["Match_File"].unique()
    first_inn_runs_by_match = {}

    for match in matches:
        first_inn = df[(df["Match_File"] == match) & (df["Innings_Index"] == 1)]
        if not first_inn.empty:
            striker_deliveries = first_inn[first_inn["DeliveryType"] == "striker"]
            if not striker_deliveries.empty:
                first_inn_runs_by_match[match] = striker_deliveries["Runs_Total"].sum()

    # Process each innings separately for pressure calculations
    from pressure_calculations import assign_separate_t20_pressure_labels, calculate_dynamic_pressure_for_innings

    out_chunks = []
    for (mf, idx), chunk in df.groupby(["Match_File", "Innings_Index"]):
        chunk = chunk.copy()
        first_inn_runs = first_inn_runs_by_match.get(mf, None)

        # Check if this is a T20 match
        if "Match_Type" in chunk.columns and chunk["Match_Type"].iloc[0] == "T20":
            # Use the new separate pressure labels for T20
            if idx == 2:
                chunk = assign_separate_t20_pressure_labels(chunk, first_innings_total=first_inn_runs)
            else:
                chunk = assign_separate_t20_pressure_labels(chunk, first_innings_total=None)
        else:
            # For non-T20 formats, keep using the old method if needed
            if idx == 2:
                chunk = calculate_dynamic_pressure_for_innings(chunk, first_innings_total=first_inn_runs)
            else:
                chunk = calculate_dynamic_pressure_for_innings(chunk, first_innings_total=None)

        out_chunks.append(chunk)

    return pd.concat(out_chunks, ignore_index=True)

def is_legal_delivery(delivery: Dict) -> bool:
    """
    Check if a delivery is legal (not a wide or no-ball).

    Parameters:
    -----------
    delivery : Dict
        Delivery data from match JSON

    Returns:
    --------
    bool
        True if the delivery is legal, False otherwise
    """
    extras = delivery.get("extras", {})
    return not ("wides" in extras or "noballs" in extras)

def convert_match_json_to_ball_df(file_path: str) -> pd.DataFrame:
    """
    Convert a match JSON file to a ball-by-ball DataFrame.

    Parameters:
    -----------
    file_path : str
        Path to the match JSON file

    Returns:
    --------
    pd.DataFrame
        Ball-by-ball data for the match
    """
    with open(file_path, "r") as f:
        match_data = json.load(f)

    # Get team batting orders from the JSON players list
    players_dict = match_data["info"].get("players", {})

    teams = match_data["info"].get("teams", [])
    match_type = match_data["info"].get("match_type", "T20")
    season = match_data["info"].get("season", "Unknown")
    # Extract match_id from filename (if needed)
    match_id = os.path.basename(file_path).split('.')[0]

    innings_list = match_data.get("innings", [])
    if not innings_list:
        return pd.DataFrame()

    # Calculate first-innings total runs (for chase context)
    first_inn_runs = 0
    if len(innings_list) >= 1:
        for over_dict in innings_list[0].get("overs", []):
            for delivery in over_dict.get("deliveries", []):
                first_inn_runs += delivery.get("runs", {}).get("total", 0)

    ball_data = []
    # Loop over innings
    for inn_idx, inn_data in enumerate(innings_list, start=1):
        batting_team = inn_data.get("team")
        if len(teams) == 2 and batting_team in teams:
            bowling_team = [t for t in teams if t != batting_team][0]
        else:
            bowling_team = None

        # Get the batting order for the team from the players list (if available)
        batting_order = players_dict.get(batting_team, [])

        # Initialize trackers
        wickets_fallen = 0
        current_partnership = 0
        balls_since_boundary = 0
        wickets_in_last_3_overs = 0
        last_3_overs_balls = 0
        total_runs = 0
        legal_balls = 0
        prev_ball_result = None
        bowler_current_wickets = {}
        bowler_current_overs = {}
        batsman_current_runs = {}

        overs_list = inn_data.get("overs", [])
        for over_dict in overs_list:
            over_num = over_dict.get("over", 0)
            deliveries = over_dict.get("deliveries", [])
            # Reset "last 3 overs" counter in final overs
            if match_type == "T20" and over_num >= 17:
                last_3_overs_balls = 0
            elif match_type != "T20" and over_num >= 47:
                last_3_overs_balls = 0

            for ball_index, delivery in enumerate(deliveries):
                runs_total = delivery.get("runs", {}).get("total", 0)
                total_runs += runs_total
                batter_runs = delivery.get("runs", {}).get("batter", 0)

                # Check wicket info
                wickets_info = delivery.get("wickets", [])
                wicket_fell = bool(wickets_info)

                extras = delivery.get("extras", {})
                # A delivery is legal if it is not a wide or a no-ball
                is_legal = not ("wides" in extras or "noballs" in extras)
                if is_legal:
                    legal_balls += 1

                # A ball faced by the striker counts only if it is not a wide
                balls_faced = 1 if "wides" not in extras else 0

                # Update last-3-overs counters (if applicable)
                if (match_type == "T20" and over_num >= 17) or (match_type != "T20" and over_num >= 47):
                    if is_legal:
                        last_3_overs_balls += 1
                    if wicket_fell:
                        wickets_in_last_3_overs += 1

                # Reset balls-since-boundary on a boundary
                if batter_runs in [4, 6]:
                    balls_since_boundary = 0
                else:
                    if is_legal:
                        balls_since_boundary += 1

                # Update partnership and wickets tally
                if wicket_fell:
                    wickets_fallen += 1
                    current_partnership = 0
                else:
                    current_partnership += batter_runs

                # Process bowler info
                bowler = delivery.get("bowler")
                if bowler:
                    if bowler not in bowler_current_wickets:
                        bowler_current_wickets[bowler] = 0
                        bowler_current_overs[bowler] = 0
                    if is_legal:
                        bowler_current_overs[bowler] += 1
                    if wicket_fell:
                        # For each wicket event, only count if the dismissal is attributable to the bowler.
                        for w in wickets_info:
                            dis_kind = w.get("kind", "").lower()
                            if dis_kind not in ["run out", "obstructing the field", "retired hurt"]:
                                bowler_current_wickets[bowler] += 1

                # Update batter runs
                batter = delivery.get("batter")
                non_striker = delivery.get("non_striker")
                if batter not in batsman_current_runs:
                    batsman_current_runs[batter] = 0
                batsman_current_runs[batter] += batter_runs

                overs_completed = legal_balls / 6.0
                current_run_rate = total_runs / overs_completed if overs_completed > 0 else 0
                required_run_rate = None
                if inn_idx == 2:
                    runs_needed = first_inn_runs - total_runs + 1
                    balls_left = (20 if match_type == "T20" else 50) * 6 - legal_balls
                    required_run_rate = runs_needed / (balls_left / 6.0) if balls_left > 0 else 0

                if ball_index > 0 or over_num > 0:
                    if wicket_fell:
                        this_ball_result = "Wicket"
                    elif batter_runs == 0 and not extras:
                        this_ball_result = "Dot"
                    elif batter_runs in [4, 6]:
                        this_ball_result = "Boundary"
                    elif extras:
                        this_ball_result = "Extra"
                    else:
                        this_ball_result = str(batter_runs)
                else:
                    this_ball_result = None

                penalty_extras = extras.get("wides", 0) + extras.get("noballs", 0)
                bowler_runs = batter_runs + penalty_extras

                # Determine dismissal details for the striker
                striker_dismissed = any(w.get("player_out") == batter for w in wickets_info)
                mode_striker = None
                if striker_dismissed:
                    for w in wickets_info:
                        if w.get("player_out") == batter:
                            mode_striker = w.get("kind", "Unknown").lower()
                            break
                # Only credit the bowler with a wicket if the dismissal is of the striker and is attributable
                striker_wicket = 1 if (striker_dismissed and mode_striker not in ["run out", "obstructing the field",
                                                                                  "retired hurt"]) else 0

                # Infer batting position using the team's batting order
                try:
                    batting_position = batting_order.index(batter) + 1  # positions are 1-indexed
                except ValueError:
                    batting_position = 99

                # Build the striker row (this row represents the delivery for the batter facing the ball)
                row_striker = {
                    "Match_File": os.path.basename(file_path),
                    "Season": season,
                    "Innings_Index": inn_idx,
                    "Batting_Team": batting_team,
                    "Bowling_Team": bowling_team,
                    "Batter": batter,
                    "Non_Striker": non_striker,
                    "Bowler": bowler,
                    "Over": over_num,
                    "Ball_In_Over": ball_index + 1,
                    "Runs_Total": runs_total,
                    "Runs_Batter": batter_runs,
                    "Bowler_Runs": bowler_runs,
                    "Match_Type": match_type,
                    "IsLegalDelivery": 1 if is_legal else 0,
                    "IsBattingDelivery": balls_faced,
                    "Current_Score": f"{total_runs}-{wickets_fallen}",
                    "Wickets_Fallen": wickets_fallen,
                    "Current_Run_Rate": current_run_rate,
                    "Required_Run_Rate": required_run_rate,
                    "Current_Partnership": current_partnership,
                    "Balls_Since_Boundary": balls_since_boundary,
                    "Wickets_In_Last_3_Overs": wickets_in_last_3_overs,
                    "Previous_Ball_Result": prev_ball_result,
                    "Bowler_Current_Wickets": bowler_current_wickets.get(bowler, 0) if bowler else 0,
                    "Bowler_Current_Overs": bowler_current_overs.get(bowler, 0) / 6.0 if bowler else 0,
                    "Strike_Batsman_Runs": batsman_current_runs[batter],
                    "Batting_Task": "Setting" if inn_idx == 1 else "Chasing",
                    "Bowling_Task": "Bowling First" if inn_idx == 1 else "Defending",
                    "Balls_Faced": balls_faced,
                    "Wicket": striker_wicket,
                    "BatterDismissed": 1 if striker_dismissed else 0,
                    "Mode_Of_Dismissal": mode_striker,
                    "PlayerDismissed": batter if striker_dismissed else None,
                    "DeliveryType": "striker",
                    "Batting_Position": batting_position
                }

                # Build the non-striker row (to capture run-out details etc.)
                row_non_striker = row_striker.copy()
                row_non_striker["Batter"] = non_striker
                row_non_striker["Balls_Faced"] = 0  # non-striker does not face the ball
                row_non_striker["Runs_Batter"] = 0
                row_non_striker["IsBattingDelivery"] = 0
                row_non_striker["Strike_Batsman_Runs"] = 0
                # For non-striker, do not credit any wicket to the bowler.
                row_non_striker["Wicket"] = 0
                row_non_striker["BatterDismissed"] = 1 if any(
                    w.get("player_out") == non_striker for w in wickets_info) else 0
                mode_non_striker = None
                if any(w.get("player_out") == non_striker for w in wickets_info):
                    for w in wickets_info:
                        if w.get("player_out") == non_striker:
                            mode_non_striker = w.get("kind", "Unknown").lower()
                            break
                row_non_striker["Mode_Of_Dismissal"] = mode_non_striker
                row_non_striker["PlayerDismissed"] = non_striker if row_non_striker["BatterDismissed"] == 1 else None
                row_non_striker["DeliveryType"] = "non-striker"

                ball_data.append(row_striker)
                ball_data.append(row_non_striker)
                prev_ball_result = this_ball_result

    df = pd.DataFrame(ball_data)
    if df.empty:
        return df

    # Label game phase based on over number
    def game_phase_t20(over):
        if over < 6:
            return "Powerplay"
        elif over < 16:
            return "Middle"
        else:
            return "Death"

    def game_phase_50(over):
        if over < 10:
            return "Powerplay"
        elif over < 40:
            return "Middle"
        else:
            return "Death"

    if match_type == "T20":
        df["Game_Phase"] = df["Over"].apply(game_phase_t20)
    else:
        df["Game_Phase"] = df["Over"].apply(game_phase_50)
    phase_map = {"Powerplay": "Low", "Middle": "Medium", "Death": "High"}
    df["Pressure"] = df["Game_Phase"].map(phase_map)

    # Add dynamic pressure to each innings group
    from pressure_calculations import calculate_dynamic_pressure_for_innings

    out_chunks = []
    for (mf, idx), chunk in df.groupby(["Match_File", "Innings_Index"]):
        chunk = chunk.copy()
        if idx == 2:
            chunk = calculate_dynamic_pressure_for_innings(chunk, first_innings_total=first_inn_runs)
        else:
            chunk = calculate_dynamic_pressure_for_innings(chunk, first_innings_total=None)
        out_chunks.append(chunk)
    return pd.concat(out_chunks, ignore_index=True)

def add_partnership_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add batting partnership information to each ball.
    Tracks current partnership runs and balls for each delivery.

    Parameters:
    -----------
    df : pd.DataFrame
        Ball-by-ball data

    Returns:
    --------
    pd.DataFrame
        DataFrame with added partnership information
    """
    if df.empty:
        return df

    # Let's first check the actual column names to be safe
    batter_column = "Batter" if "Batter" in df.columns else "striker" if "striker" in df.columns else None
    non_striker_column = "Non_Striker" if "Non_Striker" in df.columns else "non_striker" if "non_striker" in df.columns else None

    # If we can't find the batter columns, return the dataframe unchanged
    if batter_column is None or non_striker_column is None:
        df["Current_Partnership"] = 0
        return df

    result_chunks = []

    # Process each innings separately
    for (match_file, innings_idx), innings_df in df.groupby(["Match_File", "Innings_Index"]):
        innings_df = innings_df.copy()

        # Initialize partnership tracking
        partnership_runs = 0
        partnership_balls = 0
        current_batters = set()
        partnerships = []

        # Process deliveries in order
        for i, row in innings_df.iterrows():
            # Track current batters
            if pd.notna(row[batter_column]):
                current_batters.add(row[batter_column])
            if pd.notna(row[non_striker_column]):
                current_batters.add(row[non_striker_column])

            # Add runs to partnership
            if row["DeliveryType"] == "striker":
                partnership_runs += row["Runs_Total"]
                if row["IsBattingDelivery"] == 1:
                    partnership_balls += 1

            # Reset partnership if a wicket falls (and it's not a run out of non-striker)
            if row["Wicket"] == 1 and row["BatterDismissed"] == 1:
                partnership_runs = 0
                partnership_balls = 0
                if pd.notna(row.get("PlayerDismissed")) and row["PlayerDismissed"] in current_batters:
                    current_batters.remove(row["PlayerDismissed"])

            partnerships.append(partnership_runs)

        innings_df["Current_Partnership"] = partnerships
        result_chunks.append(innings_df)

    if result_chunks:
        return pd.concat(result_chunks, ignore_index=True)
    else:
        df["Current_Partnership"] = 0
        return df

def add_rolling_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling statistics to each ball, including:
    - Wickets_In_Last_3_Overs: Number of wickets fallen in the previous 3 overs
    - Balls_Since_Boundary: Number of balls since last boundary

    Parameters:
    -----------
    df : pd.DataFrame
        Ball-by-ball data

    Returns:
    --------
    pd.DataFrame
        DataFrame with added rolling statistics
    """
    if df.empty:
        return df

    result_chunks = []

    for (match_file, innings_idx), innings_df in df.groupby(["Match_File", "Innings_Index"]):
        innings_df = innings_df.copy()

        # Initialize tracking variables
        wickets_last_3_overs = []
        balls_since_boundary = []
        recent_wickets = []
        since_boundary = 0

        # Process each delivery
        for i, row in innings_df.iterrows():
            # Track boundaries
            if row["DeliveryType"] == "striker":
                if row["Runs_Batter"] == 4 or row["Runs_Batter"] == 6:
                    since_boundary = 0
                else:
                    since_boundary += 1

            # Track wickets
            if row["Wicket"] == 1:
                recent_wickets.append(row["Over"])

            # Remove wickets older than 3 overs
            current_over = row["Over"]
            recent_wickets = [over for over in recent_wickets if (current_over - over) <= 3]

            wickets_last_3_overs.append(len(recent_wickets))
            balls_since_boundary.append(since_boundary)

        innings_df["Wickets_In_Last_3_Overs"] = wickets_last_3_overs
        innings_df["Balls_Since_Boundary"] = balls_since_boundary

        result_chunks.append(innings_df)

    return pd.concat(result_chunks, ignore_index=True)

def add_run_rate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate run rate metrics for each ball:
    - Current_Run_Rate: Runs per over at current point
    - Required_Run_Rate: Run rate needed to win (2nd innings)

    Parameters:
    -----------
    df : pd.DataFrame
        Ball-by-ball data

    Returns:
    --------
    pd.DataFrame
        DataFrame with added run rate metrics
    """
    if df.empty:
        return df

    result_chunks = []

    for (match_file, innings_idx), innings_df in df.groupby(["Match_File", "Innings_Index"]):
        innings_df = innings_df.copy()

        # Calculate current run rate
        runs_cumulative = 0
        legal_balls = 0
        current_rr = []
        required_rr = []

        # Get first innings total (if this is second innings)
        first_innings_total = None
        if innings_idx == 2:
            first_inn = df[(df["Match_File"] == match_file) & (df["Innings_Index"] == 1)]
            if not first_inn.empty:
                striker_deliveries = first_inn[first_inn["DeliveryType"] == "striker"]
                if not striker_deliveries.empty:
                    first_innings_total = striker_deliveries["Runs_Total"].sum()

        # Calculate for each ball
        for i, row in innings_df.iterrows():
            if row["DeliveryType"] == "striker":
                runs_cumulative += row["Runs_Total"]

            if row["IsLegalDelivery"] == 1:
                legal_balls += 1

            overs_completed = legal_balls / 6.0

            # Calculate current run rate
            if overs_completed > 0:
                crr = runs_cumulative / overs_completed
            else:
                crr = 0
            current_rr.append(crr)

            # Calculate required run rate (2nd innings only)
            if innings_idx == 2 and first_innings_total is not None:
                balls_remaining = max(0, 120 - legal_balls)  # Assuming T20
                if balls_remaining > 0:
                    runs_needed = first_innings_total + 1 - runs_cumulative
                    rrr = (runs_needed / (balls_remaining / 6.0)) if runs_needed > 0 else 0
                else:
                    rrr = 0
                required_rr.append(rrr)
            else:
                required_rr.append(None)

        innings_df["Current_Run_Rate"] = current_rr
        innings_df["Required_Run_Rate"] = required_rr
        innings_df["Wickets_Fallen"] = innings_df["Wicket"].cumsum()

        result_chunks.append(innings_df)

    return pd.concat(result_chunks, ignore_index=True)

def get_match_metadata(file_path: str) -> Dict:
    """
    Extract match metadata from a cricket match JSON file

    Parameters:
    -----------
    file_path : str
        Path to the match JSON file

    Returns:
    --------
    Dict
        Dictionary containing match metadata
    """
    with open(file_path, "r") as f:
        match_data = json.load(f)

    info = match_data.get("info", {})
    match_date = info.get("dates", ["Unknown"])[0] if "dates" in info else "Unknown"
    venue = info.get("venue", "Unknown Venue")
    teams = info.get("teams", [])
    city = info.get("city", "Unknown City")
    match_type = info.get("match_type", "Unknown Format")

    return {
        "date": match_date,
        "venue": venue,
        "teams": teams,
        "city": city,
        "match_type": match_type
    }