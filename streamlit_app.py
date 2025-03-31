import streamlit as st
import pandas as pd
import numpy as np
import json
import glob
import os
import plotly.express as px
import plotly.graph_objects as go

# Define global constants
PRESSURE_LEVEL_ORDER = ["Low", "Medium", "High", "Extreme"]


# Helper function to order pressure levels consistently
def order_pressure_levels(df, pressure_col="DynamicPressureLabel"):
    """Apply consistent ordering to pressure level categories"""
    # Check if the specified pressure column exists
    if pressure_col not in df.columns:
        # Try to find an alternative pressure column
        alternatives = ["BattingPressureLabel", "BowlingPressureLabel", "DynamicPressureLabel"]
        for alt_col in alternatives:
            if alt_col in df.columns:
                pressure_col = alt_col
                break
        else:
            # If no pressure column exists, return the DataFrame unchanged
            return df

    # Apply categorical ordering
    df[pressure_col] = pd.Categorical(
        df[pressure_col],
        categories=PRESSURE_LEVEL_ORDER,
        ordered=True
    )
    return df


############################################
# 0) OPTIONAL: Custom CSS for Theming
############################################
def inject_custom_css():
    """
    Inject a custom CSS style to create a modern, professional dashboard aesthetic.
    """
    st.markdown("""
    <style>
    /* Overall background and base theme */
    .main {
        background-color: #0E1117; /* darker background for contrast */
        padding: 1rem;
    }
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #1E2130;
        border-right: 1px solid #2D3250;
        padding-top: 2rem;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1rem;
    }

    /* Card-like containers for sections */
    div.stBlock, div[data-testid="stVerticalBlock"] > div:has(div.stBlock) {
        background-color: #1E2130;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid #2D3250;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* Font and text styling */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
        color: #E6E6E6;
    }

    /* Heading styling with accent underlines */
    h1 {
        color: #FFFFFF;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 1.5rem;
        border-bottom: 4px solid #4F8BFF;
        padding-bottom: 0.5rem;
        width: fit-content;
    }

    h2 {
        color: #FFFFFF;
        font-weight: 600;
        font-size: 1.8rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #4F8BFF;
        padding-bottom: 0.3rem;
        width: fit-content;
    }

    h3 {
        color: #FFFFFF;
        font-weight: 600;
        font-size: 1.4rem;
        margin-bottom: 0.8rem;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1E2130;
        border-radius: 8px;
        padding: 0px 10px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        border-radius: 8px;
        color: #FFFFFF;
        background-color: #272D3F;
        border: none;
        padding: 0px 16px;
        font-weight: 500;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: #3A4566;
        color: #FFFFFF;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #4F8BFF;
        color: #FFFFFF;
    }

    /* Button styling */
    .stButton button {
        border-radius: 6px;
        background-color: #4F8BFF;
        color: white;
        border: none;
        padding: 0.4rem 1rem;
        font-weight: 500;
    }

    .stButton button:hover {
        background-color: #3A6ED0;
    }

    /* Multiselect and selectbox styling */
    div[data-baseweb="select"] {
        border-radius: 6px;
    }

    div[data-baseweb="select"] > div {
        background-color: #272D3F;
        border-color: #3A4566;
    }

    /* Slider styling */
    div[data-testid="stThumbValue"] {
        background-color: #4F8BFF !important;
        color: white;
    }

    /* DataFrame styling */
    .stDataFrame {
        border: 1px solid #2D3250;
        border-radius: 8px;
        overflow: hidden;
    }

    .stDataFrame table {
        border-collapse: collapse;
    }

    .stDataFrame th {
        background-color: #272D3F;
        color: white;
        padding: 10px 15px;
        border-bottom: 2px solid #3A4566;
    }

    .stDataFrame td {
        padding: 8px 15px;
        border-bottom: 1px solid #2D3250;
        color: #E6E6E6;
    }

    .stDataFrame tr:hover td {
        background-color: #272D3F;
    }

    /* Caption and helper text */
    .helper-text {
        color: #B0B0B0;
        font-size: 0.85rem;
        margin-bottom: 1rem;
        padding: 0.6rem 1rem;
        background-color: #272D3F;
        border-radius: 6px;
        border-left: 3px solid #4F8BFF;
    }

    /* Metric cards */
    .metric-card {
        background-color: #272D3F;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        border: 1px solid #3A4566;
        margin-bottom: 1rem;
    }

    .metric-title {
        font-size: 1rem;
        font-weight: 600;
        color: #B0B0B0;
        margin-bottom: 0.5rem;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #FFFFFF;
    }

    /* Filter section */
    .filter-section {
        background-color: #1E2130;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1.5rem;
        border: 1px solid #2D3250;
    }

    /* Banner for key insights */
    .insight-banner {
        background-color: #2C3654;
        border-left: 5px solid #4F8BFF;
        border-radius: 6px;
        padding: 1rem;
        margin: 1rem 0;
    }

    /* Footer styling */
    .footer {
        text-align: center;
        padding: 1.5rem 0;
        color: #B0B0B0;
        font-size: 0.8rem;
        margin-top: 2rem;
        border-top: 1px solid #2D3250;
    }
    </style>
    """, unsafe_allow_html=True)


############################################
# 1) LOAD & COMBINE JSON FILES
############################################
def load_and_combine_data(data_dir):
    """
    Load all match JSON files from the specified directory and combine into a single DataFrame.
    """
    match_files = glob.glob(os.path.join(data_dir, "*.json"))

    if not match_files:
        st.error(f"No JSON files found in {data_dir}. Please check the directory path.")
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

    # IMPORTANT CHANGE: Replace this section to use the new pressure labels calculation
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


############################################
# 2) CALCULATE DYNAMIC PRESSURE (1ST + 2ND INNINGS)
############################################
def calculate_dynamic_pressure_for_innings(df_innings, first_innings_total=None):
    if df_innings.empty:
        df_innings["DynamicPressureScore"] = np.nan
        df_innings["DynamicPressureLabel"] = None
        return df_innings
    if df_innings["Match_Type"].iloc[0] != "T20":
        df_innings["DynamicPressureScore"] = np.nan
        df_innings["DynamicPressureLabel"] = None
        return df_innings

    # Constants - consider moving these to a config dictionary
    alpha = 1.2
    beta = 0.8
    gamma = 0.5
    delta = 0.3
    TYPICAL_RUN_RATE = 8.0

    runs_cumulative = 0
    wickets_cumulative = 0
    legal_balls_count = 0
    inn_idx = df_innings["Innings_Index"].iloc[0]
    scores = []

    for i, row in df_innings.iterrows():
        runs_cumulative += row["Runs_Total"]
        if row["Wicket"]:
            wickets_cumulative += 1
        if row["IsLegalDelivery"] == 1:
            legal_balls_count += 1

        wickets_in_hand = 10 - wickets_cumulative
        overs_used = legal_balls_count / 6.0

        # Improved time factor with powerplay consideration
        is_powerplay = overs_used < 6.0
        end_overs = overs_used >= 16.0

        # W-shaped time factor: high in powerplay, dips in middle, rises in death overs
        if is_powerplay:
            # Powerplay overs have elevated pressure (1.2-1.3)
            time_factor = 1.2 + (overs_used / 30.0)
        elif end_overs:
            # Death overs have highest pressure (1.4+)
            time_factor = 1.4 + ((overs_used - 16.0) / 10.0)
        else:
            # Middle overs have lower but gradually increasing pressure
            time_factor = 1.0 + (overs_used / 40.0)

        # Add explicit powerplay pressure for bowlers
        powerplay_bowling_factor = 0.3 if is_powerplay else 0.0

        ball_drought_factor = min(row["Balls_Since_Boundary"] / 12.0, 1.0)
        wicket_pressure = min(row["Wickets_In_Last_3_Overs"] / 3.0, 1.0)

        partnership_factor = 1.0
        if row["Current_Partnership"] < 20 and wickets_cumulative > 0:
            partnership_factor = 1.2

        if inn_idx == 1:
            expected_runs_so_far = TYPICAL_RUN_RATE * overs_used
            par_deficit = max(expected_runs_so_far - runs_cumulative, 0)

            if wickets_in_hand <= 0 or overs_used >= 20:
                pressure_score = 0
            else:
                part1 = alpha * (par_deficit / 35.0)
                part2 = beta * (1.0 - (wickets_in_hand / 10.0))
                part3 = gamma * wicket_pressure
                part4 = delta * ball_drought_factor

                # Add powerplay pressure component for first innings
                base_score = (part1 + part2 + part3 + part4 + powerplay_bowling_factor) * partnership_factor
                pressure_score = base_score * time_factor
        else:
            if not first_innings_total:
                pressure_score = np.nan
            else:
                runs_needed = first_innings_total - runs_cumulative + 1
                balls_left = 120 - legal_balls_count

                if runs_needed <= 0 or balls_left <= 0 or wickets_in_hand <= 0:
                    pressure_score = 0
                else:
                    req_run_rate = runs_needed / (balls_left / 6.0)
                    rrr_factor = req_run_rate / TYPICAL_RUN_RATE
                    part1 = alpha * min(rrr_factor, 2.0)
                    part2 = beta * (1.0 - (wickets_in_hand / 10.0))
                    part3 = gamma * wicket_pressure
                    part4 = delta * ball_drought_factor

                    chase_factor = 1.0
                    if balls_left < 36 and runs_needed > 30:
                        chase_factor = 1.3

                    # Add powerplay pressure component for second innings
                    base_score = (
                                             part1 + part2 + part3 + part4 + powerplay_bowling_factor) * partnership_factor * chase_factor
                    pressure_score = base_score * time_factor

        scores.append(pressure_score)

    df_innings["DynamicPressureScore"] = scores

    def label_pressure(val):
        if pd.isna(val):
            return None
        if val < 1.0:
            return "Low"
        elif val < 2.0:
            return "Medium"
        elif val < 3.0:
            return "High"
        else:
            return "Extreme"

    df_innings["DynamicPressureLabel"] = df_innings["DynamicPressureScore"].apply(label_pressure)
    return df_innings


############################################
# 3) CONVERT MATCH JSON => BALL DF
############################################
def is_legal_delivery(delivery):
    extras = delivery.get("extras", {})
    return not ("wides" in extras or "noballs" in extras)


def convert_match_json_to_ball_df(file_path):
    import json, os
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
    out_chunks = []
    for (mf, idx), chunk in df.groupby(["Match_File", "Innings_Index"]):
        chunk = chunk.copy()
        if idx == 2:
            chunk = calculate_dynamic_pressure_for_innings(chunk, first_innings_total=first_inn_runs)
        else:
            chunk = calculate_dynamic_pressure_for_innings(chunk, first_innings_total=None)
        out_chunks.append(chunk)
    return pd.concat(out_chunks, ignore_index=True)


############################################
# 4) GAMES PLAYED HELPERS
############################################
def compute_batting_games_played(df):
    sub = df.dropna(subset=["Batter"])[["Batter", "Match_File"]].drop_duplicates()
    out = sub.groupby("Batter")["Match_File"].nunique().reset_index()
    out.rename(columns={"Match_File": "GamesPlayed"}, inplace=True)
    return out


def compute_bowling_games_played(df):
    sub = df.dropna(subset=["Bowler"])[["Bowler", "Match_File"]].drop_duplicates()
    out = sub.groupby("Bowler")["Match_File"].nunique().reset_index()
    out.rename(columns={"Match_File": "GamesPlayed"}, inplace=True)
    return out


############################################
# 5) METRICS BY PRESSURE
############################################
def compute_batting_metrics_by_pressure(df, pressure_col="BattingPressureLabel", aggregate=False):
    """
    Calculate batting performance metrics grouped by pressure level.
    
    Args:
        df: The data frame containing ball-by-ball data
        pressure_col: Which pressure column to use ("BattingPressureLabel" or "DynamicPressureLabel")
        aggregate: If True, return metrics aggregated by pressure level only. 
                   If False (default), return metrics by batter and pressure level.
    
    Returns:
        DataFrame with batting metrics, either by batter+pressure or just pressure level
    """
    if df.empty or pressure_col not in df.columns:
        return pd.DataFrame()
        
    # Filter to only striker deliveries for batting analysis
    striker_df = df[df["DeliveryType"] == "striker"].copy()
    
    # Calculate dot balls
    striker_df["DotBall"] = ((striker_df["Runs_Batter"] == 0) & 
                            (striker_df["Wicket"] == 0) & 
                            (striker_df["IsBattingDelivery"] == 1)).astype(int)
    
    # Group by pressure and batter
    grouped = striker_df.groupby([pressure_col, "Batter"]).agg(
        Runs=("Runs_Batter", "sum"),
        Balls=pd.NamedAgg(column="IsBattingDelivery", aggfunc=lambda x: (x == 1).sum()),
        Dismissals=pd.NamedAgg(column="Wicket", aggfunc=lambda x: (x == 1).sum()),
        Dots=("DotBall", "sum")
    ).reset_index()
    
    # Calculate Strike Rate and Dot Ball Percentage
    grouped["StrikeRate"] = grouped.apply(lambda x: (x["Runs"] / x["Balls"]) * 100 if x["Balls"] > 0 else 0, axis=1)
    grouped["DotBallPct"] = grouped.apply(lambda x: x["Dots"] / x["Balls"] if x["Balls"] > 0 else 0, axis=1)
    
    # If we don't need to aggregate further, return batter-level data
    if not aggregate:
        return grouped
    
    # Otherwise, aggregate across batters to get metrics by pressure level
    pressure_metrics = grouped.groupby(pressure_col).agg(
        TotalRuns=("Runs", "sum"),
        TotalBalls=("Balls", "sum"),
        TotalDismissals=("Dismissals", "sum"),
        TotalDots=("Dots", "sum")
    ).reset_index()
    
    # Calculate overall strike rate and dot ball percentage by pressure
    pressure_metrics["StrikeRate"] = pressure_metrics.apply(
        lambda x: (x["TotalRuns"] / x["TotalBalls"]) * 100 if x["TotalBalls"] > 0 else 0, axis=1
    )
    pressure_metrics["DotBallPct"] = pressure_metrics.apply(
        lambda x: x["TotalDots"] / x["TotalBalls"] if x["TotalBalls"] > 0 else 0, axis=1
    )
    
    # Order by pressure level
    pressure_metrics[pressure_col] = pd.Categorical(
        pressure_metrics[pressure_col], 
        categories=PRESSURE_LEVEL_ORDER, 
        ordered=True
    )
    pressure_metrics = pressure_metrics.sort_values(pressure_col)
    
    # Rename columns for display
    pressure_metrics = pressure_metrics.rename(columns={pressure_col: "Pressure"})
    
    return pressure_metrics

def compute_bowling_metrics_by_pressure(df, pressure_col="BowlingPressureLabel", aggregate=False):
    """
    Calculate bowling performance metrics grouped by pressure level.
    
    Args:
        df: The data frame containing ball-by-ball data
        pressure_col: Which pressure column to use ("BowlingPressureLabel" or "DynamicPressureLabel")
        aggregate: If True, return metrics aggregated by pressure level only. 
                   If False (default), return metrics by bowler and pressure level.
    
    Returns:
        DataFrame with bowling metrics, either by bowler+pressure or just pressure level
    """
    if df.empty or pressure_col not in df.columns:
        return pd.DataFrame()
        
    # Filter to only striker deliveries for bowling analysis
    striker_df = df[df["DeliveryType"] == "striker"].copy()
    
    # Calculate dot balls for bowling
    striker_df["DotBall"] = ((striker_df["Runs_Batter"] == 0) & 
                            (striker_df["IsLegalDelivery"] == 1)).astype(int)
    
    # Group by pressure and bowler
    grouped = striker_df.groupby([pressure_col, "Bowler"]).agg(
        Runs=("Bowler_Runs", "sum"),
        Balls=pd.NamedAgg(column="IsLegalDelivery", aggfunc=lambda x: (x == 1).sum()),
        Wickets=pd.NamedAgg(column="Wicket", aggfunc=lambda x: (x == 1).sum()),
        Dots=("DotBall", "sum")
    ).reset_index()
    
    # Calculate Economy Rate and Dot Ball Percentage
    grouped["Economy"] = grouped.apply(lambda x: (x["Runs"] / (x["Balls"] / 6)) if x["Balls"] > 0 else 0, axis=1)
    grouped["DotBallPct"] = grouped.apply(lambda x: x["Dots"] / x["Balls"] if x["Balls"] > 0 else 0, axis=1)
    
    # If we don't need to aggregate further, return bowler-level data
    if not aggregate:
        return grouped
    
    # Aggregate across bowlers to get metrics by pressure level
    pressure_metrics = grouped.groupby(pressure_col).agg(
        TotalRuns=("Runs", "sum"),
        TotalBalls=("Balls", "sum"),
        TotalWickets=("Wickets", "sum"),
        TotalDots=("Dots", "sum")
    ).reset_index()
    
    # Calculate overall economy rate and dot ball percentage by pressure
    pressure_metrics["Economy"] = pressure_metrics.apply(
        lambda x: (x["TotalRuns"] / (x["TotalBalls"] / 6)) if x["TotalBalls"] > 0 else 0, axis=1
    )
    pressure_metrics["DotBallPct"] = pressure_metrics.apply(
        lambda x: x["TotalDots"] / x["TotalBalls"] if x["TotalBalls"] > 0 else 0, axis=1
    )
    
    # Order by pressure level
    pressure_metrics[pressure_col] = pd.Categorical(
        pressure_metrics[pressure_col], 
        categories=PRESSURE_LEVEL_ORDER, 
        ordered=True
    )
    pressure_metrics = pressure_metrics.sort_values(pressure_col)
    
    # Rename columns for display
    pressure_metrics = pressure_metrics.rename(columns={pressure_col: "Pressure"})
    
    return pressure_metrics


############################################
# 6) ADVANCED BATTING + BOWLING
############################################
def compute_overall_batting_metrics(df):
    """Compute batting metrics across all matches with correct dismissal tracking"""
    if df.empty:
        return pd.DataFrame()

    # First, identify unique innings per batsman
    # Only count when the batter themselves was dismissed
    innings_data = df.dropna(subset=["Batter"]).groupby(["Match_File", "Innings_Index", "Batter"]).agg(
        Runs=("Runs_Batter", "sum"),
        Was_Dismissed=("BatterDismissed", "max")  # Only count dismissals if the actual batter was out
    ).reset_index()

    # Now aggregate across all innings for each batter
    grouped = innings_data.groupby("Batter").agg(
        Innings=("Match_File", "count"),  # Total innings played
        Dismissals=("Was_Dismissed", "sum"),  # Only counts innings where the batter was dismissed
        Total_Runs=("Runs", "sum")  # Total runs scored
    ).reset_index()

    # Get ball-by-ball stats for strike rate
    ball_stats = df.dropna(subset=["Batter"]).groupby("Batter").agg(
        Balls=("IsBattingDelivery", "sum"),
        Boundaries_4s=("Runs_Batter", lambda x: (x == 4).sum()),
        Boundaries_6s=("Runs_Batter", lambda x: (x == 6).sum()),
    ).reset_index()

    # Merge the data
    final = grouped.merge(ball_stats, on="Batter", how="left")

    # Calculate metrics
    final["Average"] = final.apply(lambda x: x["Total_Runs"] / x["Dismissals"] if x["Dismissals"] > 0 else float('inf'),
                                   axis=1)
    final["StrikeRate"] = (final["Total_Runs"] / final["Balls"]) * 100
    final["Boundary_Rate"] = ((final["Boundaries_4s"] + final["Boundaries_6s"]) / final["Balls"]) * 100
    final["NotOuts"] = final["Innings"] - final["Dismissals"]

    # Rename to match expected columns
    final = final.rename(columns={"Total_Runs": "Runs"})

    return final.sort_values("Runs", ascending=False)


def compute_advanced_batting_metrics(df, bat_ppi_agg=None):
    """Compute advanced batting metrics"""
    if df.empty:
        return pd.DataFrame(columns=["Batter", "Total_Runs", "Average", "StrikeRate", "Finisher",
                                     "BoundaryRate", "DotBallPct", "StrikeRotationEfficiency",
                                     "PressurePerformanceIndex", "AvgBatPPI"])

    # Get basic batting stats
    batters = df[df["Batter"].notna()]["Batter"].unique()
    metrics = []

    for batter in batters:
        batter_df = df[df["Batter"] == batter]

        # Basic stats
        runs = batter_df["Runs_Batter"].sum()
        balls = batter_df["IsBattingDelivery"].sum()
        outs = batter_df["BatterDismissed"].sum()

        # Skip batters with minimal balls
        if balls < 10:
            continue

        # Averages, Strike Rates
        avg = runs / outs if outs > 0 else float('inf')
        sr = (runs / balls) * 100 if balls > 0 else 0

        # Boundary rate
        boundaries = ((batter_df["Runs_Batter"] == 4) | (batter_df["Runs_Batter"] == 6)).sum()
        boundary_rate = boundaries / balls if balls > 0 else 0

        # Dot ball %
        dots = ((batter_df["Runs_Batter"] == 0) & (batter_df["IsBattingDelivery"] == 1)).sum()
        dot_pct = dots / balls if balls > 0 else 0

        # Strike rotation
        ones_and_twos = ((batter_df["Runs_Batter"] == 1) | (batter_df["Runs_Batter"] == 2)).sum()
        strike_rotation = ones_and_twos / balls if balls > 0 else 0

        # Finisher metric (not out %)
        if "Match_File_Path" in batter_df.columns:
            innings = len(batter_df.groupby(["Match_File_Path", "Innings_Index"]))
        elif "Match_File" in batter_df.columns:
            innings = len(batter_df.groupby(["Match_File", "Innings_Index"]))
        else:
            innings = outs + 1  # Rough fallback

        not_outs = innings - outs
        finisher = not_outs / innings if innings > 0 else 0

        # Pressure performance
        high_pressure = batter_df[batter_df["DynamicPressureLabel"].isin(["High", "Extreme"])]
        if not high_pressure.empty and high_pressure[high_pressure["IsLegalDelivery"] == 1].shape[0] >= 10:
            pressure_sr = (high_pressure["Runs_Batter"].sum() /
                           high_pressure[high_pressure["IsLegalDelivery"] == 1].shape[0]) * 100
        else:
            pressure_sr = 0
        pressure_index = pressure_sr / sr if sr > 0 else 0

        # Merge with PPI data if available
        if bat_ppi_agg is not None:
            bat_ppi_data = bat_ppi_agg[bat_ppi_agg["Batter"] == batter]
            if not bat_ppi_data.empty:
                avg_bat_ppi = bat_ppi_data["AvgBatPPI"].values[0]
            else:
                avg_bat_ppi = np.nan
        else:
            avg_bat_ppi = np.nan

        metrics.append({
            "Batter": batter,
            "Total_Runs": runs,
            "Average": avg,
            "StrikeRate": sr,
            "BoundaryRate": boundary_rate,
            "DotBallPct": dot_pct,
            "StrikeRotationEfficiency": 1 - dot_pct,  # Inverse of dot ball percentage
            "Finisher": finisher,
            "PressurePerformanceIndex": pressure_index,
            "AvgBatPPI": avg_bat_ppi
        })

    return pd.DataFrame(metrics)


def compute_advanced_bowling_metrics(df, bowl_ppi_agg=None):
    """Compute advanced bowling metrics"""
    if df.empty:
        columns = ["Bowler", "Wickets", "Economy", "StrikeRate",
                   "BounceBackRate", "KeyWicketIndex", "DeathOversEconomy"]
        if bowl_ppi_agg is not None:
            columns.append("AvgBowlPPI")
        return pd.DataFrame(columns=columns)

    # Use only striker deliveries
    sub = df[df["DeliveryType"] == "striker"].dropna(subset=["Bowler"]).copy()
    sub["LegalDelivery"] = sub["IsLegalDelivery"]
    # Updated key wicket logic:
    sub["KeyWicket"] = sub.apply(
        lambda row: 1 if row["Wicket"] and (row.get("Mode_Of_Dismissal", "").lower() != "run out") and (
                (row.get("Batting_Position", 99) <= 3) or
                (((row.get("Batting_Position", 99) <= 6) and (row["Strike_Batsman_Runs"] > 10)) or
                 (row["Strike_Batsman_Runs"] > 30) or
                 (row["Current_Partnership"] > 30))
        )
        else 0,
        axis=1
    )
    sub["DeathOver"] = sub.apply(
        lambda row: 1 if (
                (row["Match_Type"] == "T20" and row["Over"] >= 15) or
                (row["Match_Type"] != "T20" and row["Over"] >= 40)
        ) else 0, axis=1
    )
    sub["PrevBallBoundary"] = sub["Previous_Ball_Result"] == "Boundary"
    sub["BounceBackSuccess"] = (sub["PrevBallBoundary"]) & (
            (sub["Runs_Batter"] == 0) | (sub["Wicket"])
    )
    grouped = sub.groupby("Bowler").agg(
        Balls=("LegalDelivery", "sum"),
        Runs=("Bowler_Runs", "sum"),
        Wickets=("Wicket", "sum"),
        KeyWickets=("KeyWicket", "sum"),
        BoundaryBefore=("PrevBallBoundary", "sum"),
        BounceBackSuccess=("BounceBackSuccess", "sum"),
        DeathBalls=("DeathOver", lambda x: sum(x * sub.loc[x.index, "LegalDelivery"])),
        DeathRuns=("DeathOver", lambda x: sum(x * sub.loc[x.index, "Runs_Total"]))
    ).reset_index()
    grouped["Economy"] = grouped.apply(
        lambda row: (row["Runs"] / (row["Balls"] / 6)) if row["Balls"] > 0 else 0,
        axis=1
    )
    grouped["StrikeRate"] = grouped.apply(
        lambda row: (row["Balls"] / row["Wickets"]) if row["Wickets"] > 0 else float('inf'),
        axis=1
    )
    grouped["BounceBackRate"] = grouped.apply(
        lambda row: (row["BounceBackSuccess"] / row["BoundaryBefore"])
        if row["BoundaryBefore"] > 0 else 0,
        axis=1
    )
    grouped["KeyWicketIndex"] = grouped.apply(
        lambda row: (row["KeyWickets"] / row["Wickets"]) if row["Wickets"] > 0 else 0,
        axis=1
    )
    grouped["DeathOversEconomy"] = grouped.apply(
        lambda row: (row["DeathRuns"] / (row["DeathBalls"] / 6))
        if row["DeathBalls"] > 0 else row["Economy"],
        axis=1
    )
    grouped.loc[grouped["StrikeRate"] == float('inf'), "StrikeRate"] = grouped["Balls"].max() * 2

    # Merge with PPI data if available
    if bowl_ppi_agg is not None:
        grouped = pd.merge(grouped, bowl_ppi_agg, on="Bowler", how="left")
        return grouped[["Bowler", "Wickets", "Economy", "StrikeRate",
                        "BounceBackRate", "KeyWicketIndex", "DeathOversEconomy", "AvgBowlPPI"]]
    else:
        return grouped[["Bowler", "Wickets", "Economy", "StrikeRate",
                        "BounceBackRate", "KeyWicketIndex", "DeathOversEconomy"]]


############################################
# 7) RADAR CHART
############################################
def create_radar_chart(df, player_col, player_list, metrics, title):
    if df.empty or not player_list:
        return None
    sub = df[df[player_col].isin(player_list)].copy()
    if sub.empty:
        return None

    # Metrics to invert
    invert_metrics = []
    if player_col == "Batter":
        invert_metrics = ["DotBallPct"]
    elif player_col == "Bowler":
        invert_metrics = ["Economy", "StrikeRate", "DeathOversEconomy"]

    clipped_rows = []
    for _, row in sub.iterrows():
        player_name = row[player_col]
        data_row = {"player": player_name}
        for m in metrics:
            val_raw = row[m]
            # Calculate clipped ranges
            low_ = df[m].quantile(0.05)
            high_ = df[m].quantile(0.95)
            val_clipped = np.clip(val_raw, low_, high_)
            min_ = df[m].clip(low_, high_).min()
            max_ = df[m].clip(low_, high_).max()
            rng = max_ - min_ if max_ > min_ else 1e-9

            if m in invert_metrics:
                val_scaled = 1 - ((val_clipped - min_) / rng)
            else:
                val_scaled = (val_clipped - min_) / rng

            data_row[m] = val_scaled
            data_row[f"{m}_original"] = val_raw
        clipped_rows.append(data_row)

    radar_data = []
    for row in clipped_rows:
        for m in metrics:
            radar_data.append({
                "player": row["player"],
                "Metric": m,
                "Value": row[m],
                "Original": row[f"{m}_original"],
                "Better": "Lower" if m in invert_metrics else "Higher"
            })
    radar_df = pd.DataFrame(radar_data)

    radar_df["OriginalFormatted"] = radar_df.apply(
        lambda x: f"{x['Original']:.2f}" if isinstance(x['Original'], (int, float)) else str(x['Original']),
        axis=1
    )
    radar_df["HoverText"] = radar_df.apply(
        lambda x: (
            f"{x['player']}<br>{x['Metric']}: {x['OriginalFormatted']}"
            f"<br>({'Lower is better' if x['Better'] == 'Lower' else 'Higher is better'})"
        ),
        axis=1
    )

    fig = px.line_polar(
        radar_df,
        r="Value",
        theta="Metric",
        color="player",
        line_close=True,
        hover_name="player",
        hover_data={"Value": False, "Metric": False, "player": False, "Original": False, "Better": False,
                    "OriginalFormatted": False},
        custom_data=["HoverText"],
        template="plotly_dark",
        title=title,
        range_r=[0, 1]
    )
    fig.update_traces(
        fill='toself',
        hovertemplate="%{customdata[0]}<extra></extra>"
    )
    fig.add_annotation(
        text="All metrics oriented so outward = better performance",
        xref="paper", yref="paper",
        x=0.5, y=-0.1,
        showarrow=False,
        font=dict(size=12, color="#B0B0B0")
    )
    return fig


############################################
# 8) DEFAULT SELECTION HELPERS
############################################
def get_default_batters_pressure(df, n=5):
    if df.empty:
        return []
    grouped = df.groupby("Batter", dropna=True)["Runs"].sum().reset_index()
    grouped = grouped.sort_values("Runs", ascending=False)
    top = grouped.head(n)["Batter"].tolist()
    return top


def get_default_bowlers_pressure(df, n=5):
    if df.empty:
        return []
    grouped = df.groupby("Bowler", dropna=True)["Wickets"].sum().reset_index()
    grouped = grouped.sort_values("Wickets", ascending=False)
    top = grouped.head(n)["Bowler"].tolist()
    return top


def get_default_batters_adv(df, n=5):
    if df.empty:
        return []
    top = df.sort_values("Total_Runs", ascending=False).head(n)["Batter"].tolist()
    return top


def get_default_bowlers_adv(df, n=5):
    if df.empty:
        return []
    top = df.sort_values("Wickets", ascending=False).head(n)["Bowler"].tolist()
    return top


############################################
# 9) STREAMLIT APP
############################################
def display_metric_card(title, value, description=None):
    html = f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        {f'<div style="color: #B0B0B0; font-size: 0.8rem; margin-top: 0.3rem;' +
         f'">{description}</div>' if description else ''}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def create_helper_text(text):
    st.markdown(f'<div class="helper-text">{text}</div>', unsafe_allow_html=True)


def create_insight_banner(text, icon="ℹ️"):
    st.markdown(f'<div class="insight-banner">{icon} <span style="font-weight: 500;">{text}</span></div>',
                unsafe_allow_html=True)


def display_performance_insights(df):
    """
    Show performance insights using the separate batting and bowling pressure labels.
    """
    # Define minimum ball threshold for including players in analysis
    min_ball_filter = 20  # Minimum balls faced/bowled to qualify

    # Tabs for batting and bowling insights
    tab1, tab2 = st.tabs(["Batting Insights", "Bowling Insights"])

    with tab1:
        st.markdown("### Batting Performance by Pressure")
        # Use BattingPressureLabel for batting analysis
        bat_pressure = compute_batting_metrics_by_pressure(df, pressure_col="BattingPressureLabel")
        if not bat_pressure.empty:
            # Determine which pressure column to use
            pressure_col = "BattingPressureLabel" if "BattingPressureLabel" in bat_pressure.columns else "DynamicPressureLabel"
            bat_pressure = order_pressure_levels(bat_pressure, pressure_col=pressure_col)

            # Use the same pressure_col for filtering
            high_pressure = bat_pressure[bat_pressure[pressure_col].isin(["High", "Extreme"])]
            if not high_pressure.empty:
                # Sort by descending Strike Rate
                top_strikers = high_pressure.sort_values("StrikeRate", ascending=False).head(10)

                # Create an insights card
                st.markdown(
                    f"""<div class="insight-card">
                    <h4>Best Strikers Under Pressure</h4>
                    <p>Players with highest strike rates under high/extreme pressure (min {min_ball_filter} balls):</p>
                    </div>""",
                    unsafe_allow_html=True
                )
                # Display results in a clean format
                for i, row in top_strikers.iterrows():
                    if row["Balls"] >= min_ball_filter:
                        st.markdown(
                            f"""<div class="insight-card">
                            <b>{row["Batter"]}</b> - {row['StrikeRate']:.1f} SR ({row['Runs']} runs off {int(row['Balls'])} balls)
                            <br><small>Under {row[pressure_col]} pressure</small>
                            </div>""",
                            unsafe_allow_html=True
                        )

    with tab2:
        st.markdown("### Bowling Performance by Pressure")
        # Use BowlingPressureLabel for bowling analysis
        bowl_pressure = compute_bowling_metrics_by_pressure(df, pressure_col="BowlingPressureLabel")
        if not bowl_pressure.empty:
            # Determine which pressure column to use
            pressure_col = "BowlingPressureLabel" if "BowlingPressureLabel" in bowl_pressure.columns else "DynamicPressureLabel"
            bowl_pressure = order_pressure_levels(bowl_pressure, pressure_col=pressure_col)

            # Use the same pressure_col for filtering
            high_pressure = bowl_pressure[bowl_pressure[pressure_col].isin(["High", "Extreme"])]
            if not high_pressure.empty:
                # Sort by ascending Economy
                top_bowlers = high_pressure.sort_values("Economy", ascending=True).head(10)

                # Create an insights card
                st.markdown(
                    f"""<div class="insight-card">
                    <h4>Best Bowlers Under Pressure</h4>
                    <p>Players with lowest economy rates under high/extreme pressure (min {min_ball_filter} balls):</p>
                    </div>""",
                    unsafe_allow_html=True
                )
                # Display results in a clean format
                for i, row in top_bowlers.iterrows():
                    if row["Balls"] >= min_ball_filter:
                        st.markdown(
                            f"""<div class="insight-card">
                            <b>{row["Bowler"]}</b> - {row['Economy']:.1f} Economy ({row['Wickets']} wickets in {row['Balls'] / 6:.1f} overs)
                            <br><small>Under {row[pressure_col]} pressure</small>
                            </div>""",
                            unsafe_allow_html=True
                        )


############################################
# 6.5) PLAYER PERFORMANCE INDEX (PPI) CALCULATIONS
############################################
def compute_match_ppi_batting(df, min_balls=5):
    """
    For each match_file + Batter, calculate a Batting Performance Index
    scaled 0-100 among qualified batters in that match.
    """
    if df.empty:
        return pd.DataFrame(columns=["Match_File", "Batter", "BatPPI"])

    sub = df.dropna(subset=["Batter"]).copy()

    # 1. Calculate batting impact
    sub["BatImpact"] = 0.8 * sub["Runs_Batter"]
    sub.loc[sub["Runs_Batter"] == 4, "BatImpact"] += 1.5  # Enhanced boundary bonus
    sub.loc[sub["Runs_Batter"] == 6, "BatImpact"] += 3.0  # Bigger six bonus
    sub.loc[sub["BatterDismissed"] == 1, "BatImpact"] -= 3.0  # Reduced dismissal penalty

    # 2. Determine pressure - handle multiple possible column names
    if "BattingPressureLabel" in sub.columns:
        # Use the new batting-specific pressure
        sub["AdjustedPressure"] = sub["BattingPressureLabel"].map({
            "Low": 1.0,
            "Medium": 1.5,
            "High": 2.0,
            "Extreme": 2.5
        }).fillna(1.0)
    elif "DynamicPressureLabel" in sub.columns:
        # Fall back to the old single pressure label
        sub["AdjustedPressure"] = sub["DynamicPressureLabel"].map({
            "Low": 1.0,
            "Medium": 1.5,
            "High": 2.0,
            "Extreme": 2.5
        }).fillna(1.0)
    elif "DynamicPressureScore" in sub.columns:
        # Direct use of the numeric score
        sub["AdjustedPressure"] = sub["DynamicPressureScore"].clip(lower=1.0)
    else:
        # No pressure data available, use 1.0 as default
        sub["AdjustedPressure"] = 1.0

    # 3. Calculate weighted impact with adjusted pressure
    sub["WeightedBallImpact"] = sub["BatImpact"] * sub["AdjustedPressure"]

    # Group by match and batter
    grouped = sub.groupby(["Match_File", "Batter"]).agg(
        RunsScored=("Runs_Batter", "sum"),
        BallsFaced=("IsBattingDelivery", "sum"),
        DismissedCount=("BatterDismissed", "sum"),
        SumWeightedImpact=("WeightedBallImpact", "sum")
    ).reset_index()

    # Process each match separately for consistent PPI scaling
    result_dfs = []
    for match_file, match_group in grouped.groupby("Match_File"):
        # Allow dismissed batters to qualify regardless of balls faced
        match_group["Qualified"] = match_group.apply(
            lambda row: (row["BallsFaced"] >= min_balls) or (row["DismissedCount"] > 0),
            axis=1
        )

        qualified_group = match_group[match_group["Qualified"]]
        if qualified_group.empty:
            match_group["BatPPI"] = np.nan
        else:
            # Handle scaling with outlier protection
            wmin = qualified_group["SumWeightedImpact"].min()
            wmax = qualified_group["SumWeightedImpact"].max()

            # Prevent division by zero and handle outliers
            impact_range = max(1.0, wmax - wmin)

            # Calculate PPI for qualified batters
            match_group.loc[match_group["Qualified"], "BatPPI"] = (
                    (match_group.loc[match_group["Qualified"], "SumWeightedImpact"] - wmin) / impact_range * 100
            )

            # Set NaN for unqualified batters
            match_group.loc[~match_group["Qualified"], "BatPPI"] = np.nan

        result_dfs.append(match_group)

    # Combine all match results
    result_df = pd.concat(result_dfs, ignore_index=True)
    return result_df[["Match_File", "Batter", "BatPPI"]]


def compute_match_ppi_bowling(df, min_balls=6, cameo_overs=2.5):
    """
    For each match_file + Bowler, calculate a Bowling Performance Index
    scaled 0-100 among qualified bowlers in that match.
    """
    if df.empty:
        return pd.DataFrame(columns=["Match_File", "Bowler", "BowlPPI"])

    # Filter to deliveries where bowler is known
    sub = df.dropna(subset=["Bowler"]).copy()

    # 1) Base impact: enhanced weighting for wickets, reduced penalty for runs
    sub["BowlImpact"] = 0.0
    sub.loc[sub["Wicket"] == 1, "BowlImpact"] += 6.0  # Increased from 4.0 to 6.0
    # Reduced penalty per run conceded
    sub["BowlImpact"] -= 0.4 * sub["Bowler_Runs"]  # Reduced from 0.6 to 0.4
    # Small bonus for dot balls
    sub.loc[sub["Bowler_Runs"] == 0, "BowlImpact"] += 0.5

    # 2) Determine pressure based on available columns
    if "BowlingPressureLabel" in sub.columns:
        # Use the new bowling-specific pressure
        sub["AdjustedPressure"] = sub["BowlingPressureLabel"].map({
            "Low": 1.0,
            "Medium": 1.5,
            "High": 2.0,
            "Extreme": 2.5
        }).fillna(1.0)
    elif "DynamicPressureLabel" in sub.columns:
        # Fall back to the old single pressure label
        sub["AdjustedPressure"] = sub["DynamicPressureLabel"].map({
            "Low": 1.0,
            "Medium": 1.5,
            "High": 2.0,
            "Extreme": 2.5
        }).fillna(1.0)
    elif "DynamicPressureScore" in sub.columns:
        # Direct use of the numeric score
        sub["AdjustedPressure"] = sub["DynamicPressureScore"].clip(lower=1.0)
    else:
        # No pressure data available, use 1.0 as default
        sub["AdjustedPressure"] = 1.0

    # 3) Calculate weighted impact
    sub["WeightedBallImpact"] = sub["BowlImpact"] * sub["AdjustedPressure"]

    # Group by match + bowler
    grouped = sub.groupby(["Match_File", "Bowler"]).agg(
        BallsBowled=("IsLegalDelivery", "sum"),
        WicketsTaken=("Wicket", "sum"),  # Added to track wickets
        SumWeightedImpact=("WeightedBallImpact", "sum")
    ).reset_index()

    # Process each match separately for consistent PPI scaling
    results = []
    for match_id, grp in grouped.groupby("Match_File"):
        grp = grp.copy()
        grp["OversBowled"] = grp["BallsBowled"] / 6.0
        grp["Qualified"] = grp["BallsBowled"] >= min_balls

        qualified = grp[grp["Qualified"]]
        if not qualified.empty:
            # Less aggressive clipping - 2nd to 98th percentile
            low = qualified["SumWeightedImpact"].quantile(0.02)
            high = qualified["SumWeightedImpact"].quantile(0.98)
            rng = max(1e-9, high - low)

            # Scale to 0-100
            for idx in qualified.index:
                raw_val = qualified.at[idx, "SumWeightedImpact"]
                clipped_val = max(low, min(raw_val, high))
                ppi_0_100 = (clipped_val - low) / rng * 100
                grp.at[idx, "BowlPPI"] = ppi_0_100
        else:
            grp["BowlPPI"] = np.nan

        # Set non-qualified to NaN
        grp.loc[~grp["Qualified"], "BowlPPI"] = np.nan

        # Apply a milder cameo penalty AFTER scaling to 0-100
        # Using linear rather than squared discount
        for idx in grp.index:
            overs = grp.at[idx, "OversBowled"]
            if overs < cameo_overs and pd.notna(grp.at[idx, "BowlPPI"]):
                # Apply linear discount instead of squared
                final_discount = overs / cameo_overs
                grp.at[idx, "BowlPPI"] *= final_discount

        # Apply a floor for multi-wicket performances
        for idx in grp.index:
            wickets = grp.at[idx, "WicketsTaken"]
            if pd.notna(grp.at[idx, "BowlPPI"]) and wickets >= 2:
                # Ensure multi-wicket hauls get at least 20 PPI
                grp.at[idx, "BowlPPI"] = max(20, grp.at[idx, "BowlPPI"])

        results.append(grp)

    return pd.concat(results, ignore_index=True)[["Match_File", "Bowler", "BowlPPI"]]


def main():
    st.set_page_config(
        page_title="Cric Assist - Analytics",
        page_icon="🏏",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    inject_custom_css()

    # Sidebar
    with st.sidebar:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("🏏")
        with col2:
            st.markdown("## Cric Assist")
        st.markdown("---")
        st.markdown("### Analytics Dashboard")
        st.markdown("#### About")
        st.markdown("""
        This dashboard analyzes cricket match data with a focus on 
        performance under different pressure situations.
        """)
        st.markdown("---")
        st.markdown("#### Data Source")
        st.markdown("Match data in JSON format from the data directory.")
        st.markdown("---")
        st.caption("Version 1.0.0")
        st.caption("© 2025 Cric Assist")

    st.markdown("# 🏏 Cric Assist Analytics")

    # Load data
    with st.spinner("Loading and processing cricket match data..."):
        df_all = load_and_combine_data("data")
    if df_all.empty:
        st.error("### No data found")
        st.markdown("Please place your JSON files in the 'data' folder.")
        return

    # Calculate PPIs after loading the data
    ppi_bat_df = compute_match_ppi_batting(df_all, min_balls=5)
    ppi_bowl_df = compute_match_ppi_bowling(df_all, min_balls=6)

    # Aggregate PPIs across matches - fix the skipna error
    bat_ppi_agg = ppi_bat_df.groupby("Batter")["BatPPI"].mean().reset_index()
    bat_ppi_agg.rename(columns={"BatPPI": "AvgBatPPI"}, inplace=True)

    bowl_ppi_agg = ppi_bowl_df.groupby("Bowler")["BowlPPI"].mean().reset_index()
    bowl_ppi_agg.rename(columns={"BowlPPI": "AvgBowlPPI"}, inplace=True)

    # Top metrics
    with st.container():
        st.markdown("## Key Metrics & Filters")
        match_count = df_all["Match_File"].nunique()
        player_count = len(set(df_all["Batter"].dropna().tolist() + df_all["Bowler"].dropna().tolist()))
        col1, col2, col3 = st.columns(3)
        with col1:
            display_metric_card("Matches Analyzed", f"{match_count:,}", "Total matches in dataset")
        with col2:
            display_metric_card("Players", f"{player_count:,}", "Total unique players")
        with col3:
            display_metric_card("Data Points", f"{len(df_all):,}", "Ball-by-ball records")

        st.markdown("### Data Filters")
        create_helper_text(
            "Use these filters to narrow down the analysis to specific seasons or players with a minimum number of games"
        )
        col1, col2 = st.columns([2, 1])
        with col1:
            seasons = sorted(df_all["Season"].dropna().unique().tolist())
            if seasons:
                selected_seasons = st.multiselect("Select Season(s)", seasons, default=seasons)
                df_all = df_all[df_all["Season"].isin(selected_seasons)]
        with col2:
            min_games = st.number_input("Minimum Games Played", min_value=1, value=3,
                                        help="Only include players who have played at least this many games")

    if df_all.empty:
        st.warning("No data remains after applying filters.")
        return

    # Explanation of metrics
    with st.expander("Understanding Cricket Metrics", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Dynamic Pressure")
            st.markdown("""
            Measures how stressful the game situation is, considering:
            - Match innings (1st or 2nd)
            - Expected vs. actual run rates (1st innings)
            - Run chase difficulty (2nd innings)
            - Wickets in hand
            - Stage of the innings

            Pressure levels: **Low**, **Medium**, **High**, and **Extreme**
            """)
            st.markdown("### Finisher Rate")
            st.markdown("""
            The percentage of innings where a batter remains not out, reflecting 
            their ability to "finish" an innings. Higher is better for middle/lower order batters.
            """)
        with col2:
            st.markdown("### Bounce Back Rate (BBR)")
            st.markdown("""
            For bowlers, after conceding a boundary, how often do they respond with a 
            dot ball or wicket on the next delivery? Higher BBR = better recovery after being hit.
            """)
            st.markdown("### Key Wicket Index (KWI)")
            st.markdown("""
            Measures the impact of a bowler's wicket-taking by flagging "key wickets" as those that either:
            - Dismiss an opener (one of the first three batsmen), or 
            - Remove a top-order batsman (batting position ≤ 6 with more than 10 runs)
            - Or dismiss a batsman who has scored more than 30 runs
            - Or break a partnership exceeding 30 runs
            """)
            st.markdown("### Death Overs Economy")
            st.markdown("""
            Economy rate specifically in the final overs (16–20) when batting acceleration 
            typically occurs. Lower = better performance at the death.
            """)

    # Tabs
    tabs = st.tabs([
        "📊 Batting - Dynamic Pressure",
        "🎯 Bowling - Dynamic Pressure",
        "📈 Advanced Metrics + Radar",
        "🔍 Raw Data Explorer",
        "📝 Match PPI Viewer"  # New tab
    ])

    # 1) Batting: Dynamic Pressure
    with tabs[0]:
        st.markdown("## Batting Performance Under Pressure")
        create_helper_text("Analyze how batters perform under different levels of match pressure")
        bat_pressure = compute_batting_metrics_by_pressure(df_all, pressure_col="DynamicPressureLabel")
        if bat_pressure.empty:
            st.info("No batting data or no dynamic pressure calculation available.")
        else:
            bat_games = compute_batting_games_played(df_all)
            merged_bat = bat_pressure.merge(bat_games, on="Batter", how="left")
            merged_bat = merged_bat[merged_bat["GamesPlayed"] >= min_games]
            if merged_bat.empty:
                st.info("No batters meet the minimum games criteria.")
            else:
                col1, col2 = st.columns([3, 1])
                with col1:
                    all_batters = sorted(merged_bat["Batter"].unique().tolist())
                    default_batters = get_default_batters_pressure(merged_bat, n=5)
                    st.markdown("### Select Batters to Compare")
                    selected_batters = st.multiselect(
                        "",
                        all_batters,
                        default=default_batters,
                        help="Select up to 5 batters for best visualization"
                    )
                with col2:
                    create_insight_banner(
                        "Compare how strike rates and dot ball percentages change under different pressure situations."
                    )

                if selected_batters:
                    sub_bat = merged_bat[merged_bat["Batter"].isin(selected_batters)]
                    st.markdown("### Performance Metrics by Pressure Level")
                    display_df = sub_bat.copy()
                    display_df["StrikeRate"] = display_df["StrikeRate"].round(2)
                    display_df["DotBallPct"] = (display_df["DotBallPct"] * 100).round(2).astype(str) + '%'
                    order = ["Low", "Medium", "High", "Extreme"]
                    display_df["DynamicPressureLabel"] = pd.Categorical(
                        display_df["DynamicPressureLabel"],
                        categories=order,
                        ordered=True
                    )
                    display_df = display_df.sort_values("DynamicPressureLabel")
                    st.dataframe(display_df, use_container_width=True)

                    st.markdown("### Strike Rate Comparison")
                    sub_bat["DynamicPressureLabel"] = pd.Categorical(
                        sub_bat["DynamicPressureLabel"],
                        categories=order,
                        ordered=True
                    )
                    fig_bat = px.bar(
                        sub_bat,
                        x="DynamicPressureLabel",
                        y="StrikeRate",
                        color="Batter",
                        barmode="group",
                        title="Batting Strike Rate by Pressure Level",
                        color_discrete_sequence=px.colors.qualitative.Bold,
                        labels={"DynamicPressureLabel": "Pressure Level", "StrikeRate": "Strike Rate"},
                        category_orders={"DynamicPressureLabel": order}
                    )
                    fig_bat.update_layout(
                        template="plotly_dark",
                        plot_bgcolor="#1E2130",
                        paper_bgcolor="#1E2130",
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                        xaxis_title="Pressure Level",
                        yaxis_title="Strike Rate",
                        margin=dict(l=10, r=10, t=50, b=10)
                    )
                    st.plotly_chart(fig_bat, use_container_width=True)

                    st.markdown("### Dot Ball Percentage Comparison")
                    fig_bat2 = px.bar(
                        sub_bat,
                        x="DynamicPressureLabel",
                        y="DotBallPct",
                        color="Batter",
                        barmode="group",
                        title="Dot Ball % by Pressure Level",
                        color_discrete_sequence=px.colors.qualitative.Bold,
                        labels={"DynamicPressureLabel": "Pressure Level", "DotBallPct": "Dot Ball %"},
                        category_orders={"DynamicPressureLabel": order}
                    )
                    fig_bat2.update_layout(
                        template="plotly_dark",
                        plot_bgcolor="#1E2130",
                        paper_bgcolor="#1E2130",
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                        xaxis_title="Pressure Level",
                        yaxis_title="Dot Ball %",
                        margin=dict(l=10, r=10, t=50, b=10)
                    )
                    st.plotly_chart(fig_bat2, use_container_width=True)
                else:
                    st.info("Please select at least one batter to display charts.")

    # 2) Bowling: Dynamic Pressure
    with tabs[1]:
        st.markdown("## Bowling Performance Under Pressure")
        create_helper_text("Analyze how bowlers perform under different levels of match pressure")
        bowl_pressure = compute_bowling_metrics_by_pressure(df_all, pressure_col="DynamicPressureLabel")
        if bowl_pressure.empty:
            st.info("No bowling data or no dynamic pressure calculation available.")
        else:
            bowl_games = compute_bowling_games_played(df_all)
            merged_bowl = bowl_pressure.merge(bowl_games, on="Bowler", how="left")
            merged_bowl = merged_bowl[merged_bowl["GamesPlayed"] >= min_games]
            if merged_bowl.empty:
                st.info("No bowlers meet the minimum games criteria.")
            else:
                col1, col2 = st.columns([3, 1])
                with col1:
                    all_bowlers = sorted(merged_bowl["Bowler"].unique().tolist())
                    default_bowlers = get_default_bowlers_pressure(merged_bowl, n=5)
                    st.markdown("### Select Bowlers to Compare")
                    selected_bowlers = st.multiselect(
                        "",
                        all_bowlers,
                        default=default_bowlers,
                        help="Select up to 5 bowlers for best visualization"
                    )
                with col2:
                    create_insight_banner(
                        "Compare how economy rates and dot ball percentages change under different pressure situations."
                    )

                if selected_bowlers:
                    sub_bowl = merged_bowl[merged_bowl["Bowler"].isin(selected_bowlers)]
                    st.markdown("### Performance Metrics by Pressure Level")
                    display_df = sub_bowl.copy()
                    display_df["Economy"] = display_df["Economy"].round(2)
                    display_df["DotBallPct"] = (display_df["DotBallPct"] * 100).round(2).astype(str) + '%'
                    order = ["Low", "Medium", "High", "Extreme"]
                    display_df["DynamicPressureLabel"] = pd.Categorical(
                        display_df["DynamicPressureLabel"],
                        categories=order,
                        ordered=True
                    )
                    display_df = display_df.sort_values("DynamicPressureLabel")
                    st.dataframe(display_df, use_container_width=True)

                    st.markdown("### Economy Rate Comparison")
                    sub_bowl["DynamicPressureLabel"] = pd.Categorical(
                        sub_bowl["DynamicPressureLabel"],
                        categories=order,
                        ordered=True
                    )
                    fig_bowl = px.bar(
                        sub_bowl,
                        x="DynamicPressureLabel",
                        y="Economy",
                        color="Bowler",
                        barmode="group",
                        title="Bowling Economy by Pressure Level",
                        color_discrete_sequence=px.colors.qualitative.Bold,
                        labels={"DynamicPressureLabel": "Pressure Level", "Economy": "Economy Rate (RPO)"},
                        category_orders={"DynamicPressureLabel": order}
                    )
                    fig_bowl.update_layout(
                        template="plotly_dark",
                        plot_bgcolor="#1E2130",
                        paper_bgcolor="#1E2130",
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                        xaxis_title="Pressure Level",
                        yaxis_title="Economy Rate (RPO)",
                        margin=dict(l=10, r=10, t=50, b=10)
                    )
                    st.plotly_chart(fig_bowl, use_container_width=True)

                    st.markdown("### Dot Ball Percentage Comparison")
                    fig_bowl2 = px.bar(
                        sub_bowl,
                        x="DynamicPressureLabel",
                        y="DotBallPct",
                        color="Bowler",
                        barmode="group",
                        title="Bowling Dot Ball % by Pressure Level",
                        color_discrete_sequence=px.colors.qualitative.Bold,
                        labels={"DynamicPressureLabel": "Pressure Level", "DotBallPct": "Dot Ball %"},
                        category_orders={"DynamicPressureLabel": order}
                    )
                    fig_bowl2.update_layout(
                        template="plotly_dark",
                        plot_bgcolor="#1E2130",
                        paper_bgcolor="#1E2130",
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                        xaxis_title="Pressure Level",
                        yaxis_title="Dot Ball %",
                        margin=dict(l=10, r=10, t=50, b=10)
                    )
                    st.plotly_chart(fig_bowl2, use_container_width=True)
                else:
                    st.info("Please select at least one bowler to display charts.")

    # 3) Advanced metrics + Radar
    with tabs[2]:
        st.markdown("## Advanced Metrics & Radar Charts")
        create_helper_text("Compare players using multiple performance dimensions in radar charts")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Batting Advanced Metrics")
            adv_bat = compute_advanced_batting_metrics(df_all, bat_ppi_agg)
            bat_games = compute_batting_games_played(df_all)
            adv_bat = adv_bat.merge(bat_games, on="Batter", how="left")
            adv_bat = adv_bat[adv_bat["GamesPlayed"] >= min_games]
            if adv_bat.empty:
                st.info("No batters meet the minimum games criteria.")
            else:
                display_bat = adv_bat.copy()
                display_bat["Average"] = display_bat["Average"].round(2)
                display_bat["StrikeRate"] = display_bat["StrikeRate"].round(2)
                display_bat["PressurePerformanceIndex"] = display_bat["PressurePerformanceIndex"].round(2)
                display_bat["Finisher"] = (display_bat["Finisher"] * 100).round(1).astype(str) + '%'
                display_bat["BoundaryRate"] = (display_bat["BoundaryRate"] * 100).round(1).astype(str) + '%'
                display_bat["DotBallPct"] = (display_bat["DotBallPct"] * 100).round(1).astype(str) + '%'
                display_bat["StrikeRotationEfficiency"] = (display_bat["StrikeRotationEfficiency"] * 100).round(
                    1).astype(str) + '%'
                st.dataframe(display_bat, use_container_width=True)

                all_bat = sorted(adv_bat["Batter"].unique().tolist())
                default_batters_adv = get_default_batters_adv(adv_bat, n=3)
                st.markdown("### Select Batters for Radar")
                sel_bat = st.multiselect(
                    "Choose up to 5 batters",
                    all_bat,
                    default=default_batters_adv,
                    help="Radar chart works best with 3-5 players"
                )
                if sel_bat:
                    metrics_batting = ["AvgBatPPI", "Average", "StrikeRate", "Finisher",
                                       "BoundaryRate", "StrikeRotationEfficiency", "PressurePerformanceIndex"]

                    with st.expander("Understanding the Radar Axes", expanded=False):
                        st.markdown("""
                        - **AvgBatPPI**: Batting Performance Index (weighted by pressure)
                        - **Average**: Batting average (runs per dismissal)
                        - **StrikeRate**: Runs per 100 balls
                        - **Finisher**: % of innings where batter remains not out
                        - **BoundaryRate**: % of balls faced that result in boundaries
                        - **StrikeRotationEfficiency**: (1 - dot ball pct)
                        - **PressurePerformanceIndex**: Strike rate in High/Extreme pressure, normalized
                        > **Note:** All metrics scale so that outward is "better."
                        """)
                    fig_radar_bat = create_radar_chart(adv_bat, "Batter", sel_bat, metrics_batting,
                                                       "Batting Comparison")
                    if fig_radar_bat:
                        fig_radar_bat.update_layout(
                            template="plotly_dark",
                            polar=dict(
                                bgcolor="#272D3F",
                                angularaxis=dict(linewidth=1, linecolor='#3A4566'),
                                radialaxis=dict(gridcolor='#3A4566')
                            ),
                            paper_bgcolor="#1E2130",
                            font=dict(color="#E6E6E6"),
                            margin=dict(l=80, r=80, t=50, b=50)
                        )
                        st.plotly_chart(fig_radar_bat, use_container_width=True)

        with col2:
            st.markdown("### Bowling Advanced Metrics")
            adv_bowl = compute_advanced_bowling_metrics(df_all, bowl_ppi_agg)
            bowl_games = compute_bowling_games_played(df_all)
            adv_bowl = adv_bowl.merge(bowl_games, on="Bowler", how="left")
            adv_bowl = adv_bowl[adv_bowl["GamesPlayed"] >= min_games]
            if adv_bowl.empty:
                st.info("No bowlers meet the minimum games criteria.")
            else:
                display_bowl = adv_bowl.copy()
                for col_ in ["Economy", "StrikeRate", "DeathOversEconomy"]:
                    display_bowl[col_] = display_bowl[col_].round(2)
                display_bowl["BounceBackRate"] = (display_bowl["BounceBackRate"] * 100).round(1).astype(str) + '%'
                display_bowl["KeyWicketIndex"] = (display_bowl["KeyWicketIndex"] * 100).round(1).astype(str) + '%'
                st.dataframe(display_bowl, use_container_width=True)

                all_bowl = sorted(adv_bowl["Bowler"].unique().tolist())
                default_bowlers_adv = get_default_bowlers_adv(adv_bowl, n=3)
                st.markdown("### Select Bowlers for Radar")
                sel_bowl = st.multiselect(
                    "Choose up to 5 bowlers",
                    all_bowl,
                    default=default_bowlers_adv,
                    help="Radar chart works best with 3-5 players"
                )
                if sel_bowl:
                    metrics_bowling = ["AvgBowlPPI", "Economy", "StrikeRate", "BounceBackRate",
                                       "KeyWicketIndex", "DeathOversEconomy"]

                    with st.expander("Understanding the Radar Axes", expanded=False):
                        st.markdown("""
                        - **AvgBowlPPI**: Bowling Performance Index (weighted by pressure)
                        - **Economy**: Runs per over (lower = better)
                        - **StrikeRate**: Balls per wicket (lower = better)
                        - **BounceBackRate**: Recovery after a boundary
                        - **KeyWicketIndex**: Proportion of wickets in first 10 overs
                        - **DeathOversEconomy**: Economy from overs 16–20
                        > **Note**: For Economy, StrikeRate, and DeathOversEconomy, 
                        > lower is better; those are inverted on the chart.
                        """)
                    fig_radar_bowl = create_radar_chart(adv_bowl, "Bowler", sel_bowl, metrics_bowling,
                                                        "Bowling Comparison")
                    if fig_radar_bowl:
                        fig_radar_bowl.update_layout(
                            template="plotly_dark",
                            polar=dict(
                                bgcolor="#272D3F",
                                angularaxis=dict(linewidth=1, linecolor='#3A4566'),
                                radialaxis=dict(gridcolor='#3A4566')
                            ),
                            paper_bgcolor="#1E2130",
                            font=dict(color="#E6E6E6"),
                            margin=dict(l=80, r=80, t=50, b=50)
                        )
                        st.plotly_chart(fig_radar_bowl, use_container_width=True)

    # 4) Raw Data Explorer
    with tabs[3]:
        st.markdown("## Raw Data Explorer")
        create_helper_text("View and filter the underlying ball-by-ball data")
        with st.expander("About the Raw Data", expanded=False):
            st.markdown("""
            This table shows the ball-by-ball data from all matches, including:
            - Match details (file, season, innings)
            - Teams and players involved
            - Ball details (over, ball in over)
            - Runs scored and wickets
            - Dynamic pressure calculations

            Use the filters below to explore specific aspects of the data.
            """)
        col1, col2, col3 = st.columns(3)
        with col1:
            teams = sorted(
                list(set(df_all["Batting_Team"].dropna().tolist() + df_all["Bowling_Team"].dropna().tolist()))
            )
            selected_team = st.selectbox("Filter by Team", ["All Teams"] + teams)
        with col2:
            phases = sorted(df_all["Game_Phase"].dropna().unique().tolist())
            selected_phase = st.selectbox("Filter by Game Phase", ["All Phases"] + phases)
        with col3:
            pressures = sorted(df_all["DynamicPressureLabel"].dropna().unique().tolist())
            selected_pressure = st.selectbox("Filter by Pressure Level", ["All Pressure Levels"] + pressures)

        filtered_df = df_all.copy()
        if selected_team != "All Teams":
            filtered_df = filtered_df[
                (filtered_df["Batting_Team"] == selected_team) | (filtered_df["Bowling_Team"] == selected_team)
                ]
        if selected_phase != "All Phases":
            filtered_df = filtered_df[filtered_df["Game_Phase"] == selected_phase]
        if selected_pressure != "All Pressure Levels":
            filtered_df = filtered_df[filtered_df["DynamicPressureLabel"] == selected_pressure]

        st.markdown(f"### Showing {len(filtered_df):,} of {len(df_all):,} rows")
        st.dataframe(filtered_df.head(500), use_container_width=True)
        if len(filtered_df) > 500:
            st.caption("Showing first 500 rows only. Apply more filters to see specific data.")

    # Add the new Match PPI Viewer tab
    with tabs[4]:
        implement_match_ppi_viewer(df_all)

    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p>Cric Assist Analytics Dashboard | Developed with Streamlit</p>
        <p>For more information on cricket analytics, visit <a href="https://cricviz.com/" target="_blank">CricViz</a></p>
    </div>
    """, unsafe_allow_html=True)

    # Show performance insights
    display_performance_insights(df_all)

    with st.container():
        st.markdown("## Player Performance Index (PPI) Explained")
        st.markdown("### Understanding Cricket Performance Metrics")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Batting PPI")
            st.markdown("""
            **What is Batting PPI?**

            The Batting Performance Index (BatPPI) is a 0-100 score that measures how well a batter performed in a match, 
            considering both scoring and game context. Think of it as a "match impact score" that goes beyond simple statistics.

            **How it works:**
            * Starts with runs scored (like traditional stats)
            * Gives bonus points for boundaries (4s and 6s)
            * Deducts points for getting out
            * Importantly, weighs each action by the pressure at that moment - scoring under high pressure counts more!
            * Scores are then calibrated on a 0-100 scale for each match

            **Why it matters:**
            A player scoring 30 runs in a tense chase might have more impact than someone scoring 40 in an easy situation.
            BatPPI captures this difference by rewarding clutch performances.

            **How to interpret:**
            * 80-100: Exceptional, match-winning performance
            * 60-80: Very strong contribution  
            * 40-60: Solid, useful contribution
            * 20-40: Below average impact
            * 0-20: Limited impact on the match
            * NA: Not enough deliveries faced to qualify
            """)

        with col2:
            st.markdown("#### Bowling PPI")
            st.markdown("""
            **What is Bowling PPI?**

            The Bowling Performance Index (BowlPPI) is a 0-100 score that measures a bowler's match impact, 
            accounting for wickets taken, runs conceded, and the match situation.

            **How it works:**
            * Awards points for taking wickets (the primary objective)
            * Deducts points for conceding runs
            * Gives small bonuses for dot balls (deliveries with no runs)
            * Multiplies each delivery's value by the pressure level - a wicket in a tense situation is worth more!
            * Calibrated on a 0-100 scale for each match

            **Why it matters:**
            Traditional bowling figures like "2/25" don't tell you when those wickets came or how important they were. 
            BowlPPI recognizes that a wicket in the final over of a close game often has more impact than one in a low-pressure situation.

            **How to interpret:**
            * 80-100: Game-changing bowling performance
            * 60-80: Highly effective spell
            * 40-60: Solid contribution
            * 20-40: Below average impact
            * 0-20: Limited effectiveness
            * NA: Not enough deliveries bowled to qualify
            """)

        st.info("""
        **Note:** PPI values are relative to other performances in the same match, making them ideal for 
        comparing player contributions within a game. A player can only receive a PPI score if they faced/bowled
        a minimum number of deliveries.
        """)

    # Add the pressure analysis tab to your app
    # This might replace an existing tab or be added as a new one
    with st.expander("Pressure Analysis", expanded=False):
        implement_pressure_analysis_tab(df_all)


def get_match_metadata(file_path):
    """Extract match metadata from a cricket match JSON file"""
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


def build_batting_scorecard(df_inn, bat_ppi_df=None, min_balls_for_ppi=5):
    """Generate a batting scorecard from innings data"""
    if df_inn.empty:
        return pd.DataFrame()

    # Get batting data, ensuring we only count striker deliveries
    bat_data = df_inn.dropna(subset=["Batter"]).copy()

    # Get dismissal information for each batter
    dismissal_info = {}
    for batter in bat_data["Batter"].unique():
        batter_data = bat_data[(bat_data["Batter"] == batter) & (bat_data["BatterDismissed"] == 1)]
        if not batter_data.empty:
            # Get the last dismissal for this batter
            last_dismissal = batter_data.iloc[-1]
            mode = last_dismissal.get("Mode_Of_Dismissal", "")
            bowler = last_dismissal.get("Bowler", "")
            if mode == "caught":
                fielder = last_dismissal.get("Fielder", "")
                dismissal_info[batter] = f"c {fielder} b {bowler}"
            elif mode == "bowled":
                dismissal_info[batter] = f"b {bowler}"
            elif mode == "lbw":
                dismissal_info[batter] = f"lbw b {bowler}"
            elif mode == "run out":
                dismissal_info[batter] = "run out"
            elif mode == "stumped":
                dismissal_info[batter] = f"st b {bowler}"
            else:
                dismissal_info[batter] = mode
        else:
            dismissal_info[batter] = "not out"

    # Get batting stats - only count when player is striker
    bat_stats = bat_data[bat_data["DeliveryType"] == "striker"].groupby("Batter").agg(
        Runs=("Runs_Batter", "sum"),
        Balls=("IsBattingDelivery", "sum"),
        Fours=("Runs_Batter", lambda x: (x == 4).sum()),
        Sixes=("Runs_Batter", lambda x: (x == 6).sum())
    ).reset_index()

    # Add dismissal info
    bat_stats["Dismissal"] = bat_stats["Batter"].map(dismissal_info)

    # Calculate Strike Rate
    bat_stats["StrikeRate"] = (bat_stats["Runs"] / bat_stats["Balls"] * 100).round(2)

    # Add PPI data if available
    if bat_ppi_df is not None:
        bat_stats = bat_stats.merge(bat_ppi_df[["Batter", "BatPPI"]], on="Batter", how="left")
        # Leave empty PPI for players with few balls
        bat_stats.loc[bat_stats["Balls"] < min_balls_for_ppi, "BatPPI"] = pd.NA
        bat_stats["BatPPI"] = bat_stats["BatPPI"].round(0).astype('Int64')

    # Determine batting order based on the order they appear in the dataframe
    # This assumes batters are recorded in order of appearance
    batting_order = {}
    seen_batters = set()

    # Go through each ball in the innings in chronological order
    for _, row in df_inn.sort_values(["Over", "Ball_In_Over"]).iterrows():
        batter = row.get("Batter")
        if batter and batter not in seen_batters:
            batting_order[batter] = len(seen_batters)
            seen_batters.add(batter)

    # Sort by batting order
    bat_stats["BattingOrder"] = bat_stats["Batter"].map(batting_order)
    bat_stats = bat_stats.sort_values("BattingOrder")

    return bat_stats


def build_bowling_scorecard(df_inn, bowl_ppi_df=None, min_balls_for_ppi=6):
    """Generate a bowling scorecard from innings data"""
    if df_inn.empty:
        return pd.DataFrame()

    # Filter to valid bowling deliveries (striker only to avoid double counting)
    bowl_data = df_inn[(df_inn["DeliveryType"] == "striker") & (~df_inn["Bowler"].isna())].copy()

    # Calculate maidens
    maiden_overs = bowl_data.groupby(["Bowler", "Over"])["Bowler_Runs"].sum().reset_index()
    maidens_by_bowler = maiden_overs[maiden_overs["Bowler_Runs"] == 0].groupby("Bowler").size().reset_index(
        name="Maidens")

    # Aggregate bowling stats
    bowl_stats = bowl_data.groupby("Bowler").agg(
        Balls=("IsLegalDelivery", "sum"),
        Runs=("Bowler_Runs", "sum"),
        Wickets=("Wicket", "sum")
    ).reset_index()

    # Merge in maidens
    bowl_stats = bowl_stats.merge(maidens_by_bowler, on="Bowler", how="left")
    bowl_stats["Maidens"] = bowl_stats["Maidens"].fillna(0).astype(int)

    # Calculate overs and economy
    bowl_stats["Overs"] = (bowl_stats["Balls"] // 6) + (bowl_stats["Balls"] % 6) / 10
    bowl_stats["Economy"] = (bowl_stats["Runs"] / (bowl_stats["Balls"] / 6)).round(2)

    # Add PPI data if available
    if bowl_ppi_df is not None:
        bowl_stats = bowl_stats.merge(bowl_ppi_df[["Bowler", "BowlPPI"]], on="Bowler", how="left")
        # Leave empty PPI for bowlers with few balls
        bowl_stats.loc[bowl_stats["Balls"] < min_balls_for_ppi, "BowlPPI"] = pd.NA
        bowl_stats["BowlPPI"] = bowl_stats["BowlPPI"].round(0).astype('Int64')

    # Order based on wickets and runs
    bowl_stats = bowl_stats.sort_values(["Wickets", "Runs"], ascending=[False, True])

    return bowl_stats[["Bowler", "Overs", "Maidens", "Runs", "Wickets", "Economy", "BowlPPI"]]


def implement_match_ppi_viewer(df_all):
    """Implement the Match PPI Viewer tab"""
    st.markdown("## Match PPI Viewer")
    
    # Helper text
    st.markdown("""
    <div class="helper-text">
    Select a match to view detailed scorecards with Player Performance Index (PPI) metrics and pressure timelines.
    The pressure timeline shows how batting and bowling pressure evolve throughout the innings with annotations at key change points.
    </div>
    """, unsafe_allow_html=True)

    # 1. Let user select a match
    match_files = sorted(df_all["Match_File"].unique().tolist())
    
    # Handle case where no matches are found
    if not match_files:
        st.error("No matches found in the dataset.")
        return
        
    selected_match_file = st.selectbox(
        "Select a Match",
        match_files,
        format_func=lambda x: f"Match ID: {x.split('.')[0]}"
    )

    # 2. Get data for this match
    match_df = df_all[df_all["Match_File"] == selected_match_file].copy()
    
    if match_df.empty:
        st.warning(f"No data found for match {selected_match_file}.")
        return

    # 3. Calculate PPIs for this specific match
    bat_ppi_df = compute_match_ppi_batting(match_df)
    bowl_ppi_df = compute_match_ppi_bowling(match_df)

    # Get match metadata
    match_path = os.path.join("data", selected_match_file)
    if os.path.exists(match_path):
        meta = get_match_metadata(match_path)
        # Create a styled match header
        st.markdown(f"""
        <div style="background-color: #1E2130; padding: 15px; border-radius: 10px; border: 1px solid #2D3250; margin-bottom: 20px;">
            <h3 style="margin-bottom: 10px; color: white;">{meta['teams'][0]} vs {meta['teams'][1]}</h3>
            <p style="margin-bottom: 5px;"><strong>Date:</strong> {meta['date']}</p>
            <p style="margin-bottom: 5px;"><strong>Venue:</strong> {meta['venue']}, {meta['city']}</p>
            <p style="margin-bottom: 0px;"><strong>Format:</strong> {meta['match_type'].upper()}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 4. Create tabs for each innings
    innings_list = sorted(match_df["Innings_Index"].unique())
    
    # Create tabs for the innings
    inn_tabs = st.tabs([f"Innings {i}" for i in innings_list])
    
    # Process each innings
    for i, inn_idx in enumerate(innings_list):
        df_inn = match_df[match_df["Innings_Index"] == inn_idx].copy()
        
        with inn_tabs[i]:
            # NEW: Add pressure timeline for this innings
            st.markdown("### Pressure Timeline")
            
            # Check if we have the new separate pressure labels
            has_separate_labels = "BattingPressureLabel" in df_inn.columns and "BowlingPressureLabel" in df_inn.columns
            
            if has_separate_labels:
                pressure_fig = plot_pressure_timeline(df_inn)
                st.plotly_chart(pressure_fig, use_container_width=True)
            else:
                st.warning("Separate batting and bowling pressure labels not found. Please ensure you've updated your pressure calculation.")
            
            # Add horizontal divider
            st.markdown("---")
            
            # Display batting and bowling scorecard (your existing code)
            batting_team = df_inn["Batting_Team"].iloc[0] if "Batting_Team" in df_inn.columns else f"Team (Innings {inn_idx})"
            bowling_team = df_inn["Bowling_Team"].iloc[0] if "Bowling_Team" in df_inn.columns else "Opposition"
            
            # Calculate innings total
            df_striker = df_inn[df_inn["DeliveryType"] == "striker"]
            total_runs = df_striker["Runs_Total"].sum()
            total_wickets = len(df_striker[df_striker["Wicket"] == 1])
            
            # Fix the overs calculation - only count legal deliveries from striker
            total_legal_balls = df_striker["IsLegalDelivery"].sum() if "IsLegalDelivery" in df_striker.columns else len(df_striker)
            total_overs = total_legal_balls // 6 + (total_legal_balls % 6) / 10
            
            # Create styled innings header
            st.markdown(f"""
            <div style="background-color: #272D3F; padding: 12px; border-radius: 8px; margin: 15px 0;">
                <h4 style="margin-bottom: 5px;">Innings {inn_idx}: {batting_team}</h4>
                <p style="font-size: 1.1em; margin-bottom: 0;">{total_runs}/{total_wickets} in {total_overs:.1f} overs</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display batting and bowling tabs for scorecards
            bat_bowl_tabs = st.tabs(["Batting Scorecard", "Bowling Scorecard"])
            
            with bat_bowl_tabs[0]:
                # Create and display batting scorecard
                st.markdown("#### Batting")
                bat_scorecard = build_batting_scorecard(df_inn, bat_ppi_df) if "build_batting_scorecard" in globals() else pd.DataFrame()
                if not bat_scorecard.empty:
                    st.dataframe(bat_scorecard, use_container_width=True)
                else:
                    st.info("No batting data available for this innings.")
            
            with bat_bowl_tabs[1]:
                # Create and display bowling scorecard
                st.markdown("#### Bowling")
                bowl_scorecard = build_bowling_scorecard(df_inn, bowl_ppi_df) if "build_bowling_scorecard" in globals() else pd.DataFrame()
                if not bowl_scorecard.empty:
                    st.dataframe(bowl_scorecard, use_container_width=True)
                else:
                    st.info("No bowling data available for this innings.")


def assign_separate_t20_pressure_labels(df_innings, first_innings_total=None):
    """
    For T20 cricket, assign separate BattingPressureLabel and BowlingPressureLabel
    using a 3-level scale (Low, Medium, High) with smoothing to create
    organic transitions between pressure states.
    
    Args:
        df_innings: DataFrame containing ball-by-ball data for ONE innings
        first_innings_total: Total runs from the 1st innings (if in 2nd innings)
        
    Returns:
        DataFrame with new pressure columns added
    """
    if df_innings.empty:
        return df_innings

    if df_innings["Match_Type"].iloc[0] != "T20":
        return df_innings
    
    # Make a working copy
    df = df_innings.copy()
    
    # Calculate continuous pressure scores
    batting_scores = []
    bowling_scores = []
    
    for _, row in df.iterrows():
        # Calculate batting pressure
        bat_score = compute_continuous_batting_pressure(row, first_innings_total)
        batting_scores.append(bat_score)
        
        # Calculate bowling pressure
        bowl_score = compute_continuous_bowling_pressure(row, first_innings_total)
        bowling_scores.append(bowl_score)
    
    # Add scores to the dataframe
    df["RawBatPressureScore"] = batting_scores
    df["RawBowlPressureScore"] = bowling_scores
    
    # Apply smoothing with rolling window
    window_size = 3  # Use 3-ball window for smoothing
    
    # Group by striker/non-striker deliveries to avoid mixing delivery types
    smoothed_bat_scores = []
    smoothed_bowl_scores = []
    
    for delivery_type, group in df.groupby("DeliveryType"):
        # Apply smoothing within each delivery type group
        group = group.sort_index()  # Ensure original order
        
        # Smooth the pressure scores
        bat_smooth = group["RawBatPressureScore"].rolling(window=window_size, min_periods=1).mean()
        bowl_smooth = group["RawBowlPressureScore"].rolling(window=window_size, min_periods=1).mean()
        
        # Collect the smoothed scores in order
        for idx in group.index:
            bat_idx = list(group.index).index(idx)
            bowl_idx = bat_idx
            
            smoothed_bat_scores.append(bat_smooth.iloc[bat_idx])
            smoothed_bowl_scores.append(bowl_smooth.iloc[bowl_idx])
    
    # Add smoothed scores to dataframe
    df["BatPressureScore"] = smoothed_bat_scores
    df["BowlPressureScore"] = smoothed_bowl_scores
    
    # Map continuous scores to discrete labels
    df["BattingPressureLabel"] = df["BatPressureScore"].apply(get_pressure_label)
    df["BowlingPressureLabel"] = df["BowlPressureScore"].apply(get_pressure_label)
    
    # For backward compatibility
    df["DynamicPressureLabel"] = df["BattingPressureLabel"]
    df["DynamicPressureScore"] = df["BatPressureScore"]
    
    return df

def compute_continuous_batting_pressure(row, first_innings_total=None):
    """
    Calculate a continuous batting pressure score based on match situation.
    
    Args:
        row: Single row from ball-by-ball dataframe
        first_innings_total: Total from first innings (for chases)
        
    Returns:
        Continuous pressure score (roughly 1.0=Low, 2.0=Medium, 3.0=High)
    """
    # Initialize base pressure
    base_pressure = 1.0  # Start with Low
    
    # Extract key metrics
    ball_index = row["Over"] + (row["Ball_In_Over"] - 1) / 6.0
    innings_idx = row["Innings_Index"]
    
    # Extract context metrics (handling None/NaN values safely)
    wickets_fallen = row["Wickets_Fallen"] if "Wickets_Fallen" in row and pd.notna(row["Wickets_Fallen"]) else 0
    # Calculate wickets in hand early so it's available in all code paths
    wickets_in_hand = 10 - wickets_fallen
    
    partnership = row["Current_Partnership"] if "Current_Partnership" in row and pd.notna(row["Current_Partnership"]) else 0
    wickets_in_3 = row["Wickets_In_Last_3_Overs"] if "Wickets_In_Last_3_Overs" in row and pd.notna(row["Wickets_In_Last_3_Overs"]) else 0
    crr = row["Current_Run_Rate"] if "Current_Run_Rate" in row and pd.notna(row["Current_Run_Rate"]) else 0.0
    rrr = row["Required_Run_Rate"] if "Required_Run_Rate" in row and pd.notna(row["Required_Run_Rate"]) else 0.0
    
    # 1) POWERPLAY RULES (applies to both innings)
    if ball_index < 6.0:
        # First 3 overs is high pressure (setting the tone)
        if ball_index < 3.0:
            base_pressure = 3.0  # High
        else:
            base_pressure = 2.5  # Medium-High
        
        # Losing wickets in powerplay increases pressure
        if wickets_fallen >= 2:
            base_pressure += 0.5
        
        # Recent wickets in powerplay create high pressure
        if wickets_in_3 >= 2:
            base_pressure += 0.5
    
    # 2) MIDDLE OVERS (6-16)
    elif 6.0 <= ball_index < 16.0:
        # Default to Medium pressure in middle overs
        base_pressure = 2.0  # Medium
        
        # Adjust based on wickets lost
        wicket_factor = min(1.0, wickets_fallen / 6.0)  # Scale up to 1.0 as wickets increase
        base_pressure += wicket_factor * 0.5
        
        # Recent wickets in middle overs
        if wickets_in_3 >= 2:
            base_pressure += 0.4
        
        # Partnership building reduces pressure
        if partnership > 40:
            base_pressure -= 0.5
        elif partnership > 25:
            base_pressure -= 0.3
    
    # 3) DEATH OVERS (16-20)
    else:
        # Default to High pressure at death
        base_pressure = 2.8  # High
        
        # Adjust based on wickets left
        if wickets_in_hand <= 3:
            base_pressure += 0.2  # Very high pressure with few wickets
        elif wickets_in_hand >= 8:
            base_pressure -= 0.3  # Less pressure with many wickets
    
    # 4) CHASE-SPECIFIC ADJUSTMENTS
    if innings_idx == 2 and pd.notna(rrr):
        # High RRR creates high pressure
        if rrr > 12:
            base_pressure += 0.5
        elif rrr > 9:
            base_pressure += 0.3
        
        # Lagging behind required rate
        if pd.notna(crr) and (rrr - crr) > 3:
            base_pressure += 0.4
        elif pd.notna(crr) and (rrr - crr) > 1.5:
            base_pressure += 0.2
        
        # Very easy chase reduces pressure
        if rrr < 6 and wickets_in_hand >= 7:
            base_pressure -= 0.7
    
    # Ensure pressure stays within logical bounds (1.0 to 3.0)
    return max(1.0, min(3.0, base_pressure))

def compute_continuous_bowling_pressure(row, first_innings_total=None):
    """
    Calculate a continuous bowling pressure score based on match situation.
    
    Args:
        row: Single row from ball-by-ball dataframe
        first_innings_total: Total from first innings (for chases)
        
    Returns:
        Continuous pressure score (roughly 1.0=Low, 2.0=Medium, 3.0=High)
    """
    # Initialize base pressure
    base_pressure = 1.0  # Start with Low
    
    # Extract key metrics
    ball_index = row["Over"] + (row["Ball_In_Over"] - 1) / 6.0
    innings_idx = row["Innings_Index"]
    
    # Extract context metrics (handling None/NaN values safely)
    wickets_fallen = row["Wickets_Fallen"] if "Wickets_Fallen" in row and pd.notna(row["Wickets_Fallen"]) else 0
    wickets_in_hand = 10 - wickets_fallen  # Calculate early for consistency
    
    partnership = row["Current_Partnership"] if "Current_Partnership" in row and pd.notna(row["Current_Partnership"]) else 0
    crr = row["Current_Run_Rate"] if "Current_Run_Rate" in row and pd.notna(row["Current_Run_Rate"]) else 0.0
    rrr = row["Required_Run_Rate"] if "Required_Run_Rate" in row and pd.notna(row["Required_Run_Rate"]) else 0.0
    
    # 1) POWERPLAY RULES (applies to both innings)
    if ball_index < 6.0:
        # Entire powerplay is high pressure for bowlers
        base_pressure = 2.8  # High
        
        # First 3 overs is extreme pressure (setting the tone)
        if ball_index < 3.0:
            base_pressure += 0.2
    
    # 2) MIDDLE OVERS (6-16)
    elif 6.0 <= ball_index < 16.0:
        # Default to Medium pressure in middle overs
        base_pressure = 1.8  # Medium
        
        # Adjust for long partnerships (harder for bowlers)
        if partnership > 50:
            base_pressure += 0.7
        elif partnership > 30:
            base_pressure += 0.4
        
        # High run rate creates pressure for bowlers
        if pd.notna(crr) and crr > 9:
            base_pressure += 0.4
        elif pd.notna(crr) and crr > 7:
            base_pressure += 0.2
    
    # 3) DEATH OVERS (16-20)
    else:
        # Death overs are high pressure by default
        base_pressure = 2.7
        
        # Adjust based on wickets taken
        if wickets_fallen <= 5:
            base_pressure += 0.2  # More pressure with many batters left
        
        # First innings: always high pressure at death
        if innings_idx == 1:
            base_pressure = max(base_pressure, 2.8)
    
    # 4) CHASE-SPECIFIC ADJUSTMENTS for bowling
    if innings_idx == 2 and pd.notna(rrr):
        # Tight chase creates high pressure for bowlers
        if 7 <= rrr <= 10:
            base_pressure = max(base_pressure, 2.8)  # High
        
        # Very high or very low RRR means less pressure for bowlers
        if rrr > 15:
            base_pressure -= 0.5  # Much easier to defend
        elif rrr < 6:
            base_pressure -= 0.3  # Batters likely to play safe
            
        # Last 4 overs of tight chase
        if ball_index >= 16 and 7 <= rrr <= 12:
            base_pressure = 3.0  # Maximum pressure
    
    # Ensure pressure stays within logical bounds (1.0 to 3.0)
    return max(1.0, min(3.0, base_pressure))

def get_pressure_label(score):
    """Map a continuous pressure score to a discrete label"""
    if score < 1.67:
        return "Low"
    elif score < 2.33:
        return "Medium"
    else:
        return "High"

def implement_pressure_analysis_tab(df_all):
    """
    Create a pressure analysis tab with separate batting and bowling pressure visualizations.
    """
    st.markdown("## Pressure Analysis")
    
    # Create tabs for Batting Pressure and Bowling Pressure
    tab1, tab2 = st.tabs(["Batting Pressure", "Bowling Pressure"])
    
    # Add a diagnostic check button
    if st.button("Check Pressure Columns (Diagnostic)"):
        check_pressure_columns(df_all)
    
    with tab1:
        st.markdown("### Batting Pressure Analysis")
        
        # Verify the column exists before attempting to use it
        if "BattingPressureLabel" in df_all.columns:
            # Filter to only striker deliveries for batting analysis
            bat_df = df_all[df_all["DeliveryType"] == "striker"].copy()
            
            # Create a distribution of batting pressure
            bat_pressure_counts = bat_df["BattingPressureLabel"].value_counts().reset_index()
            bat_pressure_counts.columns = ["Pressure", "Count"]
            
            # Order the pressure levels
            pressure_level_order = ["Low", "Medium", "High", "Extreme"]
            bat_pressure_counts["Pressure"] = pd.Categorical(
                bat_pressure_counts["Pressure"],
                categories=pressure_level_order,
                ordered=True
            )
            bat_pressure_counts = bat_pressure_counts.sort_values("Pressure")
            
            # Plot batting pressure distribution
            fig = px.bar(
                bat_pressure_counts,
                x="Pressure",
                y="Count",
                color="Pressure",
                color_discrete_map={
                    "Low": "#4CAF50",
                    "Medium": "#2196F3",
                    "High": "#FF9800",
                    "Extreme": "#F44336"
                },
                title="Distribution of Batting Pressure"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display batting metrics by pressure - use the specific batting pressure and get the aggregated view
            bat_metrics = compute_batting_metrics_by_pressure(df_all, pressure_col="BattingPressureLabel", aggregate=True)
            if not bat_metrics.empty:
                st.markdown("### Batting Performance by Pressure Level")
                st.dataframe(bat_metrics)
        else:
            st.warning("Batting pressure labels not found in the data. Please ensure the pressure calculation is working correctly.")
    
    with tab2:
        st.markdown("### Bowling Pressure Analysis")
        
        # Verify the column exists before attempting to use it
        if "BowlingPressureLabel" in df_all.columns:
            # Filter to only striker deliveries for bowling analysis
            bowl_df = df_all[df_all["DeliveryType"] == "striker"].copy()
            
            # Create a distribution of bowling pressure
            bowl_pressure_counts = bowl_df["BowlingPressureLabel"].value_counts().reset_index()
            bowl_pressure_counts.columns = ["Pressure", "Count"]
            
            # Order the pressure levels
            pressure_level_order = ["Low", "Medium", "High", "Extreme"]
            bowl_pressure_counts["Pressure"] = pd.Categorical(
                bowl_pressure_counts["Pressure"],
                categories=pressure_level_order,
                ordered=True
            )
            bowl_pressure_counts = bowl_pressure_counts.sort_values("Pressure")
            
            # Plot bowling pressure distribution
            fig = px.bar(
                bowl_pressure_counts,
                x="Pressure",
                y="Count",
                color="Pressure",
                color_discrete_map={
                    "Low": "#4CAF50",
                    "Medium": "#2196F3",
                    "High": "#FF9800",
                    "Extreme": "#F44336"
                },
                title="Distribution of Bowling Pressure"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display bowling metrics by pressure - use the specific bowling pressure and get the aggregated view
            bowl_metrics = compute_bowling_metrics_by_pressure(df_all, pressure_col="BowlingPressureLabel", aggregate=True)
            if not bowl_metrics.empty:
                st.markdown("### Bowling Performance by Pressure Level")
                st.dataframe(bowl_metrics)
        else:
            st.warning("Bowling pressure labels not found in the data. Please ensure the pressure calculation is working correctly.")


def check_pressure_columns(df):
    """
    Diagnostic function to verify pressure columns exist in the DataFrame.
    Shows value counts and sample records with pressure labels.
    """
    pressure_cols = [
        "BattingPressureLabel", 
        "BowlingPressureLabel", 
        "DynamicPressureLabel", 
        "DynamicPressureScore"
    ]
    
    st.write("### Pressure Columns Diagnostic")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### Columns Present")
        for col in pressure_cols:
            if col in df.columns:
                st.success(f"✅ {col}")
            else:
                st.error(f"❌ {col}")
    
    with col2:
        st.write("#### Pressure Distribution")
        for col in pressure_cols:
            if col in df.columns:
                counts = df[col].value_counts().to_dict()
                st.write(f"**{col}**")
                st.write(counts)
    
    # Display sample data with pressure columns
    st.write("### Sample Data with Pressure Labels")
    
    if not df.empty:
        # Filter to only include pressure columns that exist
        pressure_cols_existing = [col for col in pressure_cols if col in df.columns]
        
        # Only display if we have at least one pressure column
        if pressure_cols_existing:
            display_cols = ["Match_File", "Innings_Index", "Over", "Ball_In_Over"] + pressure_cols_existing
            
            # Show samples with high/extreme pressure for more interesting examples
            high_pressure_sample = df[df[pressure_cols_existing[0]].isin(["High", "Extreme"])].sample(min(10, len(df)))
            if not high_pressure_sample.empty:
                st.dataframe(high_pressure_sample[display_cols], use_container_width=True)
            else:
                # Fall back to random sample if no high pressure examples
                st.dataframe(df.sample(min(10, len(df)))[display_cols], use_container_width=True)
        else:
            st.warning("No pressure columns found in the data.")


############################################
# HELPER FUNCTIONS FOR DATA PREPARATION
############################################

def add_partnership_info(df):
    """
    Add batting partnership information to each ball.
    Tracks current partnership runs and balls for each delivery.
    """
    if df.empty:
        return df

    # Let's first check the actual column names to be safe
    batter_column = "Striker" if "Striker" in df.columns else "striker" if "striker" in df.columns else None
    non_striker_column = "NonStriker" if "NonStriker" in df.columns else "non_striker" if "non_striker" in df.columns else None

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
                if pd.notna(row["Player_Out"]) and row["Player_Out"] in current_batters:
                    current_batters.remove(row["Player_Out"])

            partnerships.append(partnership_runs)

        innings_df["Current_Partnership"] = partnerships
        result_chunks.append(innings_df)

    if result_chunks:
        return pd.concat(result_chunks, ignore_index=True)
    else:
        df["Current_Partnership"] = 0
        return df


def add_rolling_stats(df):
    """
    Add rolling statistics to each ball, including:
    - Wickets_In_Last_3_Overs: Number of wickets fallen in the previous 3 overs
    - Balls_Since_Boundary: Number of balls since last boundary
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


def add_run_rate_metrics(df):
    """
    Calculate run rate metrics for each ball:
    - Current_Run_Rate: Runs per over at current point
    - Required_Run_Rate: Run rate needed to win (2nd innings)
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


# NEW: Helper function to plot pressure timeline for an innings
def plot_pressure_timeline(df_innings):
    """
    Create a timeline chart of pressure labels (batting and bowling) over the course of an innings.
    Uses the new 3-level scale (Low, Medium, High).
    """
    # Define a mapping from label to numeric value for plotting
    pressure_numeric = {"Low": 1, "Medium": 2, "High": 3}
    
    # Filter to striker deliveries and sort by ball order
    df = df_innings[df_innings["DeliveryType"] == "striker"].copy()
    df["BallIndex"] = df["Over"] + (df["Ball_In_Over"] - 1) / 6.0
    df = df.sort_values("BallIndex")
    
    # Compute cumulative score (using Runs_Total)
    df["CumulativeScore"] = df["Runs_Total"].cumsum()
    
    # Check which pressure columns exist
    bat_pressure_col = "BattingPressureLabel" if "BattingPressureLabel" in df.columns else "DynamicPressureLabel"
    bowl_pressure_col = "BowlingPressureLabel" if "BowlingPressureLabel" in df.columns else "DynamicPressureLabel"
    
    # Convert pressure labels to numbers
    df["BatPressureNum"] = df[bat_pressure_col].map(pressure_numeric)
    df["BowlPressureNum"] = df[bowl_pressure_col].map(pressure_numeric)
    
    # Create the figure with two lines (batting and bowling)
    fig = go.Figure()
    
    # Add batting pressure line - include both line and raw data scores if available
    fig.add_trace(go.Scatter(
        x=df["BallIndex"], 
        y=df["BatPressureNum"],
        mode="lines+markers",
        name="Batting Pressure",
        line=dict(color="#FF9800", width=2),
        marker=dict(size=6)
    ))
    
    # Add continuous scores as a fainter line if available
    if "BatPressureScore" in df.columns:
        # Normalize to match the 1-3 scale
        normalized_scores = (df["BatPressureScore"] - 1) / 2 + 1
        fig.add_trace(go.Scatter(
            x=df["BallIndex"], 
            y=normalized_scores,
            mode="lines",
            name="Batting Score (Continuous)",
            line=dict(color="#FF9800", width=1, dash="dot"),
            opacity=0.5
        ))
    
    # Add bowling pressure line
    fig.add_trace(go.Scatter(
        x=df["BallIndex"], 
        y=df["BowlPressureNum"],
        mode="lines+markers",
        name="Bowling Pressure",
        line=dict(color="#2196F3", width=2),
        marker=dict(size=6)
    ))
    
    # Add continuous scores as a fainter line if available
    if "BowlPressureScore" in df.columns:
        # Normalize to match the 1-3 scale
        normalized_scores = (df["BowlPressureScore"] - 1) / 2 + 1
        fig.add_trace(go.Scatter(
            x=df["BallIndex"], 
            y=normalized_scores,
            mode="lines",
            name="Bowling Score (Continuous)",
            line=dict(color="#2196F3", width=1, dash="dot"),
            opacity=0.5
        ))
    
    # Identify change points for batting and bowling pressure and add annotations
    def add_annotations(series, pressure_col, color):
        if len(series) <= 1:
            return
            
        prev_value = series.iloc[0]
        prev_label = df[pressure_col].iloc[0]
        for idx in range(1, len(series)):
            current_value = series.iloc[idx]
            current_label = df[pressure_col].iloc[idx]
            
            if current_value != prev_value:
                # Get the ball index and score at the change
                ball = df["BallIndex"].iloc[idx]
                score = int(df["CumulativeScore"].iloc[idx])
                wickets = int(df["Wickets_Fallen"].iloc[idx])
                
                # Create annotation text showing transition and score
                annotation_text = f"{prev_label}→{current_label}<br>Score: {score}/{wickets}"
                
                fig.add_annotation(
                    x=ball, 
                    y=current_value,
                    text=annotation_text,
                    showarrow=True,
                    arrowhead=2,
                    ax=0, 
                    ay=-30,
                    font=dict(color=color, size=10),
                    bgcolor="rgba(255,255,255,0.7)"
                )
                
                prev_value = current_value
                prev_label = current_label

    # Add annotations for both pressure types
    add_annotations(df["BatPressureNum"], bat_pressure_col, "#FF9800")
    add_annotations(df["BowlPressureNum"], bowl_pressure_col, "#2196F3")
    
    # Update layout: set y-axis ticks and labels (now only 3 levels)
    fig.update_yaxes(
        tickvals=[1, 2, 3],
        ticktext=["Low", "Medium", "High"],
        title_text="Pressure Level"
    )
    
    fig.update_xaxes(title_text="Ball (Over + fraction)")
    
    # Get batting and bowling teams
    batting_team = df_innings["Batting_Team"].iloc[0] if "Batting_Team" in df_innings.columns else "Team"
    bowling_team = df_innings["Bowling_Team"].iloc[0] if "Bowling_Team" in df_innings.columns else "Opposition"
    
    # Set title and styling
    fig.update_layout(
        title=f"Pressure Timeline: {batting_team} vs {bowling_team}",
        template="plotly_dark",
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


if __name__ == "__main__":
    main()