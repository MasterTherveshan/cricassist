import streamlit as st
import pandas as pd
import numpy as np
import json
import glob
import os
import plotly.express as px

# Define global constants
PRESSURE_LEVEL_ORDER = ["Low", "Medium", "High", "Extreme"]


# Helper function to order pressure levels consistently
def order_pressure_levels(df, pressure_col="DynamicPressureLabel"):
    """Apply consistent ordering to pressure level categories"""
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
def load_and_combine_data(data_folder="data"):
    """
    Reads all JSON files in the specified folder, parses them,
    and returns a combined DataFrame of ball-by-ball data.
    """
    all_files = glob.glob(os.path.join(data_folder, "*.json"))
    all_dfs = []
    for fpath in all_files:
        df_match = convert_match_json_to_ball_df(fpath)
        if not df_match.empty:
            all_dfs.append(df_match)

    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)


############################################
# 2) CALCULATE DYNAMIC PRESSURE (1ST + 2ND INNINGS)
############################################
def calculate_dynamic_pressure_for_innings(df_innings, first_innings_total=None):
    """
    Enhanced pressure logic for both innings:
      1) 1st innings => compare to par, factor in wickets lost, partnership, recent wickets
      2) 2nd innings => chase logic: runs needed, wickets in hand, overs remaining, boundaries, partnerships
    """
    if df_innings.empty:
        df_innings["DynamicPressureScore"] = np.nan
        df_innings["DynamicPressureLabel"] = None
        return df_innings

    if df_innings["Match_Type"].iloc[0] != "T20":
        df_innings["DynamicPressureScore"] = np.nan
        df_innings["DynamicPressureLabel"] = None
        return df_innings

    # Weighted parameters
    alpha = 1.2  # Run rate / scoring pressure weight
    beta = 0.8  # Wickets lost weight
    gamma = 0.5  # Recent wickets weight
    delta = 0.3  # Boundary drought weight

    # For T20
    TYPICAL_RUN_RATE = 8.0  # expected run rate

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

        # time factor => from 1.0 to ~1.5 by the 20th over
        time_factor = 1.0 + (overs_used / 40.0)

        # Additional pressure factors
        ball_drought_factor = min(row["Balls_Since_Boundary"] / 12.0, 1.0)  # Max out at 12 balls
        wicket_pressure = min(row["Wickets_In_Last_3_Overs"] / 3.0, 1.0)  # Max out at 3 wickets

        # Small partnership adds pressure
        partnership_factor = 1.0
        if row["Current_Partnership"] < 20 and wickets_cumulative > 0:
            partnership_factor = 1.2  # New partnership adds 20% more pressure

        if inn_idx == 1:
            # -------- FIRST INNINGS --------
            expected_runs_so_far = TYPICAL_RUN_RATE * overs_used
            par_deficit = expected_runs_so_far - runs_cumulative
            if par_deficit < 0:
                par_deficit = 0  # if above par, no "extra" pressure from deficit

            if wickets_in_hand <= 0 or overs_used >= 20:
                pressure_score = 0
            else:
                part1 = alpha * (par_deficit / 35.0)
                part2 = beta * (1.0 - (wickets_in_hand / 10.0))
                part3 = gamma * wicket_pressure
                part4 = delta * ball_drought_factor

                base_score = (part1 + part2 + part3 + part4) * partnership_factor
                pressure_score = base_score * time_factor

        else:
            # -------- SECOND INNINGS --------
            if not first_innings_total:
                pressure_score = np.nan
            else:
                runs_needed = first_innings_total - runs_cumulative + 1
                balls_left = 120 - legal_balls_count

                if runs_needed <= 0 or balls_left <= 0 or wickets_in_hand <= 0:
                    pressure_score = 0
                else:
                    req_run_rate = runs_needed / (balls_left / 6.0)
                    rrr_factor = req_run_rate / TYPICAL_RUN_RATE  # Normalize by typical rate

                    part1 = alpha * min(rrr_factor, 2.0)  # Cap at 2x typical rate
                    part2 = beta * (1.0 - (wickets_in_hand / 10.0))
                    part3 = gamma * wicket_pressure
                    part4 = delta * ball_drought_factor

                    # Chase getting tight
                    chase_factor = 1.0
                    if balls_left < 36 and runs_needed > 30:  # Last 6 overs, still need 30+
                        chase_factor = 1.3  # 30% more pressure in tight chase

                    base_score = (part1 + part2 + part3 + part4) * partnership_factor * chase_factor
                    pressure_score = base_score * time_factor

        scores.append(pressure_score)

    df_innings["DynamicPressureScore"] = scores

    # Label
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
    """For bowling/overs: a delivery is legal if it is not a wide or a no-ball."""
    extras = delivery.get("extras", {})
    return not ("wides" in extras or "noballs" in extras)


def convert_match_json_to_ball_df(file_path):
    with open(file_path, "r") as f:
        match_data = json.load(f)

    teams = match_data["info"].get("teams", [])
    match_type = match_data["info"].get("match_type", "T20")
    season = match_data["info"].get("season", "Unknown")

    innings_list = match_data.get("innings", [])
    if not innings_list:
        return pd.DataFrame()

    # 1st innings total
    first_inn_runs = 0
    if len(innings_list) >= 1:
        for over_dict in innings_list[0].get("overs", []):
            for delivery in over_dict.get("deliveries", []):
                first_inn_runs += delivery.get("runs", {}).get("total", 0)

    ball_data = []

    # For tracking additional metrics
    wickets_fallen = 0
    current_partnership = 0
    balls_since_boundary = 0
    wickets_in_last_3_overs = 0
    last_3_overs_balls = 0  # track balls in last 3 overs
    prev_ball_result = None
    bowler_current_wickets = {}
    bowler_current_overs = {}
    batsman_current_runs = {}

    for inn_idx, inn_data in enumerate(innings_list, start=1):
        batting_team = inn_data.get("team")
        bowling_team = [t for t in teams if t != batting_team][0] if len(teams) > 1 else None

        # Reset innings-specific tracking variables
        wickets_fallen = 0
        current_partnership = 0
        balls_since_boundary = 0
        wickets_in_last_3_overs = 0
        last_3_overs_balls = 0
        total_runs = 0
        legal_balls = 0

        for over_dict in inn_data.get("overs", []):
            over_num = over_dict.get("over", 0)
            deliveries = over_dict.get("deliveries", [])

            # Track wickets in last 3 overs
            if match_type == "T20" and over_num >= 17:  # Last 3 overs in T20
                last_3_overs_balls = 0  # Reset count for this over
            elif match_type != "T20" and over_num >= 47:  # Last 3 overs in ODI
                last_3_overs_balls = 0

            for ball_index, delivery in enumerate(deliveries):
                total_runs += delivery.get("runs", {}).get("total", 0)
                batter_runs = delivery.get("runs", {}).get("batter", 0)
                wicket_fell = bool(delivery.get("wickets"))
                legal = is_legal_delivery(delivery)

                if legal:
                    legal_balls += 1

                # For batting, count the ball if it is not a wide.
                extras = delivery.get("extras", {})
                batting_delivery = 0
                if "wides" not in extras:
                    batting_delivery = 1

                # Update wickets in last 3 overs if this delivery is in last 3 overs
                if (match_type == "T20" and over_num >= 17) or (match_type != "T20" and over_num >= 47):
                    last_3_overs_balls += 1 if legal else 0
                    if wicket_fell:
                        wickets_in_last_3_overs += 1

                # Update balls since boundary
                if batter_runs in [4, 6]:
                    balls_since_boundary = 0
                else:
                    balls_since_boundary += 1 if legal else 0

                # Update current partnership
                if wicket_fell:
                    wickets_fallen += 1
                    current_partnership = 0
                else:
                    current_partnership += batter_runs

                # Track bowler stats
                bowler = delivery.get("bowler")
                if bowler:
                    if bowler not in bowler_current_wickets:
                        bowler_current_wickets[bowler] = 0
                        bowler_current_overs[bowler] = 0

                    if legal:
                        bowler_current_overs[bowler] += 1

                    if wicket_fell:
                        bowler_current_wickets[bowler] += 1

                # Track batsman stats
                batter = delivery.get("batter")
                if batter:
                    if batter not in batsman_current_runs:
                        batsman_current_runs[batter] = 0
                    batsman_current_runs[batter] += batter_runs

                # Calculate current run rate and required run rate
                overs_completed = legal_balls / 6.0
                current_run_rate = total_runs / overs_completed if overs_completed > 0 else 0

                required_run_rate = None
                if inn_idx == 2:
                    runs_needed = first_inn_runs - total_runs + 1
                    balls_remaining = (20 if match_type == "T20" else 50) * 6 - legal_balls
                    required_run_rate = (runs_needed / (balls_remaining / 6.0)) if balls_remaining > 0 else 0

                # Determine ball result for previous_ball_result
                if ball_index > 0 or over_num > 0:
                    if wicket_fell:
                        this_ball_result = "Wicket"
                    elif batter_runs == 0 and delivery.get("extras") is None:
                        this_ball_result = "Dot"
                    elif batter_runs in [4, 6]:
                        this_ball_result = "Boundary"
                    elif delivery.get("extras") is not None:
                        this_ball_result = "Extra"
                    else:
                        this_ball_result = str(batter_runs)
                else:
                    this_ball_result = None

                # Compute bowler runs: batter runs plus penalty extras (wides and no-balls only)
                penalty_extras = extras.get("wides", 0) + extras.get("noballs", 0)
                bowler_runs = delivery.get("runs", {}).get("batter", 0) + penalty_extras

                row = {
                    "Match_File": os.path.basename(file_path),
                    "Season": season,
                    "Innings_Index": inn_idx,
                    "Batting_Team": batting_team,
                    "Bowling_Team": bowling_team,
                    "Batter": delivery.get("batter"),
                    "Non_Striker": delivery.get("non_striker"),
                    "Bowler": delivery.get("bowler"),
                    "Over": over_num,
                    "Ball_In_Over": ball_index + 1,
                    "Runs_Total": delivery.get("runs", {}).get("total", 0),
                    "Runs_Batter": batter_runs,
                    "Bowler_Runs": bowler_runs,
                    "Wicket": wicket_fell,
                    "Match_Type": match_type,
                    "IsLegalDelivery": 1 if legal else 0,
                    "IsBattingDelivery": batting_delivery,
                    "Current_Score": f"{total_runs}-{wickets_fallen}",
                    "Wickets_Fallen": wickets_fallen,
                    "Current_Run_Rate": current_run_rate,
                    "Required_Run_Rate": required_run_rate,
                    "Current_Partnership": current_partnership,
                    "Balls_Since_Boundary": balls_since_boundary,
                    "Wickets_In_Last_3_Overs": wickets_in_last_3_overs,
                    "Previous_Ball_Result": prev_ball_result,
                    "Bowler_Current_Wickets": bowler_current_wickets.get(bowler, 0),
                    "Bowler_Current_Overs": bowler_current_overs.get(bowler, 0) / 6.0,
                    "Strike_Batsman_Runs": batsman_current_runs.get(batter, 0),
                    "Batting_Task": "Setting" if inn_idx == 1 else "Chasing",
                    "Bowling_Task": "Bowling First" if inn_idx == 1 else "Defending"
                }

                if wicket_fell and delivery.get("wickets"):
                    row["Mode_Of_Dismissal"] = delivery.get("wickets")[0].get("kind", "Unknown")
                else:
                    row["Mode_Of_Dismissal"] = None

                prev_ball_result = this_ball_result
                ball_data.append(row)

    df = pd.DataFrame(ball_data)
    if df.empty:
        return df

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
def compute_batting_metrics_by_pressure(df, pressure_col="DynamicPressureLabel"):
    if df.empty:
        return pd.DataFrame(columns=["Batter", pressure_col, "Runs", "Balls", "Dismissals", "StrikeRate", "DotBallPct"])

    sub = df.dropna(subset=["Batter"]).copy()
    # For batting, use IsBattingDelivery (which counts no-balls but excludes wides)
    sub["Balls_Faced"] = sub["IsBattingDelivery"]

    sub["DotBall"] = sub.apply(
        lambda row: 1 if (row["Runs_Batter"] == 0 and not row["Wicket"] and row["IsBattingDelivery"] == 1) else 0,
        axis=1
    )

    grouped = sub.groupby(["Batter", pressure_col], dropna=True).agg(
        Runs=("Runs_Batter", "sum"),
        Balls=("Balls_Faced", "sum"),
        Dismissals=("Wicket", "sum"),
        Dots=("DotBall", "sum")
    ).reset_index()

    grouped["StrikeRate"] = grouped.apply(
        lambda row: (row["Runs"] / row["Balls"] * 100) if row["Balls"] > 0 else 0,
        axis=1
    )
    grouped["DotBallPct"] = grouped.apply(
        lambda row: row["Dots"] / row["Balls"] if row["Balls"] > 0 else 0,
        axis=1
    )
    return grouped


def compute_bowling_metrics_by_pressure(df, pressure_col="DynamicPressureLabel"):
    if df.empty:
        return pd.DataFrame(columns=["Bowler", pressure_col, "Balls", "Runs", "Wickets", "Economy", "DotBallPct"])

    sub = df.dropna(subset=["Bowler"]).copy()
    # For bowling, legal deliveries still exclude wides and no-balls.
    sub["Balls_Bowled"] = sub["IsLegalDelivery"]

    sub["DotBall"] = sub.apply(
        lambda row: 1 if (row["Runs_Batter"] == 0 and row["IsLegalDelivery"] == 1) else 0,
        axis=1
    )

    grouped = sub.groupby(["Bowler", pressure_col], dropna=True).agg(
        Balls=("Balls_Bowled", "sum"),
        Runs=("Bowler_Runs", "sum"),
        Wickets=("Wicket", "sum"),
        Dots=("DotBall", "sum")
    ).reset_index()

    grouped["Overs"] = grouped["Balls"] / 6.0
    grouped["Economy"] = grouped.apply(lambda row: row["Runs"] / row["Overs"] if row["Overs"] > 0 else 0, axis=1)
    grouped["DotBallPct"] = grouped.apply(
        lambda row: row["Dots"] / row["Balls"] if row["Balls"] > 0 else 0,
        axis=1
    )
    return grouped


############################################
# 6) ADVANCED BATTING + BOWLING
############################################
def compute_advanced_batting_metrics(df):
    if df.empty:
        return pd.DataFrame(
            columns=["Batter", "Total_Runs", "Average", "StrikeRate", "Finisher",
                     "BoundaryRate", "DotBallPct", "StrikeRotationEfficiency",
                     "PressurePerformanceIndex"])

    sub = df.dropna(subset=["Batter"]).copy()
    sub["Balls_Faced"] = sub["IsBattingDelivery"]

    sub["Is_Boundary"] = sub.apply(
        lambda row: 1 if (row["Runs_Batter"] in [4, 6] and row["IsBattingDelivery"] == 1) else 0,
        axis=1
    )
    sub["Is_Dot"] = sub.apply(
        lambda row: 1 if (row["Runs_Batter"] == 0 and not row["Wicket"] and row["IsBattingDelivery"] == 1) else 0,
        axis=1
    )

    pressure_perf = []
    for batter, group in sub.groupby("Batter"):
        high_pressure = group[group["DynamicPressureLabel"].isin(["High", "Extreme"])]
        if not high_pressure.empty:
            runs = high_pressure["Runs_Batter"].sum()
            balls = high_pressure["IsBattingDelivery"].sum()
            pressure_sr = (runs / balls * 100) if balls > 0 else 0
        else:
            pressure_sr = 0
        pressure_perf.append({"Batter": batter, "PressurePerformanceIndex": pressure_sr})

    pressure_df = pd.DataFrame(pressure_perf)

    per_innings = sub.groupby(["Match_File", "Innings_Index", "Batter"], dropna=True).agg(
        RunsSum=("Runs_Batter", "sum"),
        BallsSum=("Balls_Faced", "sum"),
        WicketsSum=("Wicket", "sum"),
        BoundariesSum=("Is_Boundary", "sum"),
        DotsSum=("Is_Dot", "sum")
    ).reset_index()

    per_innings["FinisherInnings"] = per_innings["WicketsSum"].apply(lambda x: 1 if x == 0 else 0)

    final = per_innings.groupby("Batter").agg(
        Total_Runs=("RunsSum", "sum"),
        Balls_Faced=("BallsSum", "sum"),
        Total_Wickets=("WicketsSum", "sum"),
        Total_Boundaries=("BoundariesSum", "sum"),
        Total_Dots=("DotsSum", "sum"),
        Innings_Count=("RunsSum", "count"),
        FinisherCount=("FinisherInnings", "sum")
    ).reset_index()

    final["Average"] = final.apply(
        lambda row: row["Total_Runs"] / row["Total_Wickets"] if row["Total_Wickets"] > 0 else row["Total_Runs"], axis=1)
    final["StrikeRate"] = final.apply(
        lambda row: (row["Total_Runs"] / row["Balls_Faced"] * 100) if row["Balls_Faced"] > 0 else 0, axis=1)
    final["Finisher"] = final.apply(
        lambda row: row["FinisherCount"] / row["Innings_Count"] if row["Innings_Count"] > 0 else 0, axis=1)
    final["BoundaryRate"] = final.apply(
        lambda row: row["Total_Boundaries"] / row["Balls_Faced"] if row["Balls_Faced"] > 0 else 0, axis=1)
    final["DotBallPct"] = final.apply(
        lambda row: row["Total_Dots"] / row["Balls_Faced"] if row["Balls_Faced"] > 0 else 0, axis=1)

    final["StrikeRotationEfficiency"] = 1 - final["DotBallPct"]

    final = pd.merge(final, pressure_df, on="Batter", how="left")

    return final[["Batter", "Total_Runs", "Average", "StrikeRate", "Finisher", "BoundaryRate",
                  "DotBallPct", "StrikeRotationEfficiency", "PressurePerformanceIndex"]]


def compute_advanced_bowling_metrics(df):
    if df.empty:
        return pd.DataFrame(columns=[
            "Bowler", "Wickets", "Economy", "StrikeRate",
            "BounceBackRate", "KeyWicketIndex", "DeathOversEconomy"
        ])

    sub = df.dropna(subset=["Bowler"]).copy()
    sub["LegalDelivery"] = sub["IsLegalDelivery"]

    sub["KeyWicket"] = sub.apply(
        lambda row: 1 if row["Wicket"] and (
                (row["Match_Type"] == "T20" and row["Over"] < 10) or
                (row["Match_Type"] != "T20" and row["Over"] < 20)
        ) else 0, axis=1
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
        lambda
            x: f"{x['player']}<br>{x['Metric']}: {x['OriginalFormatted']}<br>({'Lower is better' if x['Better'] == 'Lower' else 'Higher is better'})",
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
        {f'<div style="color: #B0B0B0; font-size: 0.8rem; margin-top: 0.3rem;">{description}</div>' if description else ''}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def create_helper_text(text):
    st.markdown(f'<div class="helper-text">{text}</div>', unsafe_allow_html=True)


def create_insight_banner(text, icon="ℹ️"):
    st.markdown(f'<div class="insight-banner">{icon} <span style="font-weight: 500;">{text}</span></div>',
                unsafe_allow_html=True)


def display_performance_insights(df):
    if df.empty:
        return

    st.markdown("""
    <style>
    .insight-card {
        background-color: #1E2130;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        border: 1px solid #2D3250;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("## ⚡ Performance Insights")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Batting Under Pressure")
        bat_pressure = compute_batting_metrics_by_pressure(df, pressure_col="DynamicPressureLabel")
        if not bat_pressure.empty:
            bat_pressure = order_pressure_levels(bat_pressure)
            high_pressure = bat_pressure[bat_pressure["DynamicPressureLabel"].isin(["High", "Extreme"])]
            if not high_pressure.empty:
                min_balls = 20
                qualified = high_pressure[high_pressure["Balls"] >= min_balls]
                if not qualified.empty:
                    with st.expander("Best Strike Rates Under High Pressure", expanded=True):
                        best_sr = qualified.sort_values("StrikeRate", ascending=False).head(3)
                        for i, row in best_sr.iterrows():
                            st.markdown(
                                f"**{row['Batter']}**: <span style='color:#4FD1C5; font-weight:bold;'>{row['StrikeRate']:.2f}</span> SR ({row['Runs']} runs from {row['Balls']} balls)",
                                unsafe_allow_html=True)

                    batters_with_both = set(bat_pressure[bat_pressure["DynamicPressureLabel"] == "Low"]["Batter"]) & \
                                        set(bat_pressure[
                                                bat_pressure["DynamicPressureLabel"].isin(["High", "Extreme"])][
                                                "Batter"])
                    if batters_with_both:
                        pressure_impact = []
                        for batter in batters_with_both:
                            low_data = bat_pressure[
                                (bat_pressure["Batter"] == batter) & (bat_pressure["DynamicPressureLabel"] == "Low")]
                            high_data = bat_pressure[(bat_pressure["Batter"] == batter) & (
                                bat_pressure["DynamicPressureLabel"].isin(["High", "Extreme"]))]
                            if not low_data.empty and not high_data.empty and low_data["Balls"].iloc[0] >= 20 and \
                                    high_data["Balls"].iloc[0] >= 20:
                                low_sr = low_data["StrikeRate"].iloc[0]
                                high_sr = high_data["StrikeRate"].iloc[0]
                                impact = (high_sr - low_sr) / low_sr * 100
                                pressure_impact.append({
                                    "Batter": batter,
                                    "Low_SR": low_sr,
                                    "High_SR": high_sr,
                                    "Impact": impact
                                })
                        if pressure_impact:
                            impact_df = pd.DataFrame(pressure_impact)
                            with st.expander("Biggest Pressure Impact", expanded=True):
                                st.markdown('**Players who perform better under pressure:**')
                                improvers = impact_df.sort_values("Impact", ascending=False).head(3)
                                for i, row in improvers.iterrows():
                                    if row["Impact"] > 0:
                                        st.markdown(
                                            f"**{row['Batter']}**: <span style='color:#4FD1C5; font-weight:bold;'>+{row['Impact']:.1f}%</span> SR increase ({row['Low_SR']:.2f} → {row['High_SR']:.2f})",
                                            unsafe_allow_html=True)

                                st.markdown('**Players who struggle under pressure:**')
                                strugglers = impact_df.sort_values("Impact").head(3)
                                for i, row in strugglers.iterrows():
                                    if row["Impact"] < 0:
                                        st.markdown(
                                            f"**{row['Batter']}**: <span style='color:#F87171; font-weight:bold;'>{row['Impact']:.1f}%</span> SR decrease ({row['Low_SR']:.2f} → {row['High_SR']:.2f})",
                                            unsafe_allow_html=True)
    with col2:
        st.markdown("### Bowling Under Pressure")
        bowl_pressure = compute_bowling_metrics_by_pressure(df, pressure_col="DynamicPressureLabel")
        if not bowl_pressure.empty:
            bowl_pressure = order_pressure_levels(bowl_pressure)
            high_pressure = bowl_pressure[bowl_pressure["DynamicPressureLabel"].isin(["High", "Extreme"])]
            if not high_pressure.empty:
                min_balls = 12
                qualified = high_pressure[high_pressure["Balls"] >= min_balls]
                if not qualified.empty:
                    with st.expander("Best Economy Rates Under High Pressure", expanded=True):
                        best_economy = qualified.sort_values("Economy").head(3)
                        for i, row in best_economy.iterrows():
                            st.markdown(
                                f"**{row['Bowler']}**: <span style='color:#4FD1C5; font-weight:bold;'>{row['Economy']:.2f}</span> RPO ({row['Wickets']} wickets from {row['Balls'] / 6:.1f} overs)",
                                unsafe_allow_html=True)

                    with st.expander("Best Wicket-Takers Under High Pressure", expanded=True):
                        best_wickets = qualified.sort_values(["Wickets", "Economy"], ascending=[False, True]).head(3)
                        for i, row in best_wickets.iterrows():
                            st.markdown(
                                f"**{row['Bowler']}**: <span style='color:#4FD1C5; font-weight:bold;'>{row['Wickets']}</span> wickets at {row['Economy']:.2f} economy",
                                unsafe_allow_html=True)

                    adv_bowl = compute_advanced_bowling_metrics(df)
                    if not adv_bowl.empty:
                        qualified_bowlers = qualified["Bowler"].unique()
                        qualified_bounce_back = adv_bowl[adv_bowl["Bowler"].isin(qualified_bowlers)]
                        if not qualified_bounce_back.empty:
                            with st.expander("Best Bounce-Back Bowlers", expanded=True):
                                top_bounce_back = qualified_bounce_back.sort_values("BounceBackRate",
                                                                                    ascending=False).head(3)
                                for i, row in top_bounce_back.iterrows():
                                    st.markdown(
                                        f"**{row['Bowler']}**: <span style='color:#4FD1C5; font-weight:bold;'>{row['BounceBackRate'] * 100:.1f}%</span> bounce back rate",
                                        unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="Cric Assist - Analytics",
        page_icon="🏏",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    inject_custom_css()

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
        st.markdown("Match data in JSON format from the `data` directory.")
        st.markdown("---")
        st.caption("Version 1.0.0")
        st.caption("© 2023 Cric Assist")

    st.markdown("# 🏏 Cric Assist Analytics")

    with st.spinner("Loading and processing cricket match data..."):
        df_all = load_and_combine_data("data")

    if df_all.empty:
        st.error("### No data found")
        st.markdown("Please place your JSON files in the 'data' folder.")
        return

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
            "Use these filters to narrow down the analysis to specific seasons or players with a minimum number of games")

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
            The percentage of innings where a batter remains not out, reflecting their ability to "finish" an innings.
            Higher is better for middle/lower order batters.
            """)
        with col2:
            st.markdown("### Bounce Back Rate (BBR)")
            st.markdown("""
            For bowlers, after conceding a boundary, how often do they respond with a dot ball or wicket on the next delivery?
            Higher BBR = better recovery after being hit.
            """)
            st.markdown("### Key Wicket Index (KWI)")
            st.markdown("""
            Measures how many wickets a bowler takes in the first 10 overs, indicating early breakthrough ability.
            Higher KWI = more impact in early innings.
            """)
            st.markdown("### Death Overs Economy")
            st.markdown("""
            Economy rate specifically in the final overs (16–20) when batting acceleration typically occurs.
            Lower = better performance under pressure.
            """)

    tabs = st.tabs([
        "📊 Batting - Dynamic Pressure",
        "🎯 Bowling - Dynamic Pressure",
        "📈 Advanced Metrics + Radar",
        "🔍 Raw Data Preview"
    ])

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
                    selected_batters = st.multiselect("", all_batters, default=default_batters,
                                                      help="Select up to 5 batters for best visualization results")
                with col2:
                    create_insight_banner(
                        "Compare how strike rates and dot ball percentages change under different pressure situations.")
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
                    selected_bowlers = st.multiselect("", all_bowlers, default=default_bowlers,
                                                      help="Select up to 5 bowlers for best visualization results")
                with col2:
                    create_insight_banner(
                        "Compare how economy rates and dot ball percentages change under different pressure situations.")
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

    with tabs[2]:
        st.markdown("## Advanced Metrics & Radar Charts")
        create_helper_text("Compare players using multiple performance dimensions in radar charts")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Batting Advanced Metrics")
            adv_bat = compute_advanced_batting_metrics(df_all)
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
                sel_bat = st.multiselect("Choose up to 5 batters", all_bat, default=default_batters_adv,
                                         help="Radar chart works best with 3-5 players")
                if sel_bat:
                    metrics_batting = ["Total_Runs", "Average", "StrikeRate", "Finisher",
                                       "BoundaryRate", "StrikeRotationEfficiency", "PressurePerformanceIndex"]
                    with st.expander("Understanding the Radar Axes", expanded=False):
                        st.markdown("""
                        - **Total_Runs**: Total runs scored
                        - **Average**: Batting average (runs per dismissal)
                        - **StrikeRate**: Runs scored per 100 balls faced
                        - **Finisher**: Percentage of innings where batter remains not out
                        - **BoundaryRate**: Percentage of balls faced that result in boundaries (4s & 6s)
                        - **StrikeRotationEfficiency**: Ability to rotate strike (% of balls not played as dots)
                        - **PressurePerformanceIndex**: Strike rate during high pressure situations

                        > **Note:** All metrics are oriented so that higher values (outward extension) represent better performance.
                        """)
                    fig_radar_bat = create_radar_chart(adv_bat, "Batter", sel_bat, metrics_batting,
                                                       "Batting Comparison")
                    if fig_radar_bat:
                        fig_radar_bat.update_layout(
                            template="plotly_dark",
                            polar=dict(
                                bgcolor="#272D3F",
                                angularaxis=dict(
                                    linewidth=1,
                                    linecolor='#3A4566'
                                ),
                                radialaxis=dict(
                                    gridcolor='#3A4566'
                                )
                            ),
                            paper_bgcolor="#1E2130",
                            font=dict(color="#E6E6E6"),
                            margin=dict(l=80, r=80, t=50, b=50)
                        )
                        st.plotly_chart(fig_radar_bat, use_container_width=True)
        with col2:
            st.markdown("### Bowling Advanced Metrics")
            adv_bowl = compute_advanced_bowling_metrics(df_all)
            bowl_games = compute_bowling_games_played(df_all)
            adv_bowl = adv_bowl.merge(bowl_games, on="Bowler", how="left")
            adv_bowl = adv_bowl[adv_bowl["GamesPlayed"] >= min_games]
            if adv_bowl.empty:
                st.info("No bowlers meet the minimum games criteria.")
            else:
                display_bowl = adv_bowl.copy()
                for col in ["Economy", "StrikeRate", "DeathOversEconomy"]:
                    display_bowl[col] = display_bowl[col].round(2)
                display_bowl["BounceBackRate"] = (display_bowl["BounceBackRate"] * 100).round(1).astype(str) + '%'
                display_bowl["KeyWicketIndex"] = (display_bowl["KeyWicketIndex"] * 100).round(1).astype(str) + '%'
                st.dataframe(display_bowl, use_container_width=True)
                all_bowl = sorted(adv_bowl["Bowler"].unique().tolist())
                default_bowlers_adv = get_default_bowlers_adv(adv_bowl, n=3)
                st.markdown("### Select Bowlers for Radar")
                sel_bowl = st.multiselect("Choose up to 5 bowlers", all_bowl, default=default_bowlers_adv,
                                          help="Radar chart works best with 3-5 players")
                if sel_bowl:
                    metrics_bowling = ["Wickets", "Economy", "StrikeRate", "BounceBackRate", "KeyWicketIndex",
                                       "DeathOversEconomy"]
                    with st.expander("Understanding the Radar Axes", expanded=False):
                        st.markdown("""
                        - **Wickets**: Total wickets taken
                        - **Economy**: Runs conceded per over (lower is better)
                        - **StrikeRate**: Balls bowled per wicket (lower is better)
                        - **BounceBackRate**: Recovery rate after being hit for a boundary
                        - **KeyWicketIndex**: Proportion of wickets taken in first 10 overs
                        - **DeathOversEconomy**: Economy rate in overs 16-20 (lower is better)
                        """)
                        st.markdown("""
                        > **Note**: For Economy, StrikeRate, and DeathOversEconomy, the scales are inverted in the radar chart 
                        > (lower values shown as higher on chart) because lower values are better for these metrics.
                        """)
                    fig_radar_bowl = create_radar_chart(adv_bowl, "Bowler", sel_bowl, metrics_bowling,
                                                        "Bowling Comparison")
                    if fig_radar_bowl:
                        fig_radar_bowl.update_layout(
                            template="plotly_dark",
                            polar=dict(
                                bgcolor="#272D3F",
                                angularaxis=dict(
                                    linewidth=1,
                                    linecolor='#3A4566'
                                ),
                                radialaxis=dict(
                                    gridcolor='#3A4566'
                                )
                            ),
                            paper_bgcolor="#1E2130",
                            font=dict(color="#E6E6E6"),
                            margin=dict(l=80, r=80, t=50, b=50)
                        )
                        st.plotly_chart(fig_radar_bowl, use_container_width=True)
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
                list(set(df_all["Batting_Team"].dropna().tolist() + df_all["Bowling_Team"].dropna().tolist())))
            selected_team = st.selectbox("Filter by Team", ["All Teams"] + teams)
        with col2:
            phases = sorted(df_all["Game_Phase"].dropna().unique().tolist())
            selected_phase = st.selectbox("Filter by Game Phase", ["All Phases"] + phases)
        with col3:
            pressures = sorted(df_all["DynamicPressureLabel"].dropna().unique().tolist())
            selected_pressure = st.selectbox("Filter by Pressure Level", ["All Pressure Levels"] + pressures)
        filtered_df = df_all.copy()
        if selected_team != "All Teams":
            filtered_df = filtered_df[(filtered_df["Batting_Team"] == selected_team) |
                                      (filtered_df["Bowling_Team"] == selected_team)]
        if selected_phase != "All Phases":
            filtered_df = filtered_df[filtered_df["Game_Phase"] == selected_phase]
        if selected_pressure != "All Pressure Levels":
            filtered_df = filtered_df[filtered_df["DynamicPressureLabel"] == selected_pressure]
        st.markdown(f"### Showing {len(filtered_df):,} of {len(df_all):,} rows")
        st.dataframe(filtered_df.head(500), use_container_width=True)
        if len(filtered_df) > 500:
            st.caption("Showing first 500 rows only. Apply more filters to see specific data.")

    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p>Cric Assist Analytics Dashboard | Developed with Streamlit</p>
        <p>For more information on cricket analytics, visit <a href="https://cricviz.com/" target="_blank">CricViz</a></p>
    </div>
    """, unsafe_allow_html=True)
    display_performance_insights(df_all)


if __name__ == "__main__":
    main()
