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
    if df_innings.empty:
        df_innings["DynamicPressureScore"] = np.nan
        df_innings["DynamicPressureLabel"] = None
        return df_innings
    if df_innings["Match_Type"].iloc[0] != "T20":
        df_innings["DynamicPressureScore"] = np.nan
        df_innings["DynamicPressureLabel"] = None
        return df_innings
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
        time_factor = 1.0 + (overs_used / 40.0)
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
                base_score = (part1 + part2 + part3 + part4) * partnership_factor
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
                    base_score = (part1 + part2 + part3 + part4) * partnership_factor * chase_factor
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
    with open(file_path, "r") as f:
        match_data = json.load(f)

    teams = match_data["info"].get("teams", [])
    match_type = match_data["info"].get("match_type", "T20")
    season = match_data["info"].get("season", "Unknown")

    # Extract ID from filename
    match_id = os.path.basename(file_path).split('.')[0]
    innings_list = match_data.get("innings", [])
    if not innings_list:
        return pd.DataFrame()

    # For chase context
    first_inn_runs = 0
    if len(innings_list) >= 1:
        for over_dict in innings_list[0].get("overs", []):
            for delivery in over_dict.get("deliveries", []):
                first_inn_runs += delivery.get("runs", {}).get("total", 0)

    ball_data = []
    for inn_idx, inn_data in enumerate(innings_list, start=1):
        batting_team = inn_data.get("team")
        if len(teams) > 1:
            bowling_team = [t for t in teams if t != batting_team][0]
        else:
            bowling_team = None

        # Trackers
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

        for over_dict in inn_data.get("overs", []):
            over_num = over_dict["over"]
            deliveries = over_dict.get("deliveries", [])

            # Reset if final overs
            if match_type=="T20" and over_num>=17:
                last_3_overs_balls=0
            elif match_type!="T20" and over_num>=47:
                last_3_overs_balls=0

            for ball_index, delivery in enumerate(deliveries):
                total_runs += delivery["runs"]["total"]
                batter_runs = delivery["runs"].get("batter", 0)
                wicket_fell = bool(delivery.get("wickets"))

                extras = delivery.get("extras", {})
                is_legal = is_legal_delivery(delivery)
                if is_legal:
                    legal_balls += 1

                # Striker faced ball if not wide
                balls_faced = 1 if "wides" not in extras else 0
                if (match_type=="T20" and over_num>=17) or (match_type!="T20" and over_num>=47):
                    if is_legal:
                        last_3_overs_balls+=1
                    if wicket_fell:
                        wickets_in_last_3_overs+=1

                if batter_runs in [4,6]:
                    balls_since_boundary=0
                else:
                    if is_legal:
                        balls_since_boundary+=1

                if wicket_fell:
                    wickets_fallen+=1
                    current_partnership=0
                else:
                    current_partnership+=batter_runs

                bowler=delivery.get("bowler")
                if bowler:
                    if bowler not in bowler_current_wickets:
                        bowler_current_wickets[bowler]=0
                        bowler_current_overs[bowler]=0
                    if is_legal:
                        bowler_current_overs[bowler]+=1
                    if wicket_fell:
                        bowler_current_wickets[bowler]+=1

                batter = delivery.get("batter")
                non_striker = delivery.get("non_striker")

                if batter not in batsman_current_runs:
                    batsman_current_runs[batter]=0
                batsman_current_runs[batter]+=batter_runs

                overs_completed=legal_balls/6.0
                current_run_rate = total_runs/overs_completed if overs_completed>0 else 0
                required_run_rate=None
                if inn_idx==2:
                    runs_needed=first_inn_runs - total_runs + 1
                    balls_left=(20 if match_type=="T20" else 50)*6 - legal_balls
                    required_run_rate=(runs_needed/(balls_left/6.0)) if balls_left>0 else 0

                # Basic result
                if ball_index>0 or over_num>0:
                    if wicket_fell:
                        this_ball_result="Wicket"
                    elif batter_runs==0 and not extras:
                        this_ball_result="Dot"
                    elif batter_runs in [4,6]:
                        this_ball_result="Boundary"
                    elif extras:
                        this_ball_result="Extra"
                    else:
                        this_ball_result=str(batter_runs)
                else:
                    this_ball_result=None

                penalty_extras = extras.get("wides", 0)+extras.get("noballs", 0)
                bowler_runs = batter_runs + penalty_extras

                # Determine who got out
                wickets_info = delivery.get("wickets", [])
                striker_dismissed = any(w.get("player_out")==batter for w in wickets_info)
                non_striker_dismissed = any(w.get("player_out")==non_striker for w in wickets_info)

                # Build row for the STRIKER
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
                    "Ball_In_Over": ball_index+1,
                    "Runs_Total": delivery["runs"]["total"],
                    "Runs_Batter": batter_runs,
                    "Bowler_Runs": bowler_runs,
                    "Wicket": 1 if wicket_fell else 0,
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
                    "Bowler_Current_Wickets": bowler_current_wickets.get(bowler,0) if bowler else 0,
                    "Bowler_Current_Overs": bowler_current_overs.get(bowler,0)/6.0 if bowler else 0,
                    "Strike_Batsman_Runs": batsman_current_runs[batter],
                    "Batting_Task": "Setting" if inn_idx==1 else "Chasing",
                    "Bowling_Task": "Bowling First" if inn_idx==1 else "Defending",
                    "Balls_Faced": balls_faced,
                    "BatterDismissed": 1 if striker_dismissed else 0
                }

                if wicket_fell and wickets_info:
                    row_striker["Mode_Of_Dismissal"] = wickets_info[0].get("kind","Unknown")
                    row_striker["PlayerDismissed"] = wickets_info[0].get("player_out")
                else:
                    row_striker["Mode_Of_Dismissal"] = None
                    row_striker["PlayerDismissed"] = None

                ball_data.append(row_striker)

                # Build row for the NON-STRIKER to capture run-outs, etc.
                row_non_striker = row_striker.copy()
                row_non_striker["Batter"] = non_striker
                row_non_striker["Balls_Faced"] = 0       # non-striker isn't facing
                row_non_striker["Runs_Batter"] = 0
                row_non_striker["IsBattingDelivery"] = 0
                row_non_striker["Strike_Batsman_Runs"] = 0
                row_non_striker["BatterDismissed"] = 1 if non_striker_dismissed else 0
                # If the non-striker was run out, set their dismissal mode:
                row_non_striker["Mode_Of_Dismissal"] = None
                row_non_striker["PlayerDismissed"] = None
                if non_striker_dismissed and wickets_info:
                    for w in wickets_info:
                        if w["player_out"] == non_striker:
                            row_non_striker["Mode_Of_Dismissal"] = w.get("kind","Unknown")
                            row_non_striker["PlayerDismissed"] = non_striker
                            break

                ball_data.append(row_non_striker)
                prev_ball_result=this_ball_result

    df = pd.DataFrame(ball_data)
    if df.empty:
        return df

    # Label game phase
    def game_phase_t20(over):
        if over<6:
            return "Powerplay"
        elif over<16:
            return "Middle"
        else:
            return "Death"

    def game_phase_50(over):
        if over<10:
            return "Powerplay"
        elif over<40:
            return "Middle"
        else:
            return "Death"

    if match_type=="T20":
        df["Game_Phase"] = df["Over"].apply(game_phase_t20)
    else:
        df["Game_Phase"] = df["Over"].apply(game_phase_50)

    phase_map={"Powerplay":"Low","Middle":"Medium","Death":"High"}
    df["Pressure"]=df["Game_Phase"].map(phase_map)

    # Add dynamic pressure
    out_chunks=[]
    for (mf, idx), chunk in df.groupby(["Match_File","Innings_Index"]):
        chunk=chunk.copy()
        if idx==2:
            chunk=calculate_dynamic_pressure_for_innings(chunk,first_innings_total=first_inn_runs)
        else:
            chunk=calculate_dynamic_pressure_for_innings(chunk,first_innings_total=None)
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
    sub["Balls_Faced"] = sub["IsBattingDelivery"]
    sub["DotBall"] = sub.apply(
        lambda row: 1 if (row["Runs_Batter"] == 0 and not row["Wicket"] and row["IsBattingDelivery"] == 1) else 0,
        axis=1
    )
    grouped = sub.groupby(["Batter", pressure_col], dropna=True).agg(
        Runs=("Runs_Batter", "sum"),
        Balls=("Balls_Faced", "sum"),
        Dismissals=("BatterDismissed", "sum"),
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


def compute_advanced_batting_metrics(df):
    """Compute advanced batting metrics"""
    if df.empty:
        return pd.DataFrame()

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

        metrics.append({
            "Batter": batter,
            "Total_Runs": runs,
            "Average": avg,
            "StrikeRate": sr,
            "BoundaryRate": boundary_rate,
            "DotBallPct": dot_pct,
            "StrikeRotationEfficiency": 1 - dot_pct,  # Inverse of dot ball percentage
            "Finisher": finisher,
            "PressurePerformanceIndex": pressure_index
        })

    return pd.DataFrame(metrics)


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
                                f"**{row['Batter']}**: <span style='color:#4FD1C5; font-weight:bold;'>"
                                f"{row['StrikeRate']:.2f}</span> SR ({row['Runs']} runs from {row['Balls']} balls)",
                                unsafe_allow_html=True
                            )
                    batters_with_both = set(bat_pressure[bat_pressure["DynamicPressureLabel"] == "Low"]["Batter"]) & \
                                        set(bat_pressure[
                                                bat_pressure["DynamicPressureLabel"].isin(["High", "Extreme"])]
                                            ["Batter"])
                    if batters_with_both:
                        pressure_impact = []
                        for batter in batters_with_both:
                            low_data = bat_pressure[
                                (bat_pressure["Batter"] == batter) & (bat_pressure["DynamicPressureLabel"] == "Low")]
                            high_data = bat_pressure[
                                (bat_pressure["Batter"] == batter) &
                                (bat_pressure["DynamicPressureLabel"].isin(["High", "Extreme"]))
                                ]
                            if not low_data.empty and not high_data.empty \
                                    and low_data["Balls"].iloc[0] >= 20 and high_data["Balls"].iloc[0] >= 20:
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
                                            f"**{row['Batter']}**: "
                                            f"<span style='color:#4FD1C5; font-weight:bold;'>+{row['Impact']:.1f}%</span>"
                                            f" SR increase ({row['Low_SR']:.2f} → {row['High_SR']:.2f})",
                                            unsafe_allow_html=True
                                        )
                                st.markdown('**Players who struggle under pressure:**')
                                strugglers = impact_df.sort_values("Impact").head(3)
                                for i, row in strugglers.iterrows():
                                    if row["Impact"] < 0:
                                        st.markdown(
                                            f"**{row['Batter']}**: "
                                            f"<span style='color:#F87171; font-weight:bold;'>{row['Impact']:.1f}%</span>"
                                            f" SR decrease ({row['Low_SR']:.2f} → {row['High_SR']:.2f})",
                                            unsafe_allow_html=True
                                        )
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
                                f"**{row['Bowler']}**: <span style='color:#4FD1C5; font-weight:bold;'>"
                                f"{row['Economy']:.2f}</span> RPO ({row['Wickets']} wickets "
                                f"from {row['Balls'] / 6:.1f} overs)",
                                unsafe_allow_html=True
                            )
                    with st.expander("Best Wicket-Takers Under High Pressure", expanded=True):
                        best_wickets = qualified.sort_values(["Wickets", "Economy"], ascending=[False, True]).head(3)
                        for i, row in best_wickets.iterrows():
                            st.markdown(
                                f"**{row['Bowler']}**: <span style='color:#4FD1C5; font-weight:bold;'>"
                                f"{row['Wickets']}</span> wickets at {row['Economy']:.2f} economy",
                                unsafe_allow_html=True
                            )
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
                                        f"**{row['Bowler']}**: <span style='color:#4FD1C5; font-weight:bold;'>"
                                        f"{row['BounceBackRate'] * 100:.1f}%</span> bounce back rate",
                                        unsafe_allow_html=True
                                    )


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
        st.markdown("Match data in JSON format from the `data` directory.")
        st.markdown("---")
        st.caption("Version 1.0.0")
        st.caption("© 2023 Cric Assist")

    st.markdown("# 🏏 Cric Assist Analytics")

    # Load data
    with st.spinner("Loading and processing cricket match data..."):
        df_all = load_and_combine_data("data")
    if df_all.empty:
        st.error("### No data found")
        st.markdown("Please place your JSON files in the 'data' folder.")
        return

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
            Measures how many wickets a bowler takes in the first 10 overs, 
            indicating early breakthrough ability. Higher KWI = more impact early.
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
        "🔍 Raw Data Preview"
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
                sel_bat = st.multiselect(
                    "Choose up to 5 batters",
                    all_bat,
                    default=default_batters_adv,
                    help="Radar chart works best with 3-5 players"
                )
                if sel_bat:
                    metrics_batting = ["Total_Runs", "Average", "StrikeRate", "Finisher",
                                       "BoundaryRate", "StrikeRotationEfficiency", "PressurePerformanceIndex"]
                    with st.expander("Understanding the Radar Axes", expanded=False):
                        st.markdown("""
                        - **Total_Runs**: Total runs scored
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
            adv_bowl = compute_advanced_bowling_metrics(df_all)
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
                    metrics_bowling = ["Wickets", "Economy", "StrikeRate", "BounceBackRate", "KeyWicketIndex",
                                       "DeathOversEconomy"]
                    with st.expander("Understanding the Radar Axes", expanded=False):
                        st.markdown("""
                        - **Wickets**: Total wickets taken
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

    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p>Cric Assist Analytics Dashboard | Developed with Streamlit</p>
        <p>For more information on cricket analytics, visit <a href="https://cricviz.com/" target="_blank">CricViz</a></p>
    </div>
    """, unsafe_allow_html=True)

    # Show performance insights
    display_performance_insights(df_all)


if __name__ == "__main__":
    main()
