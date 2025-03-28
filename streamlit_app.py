import streamlit as st
import pandas as pd
import numpy as np
import json
import glob
import os
import plotly.express as px

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
    Revised pressure logic for both innings:
      1) 1st innings => compare to par, factor in wickets lost & overs used
      2) 2nd innings => chase logic: runs needed, wickets in hand, overs used

    Then we apply a time_factor to bump up later overs (particularly death overs).
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
    alpha = 1.2
    beta = 0.8

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
                base_score = part1 + part2
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
                    part1 = alpha * (req_run_rate / 7.0)
                    part2 = beta * (1.0 - (wickets_in_hand / 10.0))
                    base_score = part1 + part2
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

    def is_legal_delivery(delivery):
        extras = delivery.get("extras", {})
        return not ("wides" in extras or "noballs" in extras)

    for inn_idx, inn_data in enumerate(innings_list, start=1):
        batting_team = inn_data.get("team")
        bowling_team = [t for t in teams if t != batting_team][0] if len(teams) > 1 else None

        for over_dict in inn_data.get("overs", []):
            over_num = over_dict.get("over", 0)
            deliveries = over_dict.get("deliveries", [])
            for ball_index, delivery in enumerate(deliveries):
                total_runs = delivery.get("runs", {}).get("total", 0)
                batter_runs = delivery.get("runs", {}).get("batter", 0)
                wicket_fell = bool(delivery.get("wickets"))

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
                    "Runs_Total": total_runs,
                    "Runs_Batter": batter_runs,
                    "Wicket": wicket_fell,
                    "Match_Type": match_type,
                    "IsLegalDelivery": 1 if is_legal_delivery(delivery) else 0
                }
                ball_data.append(row)

    df = pd.DataFrame(ball_data)
    if df.empty:
        return df

    # Simple phase logic
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

    # Split by innings, apply dynamic
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
    """
    Group by Batter & a chosen 'pressure' column => Runs, legitimate Balls, Dismissals, SR, DotBall%.
    """
    if df.empty:
        return pd.DataFrame(columns=["Batter", pressure_col, "Runs", "Balls", "Dismissals", "StrikeRate", "DotBallPct"])

    sub = df.dropna(subset=["Batter"]).copy()
    sub["Balls_Faced"] = sub["IsLegalDelivery"]

    # Dot ball => runs_batter=0, no wicket, islegal=1
    sub["DotBall"] = sub.apply(
        lambda row: 1 if (row["Runs_Batter"] == 0 and not row["Wicket"] and row["IsLegalDelivery"] == 1) else 0,
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
    """
    Group by Bowler & a chosen 'pressure' column => Runs, legitimate Balls, Wickets => Economy, DotBall%.
    """
    if df.empty:
        return pd.DataFrame(columns=["Bowler", pressure_col, "Balls", "Runs", "Wickets", "Economy", "DotBallPct"])

    sub = df.dropna(subset=["Bowler"]).copy()
    sub["Balls_Bowled"] = sub["IsLegalDelivery"]

    # Dot ball => runs_batter=0, islegal=1
    sub["DotBall"] = sub.apply(
        lambda row: 1 if (row["Runs_Batter"] == 0 and row["IsLegalDelivery"] == 1) else 0,
        axis=1
    )

    grouped = sub.groupby(["Bowler", pressure_col], dropna=True).agg(
        Balls=("Balls_Bowled", "sum"),
        Runs=("Runs_Total", "sum"),
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
    """
    Return typical advanced metrics:
      - Total_Runs, Avg, SR, Finisher, BoundaryRate, DotBallPct
    """
    if df.empty:
        return pd.DataFrame(
            columns=["Batter", "Total_Runs", "Average", "StrikeRate", "Finisher", "BoundaryRate", "DotBallPct"])

    sub = df.dropna(subset=["Batter"]).copy()
    sub["Balls_Faced"] = sub["IsLegalDelivery"]

    sub["Is_Boundary"] = sub.apply(
        lambda row: 1 if (row["Runs_Batter"] in [4, 6] and row["IsLegalDelivery"] == 1) else 0,
        axis=1
    )
    sub["Is_Dot"] = sub.apply(
        lambda row: 1 if (row["Runs_Batter"] == 0 and not row["Wicket"] and row["IsLegalDelivery"] == 1) else 0,
        axis=1
    )

    per_innings = sub.groupby(["Match_File", "Innings_Index", "Batter"], dropna=True).agg(
        RunsSum=("Runs_Batter", "sum"),
        BallsSum=("Balls_Faced", "sum"),
        WicketsSum=("Wicket", "sum"),
        BoundariesSum=("Is_Boundary", "sum"),
        DotsSum=("Is_Dot", "sum")
    ).reset_index()

    # FinisherInnings => innings where the batter wasn't dismissed
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

    return final[["Batter", "Total_Runs", "Average", "StrikeRate", "Finisher", "BoundaryRate", "DotBallPct"]]


def compute_advanced_bowling_metrics(df):
    """
    Return advanced metrics:
      - Wickets, Economy, StrikeRate, BounceBackRate, KeyWicketIndex, DeathOversEconomy
    """
    if df.empty:
        return pd.DataFrame(columns=["Bowler", "Wickets", "Economy", "StrikeRate", "BounceBackRate", "KeyWicketIndex", "DeathOversEconomy"])

    sub = df.dropna(subset=["Bowler"]).copy()
    sub["Balls_Bowled"] = sub["IsLegalDelivery"]
    sub.sort_values(["Bowler", "Match_File", "Innings_Index", "Over", "Ball_In_Over"], inplace=True)

    # Bounce Back Rate => boundary conceded => next legal ball is dot or wicket
    bounce_data = []
    for bowler, grp in sub.groupby("Bowler"):
        grp = grp.reset_index(drop=True)
        boundaries = 0
        bounce = 0
        for i in range(len(grp) - 1):
            if grp.loc[i, "IsLegalDelivery"] == 1:
                runs_bat = grp.loc[i, "Runs_Batter"]
                if runs_bat in [4, 6]:
                    if i + 1 < len(grp):
                        next_legal = grp.loc[i + 1, "IsLegalDelivery"]
                        next_runs = grp.loc[i + 1, "Runs_Batter"]
                        next_wicket = grp.loc[i + 1, "Wicket"]
                        if next_legal == 1:
                            if (next_runs == 0) or next_wicket:
                                bounce += 1
                            boundaries += 1
        bounce_data.append({"Bowler": bowler, "Boundaries": boundaries, "BounceBacks": bounce})

    bounce_df = pd.DataFrame(bounce_data)
    bounce_df["BounceBackRate"] = bounce_df.apply(
        lambda row: row["BounceBacks"] / row["Boundaries"] if row["Boundaries"] > 0 else 0,
        axis=1
    )

    # Key Wicket => Over < 10
    sub["Is_KeyWicket"] = sub.apply(lambda r: 1 if (r["Wicket"] == True and r["Over"] < 10) else 0, axis=1)
    # Death overs => Over >= 16
    sub["Is_DeathOver"] = sub["Over"].apply(lambda x: 1 if x >= 16 else 0)

    grouped = sub.groupby("Bowler").agg(
        Balls_Bowled=("Balls_Bowled", "sum"),
        Runs_Conceded=("Runs_Total", "sum"),
        Wickets=("Wicket", "sum"),
        KeyWickets=("Is_KeyWicket", "sum"),
        DeathBalls=("Is_DeathOver", "sum"),
        DeathRuns=("Runs_Total", lambda x: x[sub.loc[x.index, "Is_DeathOver"] == 1].sum())
    ).reset_index()

    grouped["Overs_Bowled"] = grouped["Balls_Bowled"] / 6.0
    grouped["Economy"] = grouped.apply(
        lambda row: row["Runs_Conceded"] / row["Overs_Bowled"] if row["Overs_Bowled"] > 0 else 0, axis=1)
    grouped["StrikeRate"] = grouped.apply(
        lambda row: row["Balls_Bowled"] / row["Wickets"] if row["Wickets"] > 0 else row["Balls_Bowled"], axis=1)
    grouped["KeyWicketIndex"] = grouped.apply(
        lambda row: row["KeyWickets"] / row["Wickets"] if row["Wickets"] > 0 else 0, axis=1)

    grouped["DeathOvers"] = grouped["DeathBalls"] / 6.0
    grouped["DeathOversEconomy"] = grouped.apply(
        lambda row: row["DeathRuns"] / row["DeathOvers"] if row["DeathOvers"] > 0 else 0, axis=1)

    final = pd.merge(grouped, bounce_df[["Bowler", "BounceBackRate"]], on="Bowler", how="left").fillna(0)
    return final[
        ["Bowler", "Wickets", "Economy", "StrikeRate", "BounceBackRate", "KeyWicketIndex", "DeathOversEconomy"]]


############################################
# 7) RADAR CHART
############################################
def create_radar_chart(df, player_col, player_list, metrics, title):
    """
    Creates a radar chart using min-max scaling for each metric (clipped at 5th..95th percentile).
    """
    if df.empty or not player_list:
        return None

    sub = df[df[player_col].isin(player_list)].copy()
    if sub.empty:
        return None

    # For each metric, clip outliers & scale
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
            val_scaled = (val_clipped - min_) / rng
            data_row[m] = val_scaled
        clipped_rows.append(data_row)

    radar_data = []
    for row in clipped_rows:
        for m in metrics:
            radar_data.append({
                "player": row["player"],
                "Metric": m,
                "Value": row[m]
            })

    radar_df = pd.DataFrame(radar_data)
    fig = px.line_polar(
        radar_df,
        r="Value",
        theta="Metric",
        color="player",
        line_close=True,
        template="plotly_dark",
        title=title,
        range_r=[0, 1]
    )
    fig.update_traces(fill='toself')
    fig.update_layout(
        polar=dict(radialaxis=dict(showticklabels=True, ticks='outside')),
        legend=dict(title=player_col),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    return fig


############################################
# 8) DEFAULT SELECTION HELPERS
############################################
def get_default_batters_pressure(df, n=5):
    """
    For the dynamic pressure DF (batting), pick top n by total 'Runs' across all pressure levels.
    """
    if df.empty:
        return []
    grouped = df.groupby("Batter", dropna=True)["Runs"].sum().reset_index()
    grouped = grouped.sort_values("Runs", ascending=False)
    top = grouped.head(n)["Batter"].tolist()
    return top

def get_default_bowlers_pressure(df, n=5):
    """
    For the dynamic pressure DF (bowling), pick top n by total 'Wickets' across all pressure levels.
    """
    if df.empty:
        return []
    grouped = df.groupby("Bowler", dropna=True)["Wickets"].sum().reset_index()
    grouped = grouped.sort_values("Wickets", ascending=False)
    top = grouped.head(n)["Bowler"].tolist()
    return top

def get_default_batters_adv(df, n=5):
    """
    For advanced batting metrics, pick top n by 'Total_Runs'.
    """
    if df.empty:
        return []
    top = df.sort_values("Total_Runs", ascending=False).head(n)["Batter"].tolist()
    return top

def get_default_bowlers_adv(df, n=5):
    """
    For advanced bowling metrics, pick top n by 'Wickets'.
    """
    if df.empty:
        return []
    top = df.sort_values("Wickets", ascending=False).head(n)["Bowler"].tolist()
    return top


############################################
# 9) STREAMLIT APP
############################################
def display_metric_card(title, value, description=None):
    """Display a metric in a card-like container for visual emphasis"""
    html = f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        {f'<div style="color: #B0B0B0; font-size: 0.8rem; margin-top: 0.3rem;">{description}</div>' if description else ''}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def create_helper_text(text):
    """Create styled helper text container"""
    st.markdown(f'<div class="helper-text">{text}</div>', unsafe_allow_html=True)

def create_insight_banner(text, icon="ℹ️"):
    """Create a banner for displaying key insights"""
    st.markdown(f'<div class="insight-banner">{icon} <span style="font-weight: 500;">{text}</span></div>', unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="Cric Assist - Analytics",
        page_icon="🏏",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Inject our custom CSS for a modern dashboard aesthetic
    inject_custom_css()

    # Enhanced sidebar with better structure and more information
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
        
        # Data source information
        st.markdown("#### Data Source")
        st.markdown("Match data in JSON format from the `data` directory.")
        
        # Version info in footer
        st.markdown("---")
        st.caption("Version 1.0.0")
        st.caption("© 2023 Cric Assist")

    # Main content area with improved structure and aesthetics
    st.markdown("# 🏏 Cric Assist Analytics")
    
    # Load data with a spinner to indicate progress
    with st.spinner("Loading and processing cricket match data..."):
        df_all = load_and_combine_data("data")
    
    if df_all.empty:
        st.error("### No data found")
        st.markdown("Please place your JSON files in the 'data' folder.")
        return

    # Container for key stats and filters - styled for emphasis
    with st.container():
        st.markdown("## Key Metrics & Filters")
        
        # Display key stats in metric cards at the top
        match_count = df_all["Match_File"].nunique()
        player_count = len(set(df_all["Batter"].dropna().tolist() + df_all["Bowler"].dropna().tolist()))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            display_metric_card("Matches Analyzed", f"{match_count:,}", "Total matches in dataset")
        with col2:
            display_metric_card("Players", f"{player_count:,}", "Total unique players")
        with col3:
            display_metric_card("Data Points", f"{len(df_all):,}", "Ball-by-ball records")
        
        # Improved filter section with styled container
        st.markdown("### Data Filters")
        create_helper_text("Use these filters to narrow down the analysis to specific seasons or players with a minimum number of games")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            # Filter by season with default selection and better styling
            seasons = sorted(df_all["Season"].dropna().unique().tolist())
            if seasons:
                selected_seasons = st.multiselect("Select Season(s)", seasons, default=seasons)
                df_all = df_all[df_all["Season"].isin(selected_seasons)]
        
        with col2:
            # Minimum games filter with explanation
            min_games = st.number_input("Minimum Games Played", min_value=1, value=3, 
                                        help="Only include players who have played at least this many games")
    
    # Check if data remains after filtering
    if df_all.empty:
        st.warning("No data remains after applying filters.")
        return

    # Explanation of metrics with improved styling and collapsible sections
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

    # Create tabs with enhanced styling
    tabs = st.tabs([
        "📊 Batting - Dynamic Pressure",
        "🎯 Bowling - Dynamic Pressure", 
        "📈 Advanced Metrics + Radar",
        "🔍 Raw Data Preview"
    ])

    # TAB 1: BATTING - DYNAMIC PRESSURE
    with tabs[0]:
        st.markdown("## Batting Performance Under Pressure")
        create_helper_text("Analyze how batters perform under different levels of match pressure")
        
        bat_pressure = compute_batting_metrics_by_pressure(df_all, pressure_col="DynamicPressureLabel")
        if bat_pressure.empty:
            st.info("No batting data or no dynamic pressure calculation available.")
        else:
            # Filter by min games
            bat_games = compute_batting_games_played(df_all)
            merged_bat = bat_pressure.merge(bat_games, on="Batter", how="left")
            merged_bat = merged_bat[merged_bat["GamesPlayed"] >= min_games]

            if merged_bat.empty:
                st.info("No batters meet the minimum games criteria.")
            else:
                # Create two columns for selection and explanation
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Better default: top run scorers
                    all_batters = sorted(merged_bat["Batter"].unique().tolist())
                    default_batters = get_default_batters_pressure(merged_bat, n=5)
                    st.markdown("### Select Batters to Compare")
                    selected_batters = st.multiselect("", all_batters, default=default_batters,
                                                      help="Select up to 5 batters for best visualization results")
                
                with col2:
                    create_insight_banner("Compare how strike rates and dot ball percentages change under different pressure situations.")
                
                if selected_batters:
                    sub_bat = merged_bat[merged_bat["Batter"].isin(selected_batters)]
                    
                    # Display data in a cleaner format with column customization
                    st.markdown("### Performance Metrics by Pressure Level")
                    # Format columns for better readability
                    display_df = sub_bat.copy()
                    display_df["StrikeRate"] = display_df["StrikeRate"].round(2)
                    display_df["DotBallPct"] = (display_df["DotBallPct"] * 100).round(2).astype(str) + '%'
                    
                    # Order for display
                    order = ["Low", "Medium", "High", "Extreme"]
                    display_df["DynamicPressureLabel"] = pd.Categorical(
                        display_df["DynamicPressureLabel"], 
                        categories=order, 
                        ordered=True
                    )
                    display_df = display_df.sort_values("DynamicPressureLabel")
                    st.dataframe(display_df, use_container_width=True)

                    # Charts with improved styling - with proper ordering
                    st.markdown("### Strike Rate Comparison")
                    # Create ordered category for plotting
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
                    # Same for the second chart
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

    # TAB 2: BOWLING - DYNAMIC PRESSURE
    with tabs[1]:
        st.markdown("## Bowling Performance Under Pressure")
        create_helper_text("Analyze how bowlers perform under different levels of match pressure")
        
        bowl_pressure = compute_bowling_metrics_by_pressure(df_all, pressure_col="DynamicPressureLabel")
        if bowl_pressure.empty:
            st.info("No bowling data or no dynamic pressure calculation available.")
        else:
            # Filter by min games
            bowl_games = compute_bowling_games_played(df_all)
            merged_bowl = bowl_pressure.merge(bowl_games, on="Bowler", how="left")
            merged_bowl = merged_bowl[merged_bowl["GamesPlayed"] >= min_games]

            if merged_bowl.empty:
                st.info("No bowlers meet the minimum games criteria.")
            else:
                # Create two columns for selection and explanation
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Better default: top wicket-takers
                    all_bowlers = sorted(merged_bowl["Bowler"].unique().tolist())
                    default_bowlers = get_default_bowlers_pressure(merged_bowl, n=5)
                    st.markdown("### Select Bowlers to Compare")
                    selected_bowlers = st.multiselect("", all_bowlers, default=default_bowlers,
                                                      help="Select up to 5 bowlers for best visualization results")
                
                with col2:
                    create_insight_banner("Compare how economy rates and dot ball percentages change under different pressure situations.")
                
                if selected_bowlers:
                    sub_bowl = merged_bowl[merged_bowl["Bowler"].isin(selected_bowlers)]
                    
                    # Display data in a cleaner format with column customization
                    st.markdown("### Performance Metrics by Pressure Level")
                    # Format columns for better readability
                    display_df = sub_bowl.copy()
                    display_df["Economy"] = display_df["Economy"].round(2)
                    display_df["DotBallPct"] = (display_df["DotBallPct"] * 100).round(2).astype(str) + '%'
                    
                    # Order for display
                    order = ["Low", "Medium", "High", "Extreme"]
                    display_df["DynamicPressureLabel"] = pd.Categorical(
                        display_df["DynamicPressureLabel"], 
                        categories=order, 
                        ordered=True
                    )
                    display_df = display_df.sort_values("DynamicPressureLabel")
                    st.dataframe(display_df, use_container_width=True)

                    # Charts with improved styling - with proper ordering
                    st.markdown("### Economy Rate Comparison")
                    # Create ordered category for plotting
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
                    # Same for the second chart
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

    # TAB 3: ADVANCED METRICS + RADAR
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
                # Format for display
                display_bat = adv_bat.copy()
                display_bat["Average"] = display_bat["Average"].round(2)
                display_bat["StrikeRate"] = display_bat["StrikeRate"].round(2)
                display_bat["Finisher"] = (display_bat["Finisher"] * 100).round(1).astype(str) + '%'
                display_bat["BoundaryRate"] = (display_bat["BoundaryRate"] * 100).round(1).astype(str) + '%'
                display_bat["DotBallPct"] = (display_bat["DotBallPct"] * 100).round(1).astype(str) + '%'
                
                st.dataframe(display_bat, use_container_width=True)

                all_bat = sorted(adv_bat["Batter"].unique().tolist())
                default_batters_adv = get_default_batters_adv(adv_bat, n=3)
                st.markdown("### Select Batters for Radar")
                sel_bat = st.multiselect("Choose up to 5 batters", all_bat, default=default_batters_adv,
                                          help="Radar chart works best with 3-5 players")
                
                if sel_bat:
                    metrics_batting = ["Total_Runs", "Average", "StrikeRate", "Finisher", "BoundaryRate", "DotBallPct"]
                    # Add explanation for what each axis means
                    with st.expander("Understanding the Radar Axes", expanded=False):
                        st.markdown("""
                        - **Total_Runs**: Total runs scored
                        - **Average**: Batting average (runs per dismissal)
                        - **StrikeRate**: Runs scored per 100 balls faced
                        - **Finisher**: Percentage of innings where batter remains not out
                        - **BoundaryRate**: Percentage of balls faced that result in boundaries (4s & 6s)
                        - **DotBallPct**: Percentage of balls faced that result in no runs
                        """)
                    
                    fig_radar_bat = create_radar_chart(adv_bat, "Batter", sel_bat, metrics_batting, "Batting Comparison")
                    if fig_radar_bat:
                        # Add custom styling to the radar chart
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
                # Format for display
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
                    metrics_bowling = ["Wickets", "Economy", "StrikeRate", "BounceBackRate", "KeyWicketIndex", "DeathOversEconomy"]
                    # Add explanation for what each axis means
                    with st.expander("Understanding the Radar Axes", expanded=False):
                        st.markdown("""
                        - **Wickets**: Total wickets taken
                        - **Economy**: Runs conceded per over (lower is better)
                        - **StrikeRate**: Balls bowled per wicket (lower is better)
                        - **BounceBackRate**: Recovery rate after being hit for a boundary
                        - **KeyWicketIndex**: Proportion of wickets taken in first 10 overs
                        - **DeathOversEconomy**: Economy rate in overs 16-20 (lower is better)
                        """)
                        
                        # Add note about inverted scales
                        st.markdown("""
                        > **Note**: For Economy, StrikeRate, and DeathOversEconomy, the scales are inverted in the radar chart 
                        > (lower values shown as higher on chart) because lower values are better for these metrics.
                        """)
                    
                    fig_radar_bowl = create_radar_chart(adv_bowl, "Bowler", sel_bowl, metrics_bowling, "Bowling Comparison")
                    if fig_radar_bowl:
                        # Add custom styling to the radar chart
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

    # TAB 4: RAW DATA PREVIEW
    with tabs[3]:
        st.markdown("## Raw Data Explorer")
        create_helper_text("View and filter the underlying ball-by-ball data")
        
        # Add more context about the raw data
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
        
        # Add filters for better exploration
        col1, col2, col3 = st.columns(3)
        with col1:
            teams = sorted(list(set(df_all["Batting_Team"].dropna().tolist() + df_all["Bowling_Team"].dropna().tolist())))
            selected_team = st.selectbox("Filter by Team", ["All Teams"] + teams)
        
        with col2:
            phases = sorted(df_all["Game_Phase"].dropna().unique().tolist())
            selected_phase = st.selectbox("Filter by Game Phase", ["All Phases"] + phases)
        
        with col3:
            pressures = sorted(df_all["DynamicPressureLabel"].dropna().unique().tolist())
            selected_pressure = st.selectbox("Filter by Pressure Level", ["All Pressure Levels"] + pressures)
        
        # Apply filters
        filtered_df = df_all.copy()
        if selected_team != "All Teams":
            filtered_df = filtered_df[(filtered_df["Batting_Team"] == selected_team) | 
                                     (filtered_df["Bowling_Team"] == selected_team)]
        
        if selected_phase != "All Phases":
            filtered_df = filtered_df[filtered_df["Game_Phase"] == selected_phase]
        
        if selected_pressure != "All Pressure Levels":
            filtered_df = filtered_df[filtered_df["DynamicPressureLabel"] == selected_pressure]
        
        # Display the filtered data with row count info
        st.markdown(f"### Showing {len(filtered_df):,} of {len(df_all):,} rows")
        st.dataframe(filtered_df.head(500), use_container_width=True)
        
        if len(filtered_df) > 500:
            st.caption("Showing first 500 rows only. Apply more filters to see specific data.")

    # Footer with additional resources
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p>Cric Assist Analytics Dashboard | Developed with Streamlit</p>
        <p>For more information on cricket analytics, visit <a href="https://cricviz.com/" target="_blank">CricViz</a></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
