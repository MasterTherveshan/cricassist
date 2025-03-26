import streamlit as st
import pandas as pd
import numpy as np
import json
import glob
import os
import plotly.express as px


##############################
# 1) LOAD & COMBINE JSON FILES
##############################

def load_and_combine_data(data_folder="data"):
    """
    Reads all JSON files in the specified folder, parses them,
    and returns a combined DataFrame of ball-by-ball data.
    """
    all_files = glob.glob(os.path.join(data_folder, "*.json"))
    dfs = []
    for fpath in all_files:
        df = convert_match_json_to_ball_df(fpath)
        dfs.append(df)

    if len(dfs) > 0:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame()


def convert_match_json_to_ball_df(file_path):
    """
    Reads a single match JSON and returns a DataFrame of deliveries
    with columns needed for subsequent metric calculations.

    Also deduces Game Phase => Pressure for T20 or 50-over logic
    and extracts the 'season' field for filtering.
    """
    with open(file_path, "r") as f:
        match_data = json.load(f)

    teams = match_data["info"].get("teams", [])
    match_type = match_data["info"].get("match_type", "T20")
    season = match_data["info"].get("season", "Unknown")  # Grab the season

    ball_data = []
    innings_list = match_data.get("innings", [])
    for innings_index, innings in enumerate(innings_list, start=1):
        batting_team = innings.get("team")
        bowling_team = [t for t in teams if t != batting_team][0] if len(teams) > 1 else None

        for over_dict in innings.get("overs", []):
            over_number = over_dict.get("over", 0)
            deliveries = over_dict.get("deliveries", [])
            for ball_index, delivery in enumerate(deliveries):
                total_runs = delivery.get("runs", {}).get("total", 0)
                batter_runs = delivery.get("runs", {}).get("batter", 0)
                wicket_fell = bool(delivery.get("wickets"))

                row = {
                    "Match_File": os.path.basename(file_path),
                    "Season": season,
                    "Innings_Index": innings_index,
                    "Batting_Team": batting_team,
                    "Bowling_Team": bowling_team,
                    "Batter": delivery.get("batter"),
                    "Non_Striker": delivery.get("non_striker"),
                    "Bowler": delivery.get("bowler"),
                    "Over": over_number,
                    "Ball_In_Over": ball_index + 1,
                    "Runs_Total": total_runs,
                    "Runs_Batter": batter_runs,
                    "Wicket": wicket_fell,
                    "Match_Type": match_type
                }
                ball_data.append(row)

    df = pd.DataFrame(ball_data)

    # Determine Game Phase => Pressure
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

    if not df.empty:
        if df["Match_Type"].iloc[0] == "T20":
            df["Game_Phase"] = df["Over"].apply(game_phase_t20)
        else:
            df["Game_Phase"] = df["Over"].apply(game_phase_50)

        phase_to_pressure = {
            "Powerplay": "Low",
            "Middle": "Medium",
            "Death": "High"
        }
        df["Pressure"] = df["Game_Phase"].map(phase_to_pressure)

    return df


##############################
# 2) HELPER: GAMES PLAYED
##############################

def compute_batting_games_played(df):
    """
    For each Batter, how many unique Match_File did they bat in?
    """
    # Keep rows where Batter is not null
    sub = df.dropna(subset=["Batter"])[["Batter", "Match_File"]].drop_duplicates()
    # Group by Batter => count distinct Match_File
    out = sub.groupby("Batter")["Match_File"].nunique().reset_index()
    out.rename(columns={"Match_File": "GamesPlayed"}, inplace=True)
    return out


def compute_bowling_games_played(df):
    """
    For each Bowler, how many unique Match_File did they bowl in?
    """
    sub = df.dropna(subset=["Bowler"])[["Bowler", "Match_File"]].drop_duplicates()
    out = sub.groupby("Bowler")["Match_File"].nunique().reset_index()
    out.rename(columns={"Match_File": "GamesPlayed"}, inplace=True)
    return out


##############################
# 3) PRESSURE ANALYSIS
##############################

def compute_batting_metrics_by_pressure(df):
    """
    Group by Batter & Pressure => Runs, Balls, Dismissals, Strike Rate
    """
    if df.empty:
        return pd.DataFrame(columns=["Batter", "Pressure", "Runs", "Balls", "Dismissals", "StrikeRate"])

    df = df.dropna(subset=["Batter"]).copy()
    df["Balls_Faced"] = 1

    grp_cols = ["Batter", "Pressure"]
    grouped = df.groupby(grp_cols).agg(
        Runs=("Runs_Batter", "sum"),
        Balls=("Balls_Faced", "sum"),
        Dismissals=("Wicket", "sum")
    ).reset_index()

    grouped["StrikeRate"] = (grouped["Runs"] / grouped["Balls"]) * 100
    return grouped


def compute_bowling_metrics_by_pressure(df):
    """
    Group by Bowler & Pressure => runs, balls, wickets => economy
    """
    if df.empty:
        return pd.DataFrame(columns=["Bowler", "Pressure", "Balls", "Runs", "Wickets", "Economy"])

    df = df.dropna(subset=["Bowler"]).copy()
    df["Balls_Bowled"] = 1

    grp_cols = ["Bowler", "Pressure"]
    grouped = df.groupby(grp_cols).agg(
        Balls=("Balls_Bowled", "sum"),
        Runs=("Runs_Total", "sum"),
        Wickets=("Wicket", "sum")
    ).reset_index()

    grouped["Overs"] = grouped["Balls"] / 6.0
    grouped["Economy"] = grouped.apply(lambda row: row["Runs"] / row["Overs"] if row["Overs"] > 0 else 0, axis=1)
    return grouped


##############################
# 4) ADVANCED BATTING METRICS
##############################

def compute_advanced_batting_metrics(df):
    """
    6 batting metrics:
      1) Total_Runs
      2) Average
      3) StrikeRate
      4) Finisher
      5) BoundaryRate
      6) DotBallPct
    """
    if df.empty:
        return pd.DataFrame(columns=["Batter", "Total_Runs", "Average", "StrikeRate",
                                     "Finisher", "BoundaryRate", "DotBallPct"])

    df = df.copy()
    df["Balls_Faced"] = 1
    df["Is_Boundary"] = df["Runs_Batter"].apply(lambda x: 1 if x in [4, 6] else 0)
    df["Is_Dot"] = df.apply(lambda row: 1 if (row["Runs_Batter"] == 0 and not row["Wicket"]) else 0, axis=1)

    # Per-innings summary
    per_innings = df.groupby(["Match_File", "Innings_Index", "Batter"], dropna=True).agg(
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
    final["StrikeRate"] = (final["Total_Runs"] / final["Balls_Faced"] * 100).fillna(0)
    final["Finisher"] = final.apply(
        lambda row: row["FinisherCount"] / row["Innings_Count"] if row["Innings_Count"] > 0 else 0, axis=1)
    final["BoundaryRate"] = final.apply(
        lambda row: row["Total_Boundaries"] / row["Balls_Faced"] if row["Balls_Faced"] > 0 else 0, axis=1)
    final["DotBallPct"] = final.apply(
        lambda row: row["Total_Dots"] / row["Balls_Faced"] if row["Balls_Faced"] > 0 else 0, axis=1)

    return final[["Batter", "Total_Runs", "Average", "StrikeRate", "Finisher", "BoundaryRate", "DotBallPct"]]


##############################
# 5) ADVANCED BOWLING METRICS
##############################

def compute_advanced_bowling_metrics(df):
    """
    6 bowling metrics:
      1) Wickets
      2) Economy
      3) StrikeRate (balls/wicket)
      4) BounceBackRate
      5) KeyWicketIndex
      6) DeathOversEconomy
    """
    if df.empty:
        return pd.DataFrame(columns=["Bowler", "Wickets", "Economy", "StrikeRate",
                                     "BounceBackRate", "KeyWicketIndex", "DeathOversEconomy"])

    df = df.dropna(subset=["Bowler"]).copy()
    df["Balls_Bowled"] = 1
    df.sort_values(["Bowler", "Match_File", "Innings_Index", "Over", "Ball_In_Over"], inplace=True)

    # 1) Bounce Back Rate
    bounce_events = []
    for bowler, group in df.groupby("Bowler"):
        group = group.reset_index(drop=True)
        boundaries = 0
        bounce = 0
        for i in range(len(group) - 1):
            runs_bat = group.loc[i, "Runs_Batter"]
            if runs_bat in [4, 6]:
                next_runs = group.loc[i + 1, "Runs_Batter"]
                next_wicket = group.loc[i + 1, "Wicket"]
                if (next_runs == 0 and not next_wicket) or next_wicket:
                    bounce += 1
                boundaries += 1
        bounce_events.append({"Bowler": bowler, "Boundaries": boundaries, "BounceBacks": bounce})

    bounce_df = pd.DataFrame(bounce_events)
    bounce_df["BounceBackRate"] = bounce_df.apply(
        lambda row: row["BounceBacks"] / row["Boundaries"] if row["Boundaries"] > 0 else 0,
        axis=1
    )

    # 2) Key Wicket => Over < 10
    df["Is_KeyWicket"] = df.apply(lambda row: 1 if (row["Wicket"] == True and row["Over"] < 10) else 0, axis=1)

    # 3) Death Overs => Over >=16
    df["Is_DeathOver"] = df["Over"].apply(lambda o: 1 if o >= 16 else 0)

    grouped = df.groupby("Bowler").agg(
        Balls_Bowled=("Balls_Bowled", "sum"),
        Runs_Conceded=("Runs_Total", "sum"),
        Wickets=("Wicket", "sum"),
        KeyWickets=("Is_KeyWicket", "sum"),
        DeathBalls=("Is_DeathOver", "sum"),
        DeathRuns=("Runs_Total", lambda x: x[df.loc[x.index, "Is_DeathOver"] == 1].sum())
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


##############################
# 6) RADAR CHART HELPER
##############################

def create_radar_chart(df, player_col, player_list, metrics, title):
    """
    df: DataFrame with columns = [player_col, metric1, metric2, ...]
    We'll do a percentile-based capping (5th..95th) for each metric,
    then min-max scale to [0..1], then produce a single radar chart.
    """
    if df.empty or not player_list:
        return None

    sub = df[df[player_col].isin(player_list)].copy()
    if sub.empty:
        return None

    # For each metric, cap outliers (5th..95th) & scale
    capped_scaled_data = []
    for _, row in sub.iterrows():
        player_name = row[player_col]
        data_row = {"player": player_name}
        for m in metrics:
            q_low = df[m].quantile(0.05)
            q_high = df[m].quantile(0.95)
            val_capped = np.clip(row[m], q_low, q_high)
            min_val = df[m].clip(q_low, q_high).min()
            max_val = df[m].clip(q_low, q_high).max()
            rng = max_val - min_val if max_val > min_val else 1e-9
            val_scaled = (val_capped - min_val) / rng
            data_row[m] = val_scaled
        capped_scaled_data.append(data_row)

    radar_data = []
    for row in capped_scaled_data:
        player_name = row["player"]
        for m in metrics:
            radar_data.append({
                "player": player_name,
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


##############################
# 7) STREAMLIT APP
##############################

def main():
    st.set_page_config(
        page_title="Cric Assist - Season & Games Filter",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Teal/blue gradient
    st.markdown("""
        <style>
        .main {
            background: linear-gradient(120deg, #B2F7EF 0%, #7DF9FF 50%, #B2F7EF 100%);
            color: #1f3c56;
        }
        .css-18e3th9 {
            background-color: #1f3c56 !important;
        }
        .css-1d391kg {
            color: #f0f6ff !important;
        }
        .st-radio > label {
            font-weight:600;
        }
        .css-12oz5g7 {
            font-weight:600;
        }
        .css-ffhzg2 {
            color:#f0f6ff !important;
        }
        </style>
        """,
                unsafe_allow_html=True
                )

    st.title("Cric Assist - Season & Games Filter")
    st.markdown("Filter by season(s) and minimum games played, then explore pressure-based or advanced metrics.")

    # Load data
    df_all = load_and_combine_data("data")
    if df_all.empty:
        st.warning("No data found in the 'data' folder.")
        return

    # ========== Season Filter ==========
    seasons = sorted(df_all["Season"].dropna().unique().tolist())
    if seasons:
        selected_seasons = st.multiselect("Select Season(s):", seasons, default=seasons)
        # Filter by season
        df_all = df_all[df_all["Season"].isin(selected_seasons)]
    else:
        st.info("No season data found.")

    if df_all.empty:
        st.warning("No data after filtering by season(s).")
        return

    # ========== Min Games Filter ==========
    st.write("### Minimum Games Played Filter")
    min_games = st.number_input("Minimum number of games", min_value=1, value=5)

    # We'll apply the min games filter to each aggregator result.
    # Because "games played" differs for batters vs bowlers, we'll compute them separately.

    # TABS
    tab1, tab2, tab3 = st.tabs(["Pressure Analysis (Batting)", "Pressure Analysis (Bowling)", "Advanced Radar"])

    ######################
    # Tab 1: Batting Pressure
    ######################
    with tab1:
        st.subheader("Batting Under Different Pressure Situations")

        # 1) Compute the aggregator
        batting_pressure_df = compute_batting_metrics_by_pressure(df_all)

        # 2) Compute games played for batters
        batting_games_df = compute_batting_games_played(df_all)

        # 3) Merge to add "GamesPlayed"
        batting_pressure_df = batting_pressure_df.merge(batting_games_df, on="Batter", how="left")
        # 4) Filter by min_games
        batting_pressure_df = batting_pressure_df[batting_pressure_df["GamesPlayed"] >= min_games]

        if batting_pressure_df.empty:
            st.info("No batters match the min games criteria or data is empty.")
        else:
            players = sorted(batting_pressure_df["Batter"].unique().tolist())
            default_sel = ["AK Markram", "T Stubbs", "RD Rickelton"]
            default_sel = [p for p in default_sel if p in players]
            selected = st.multiselect("Select batters:", players, default=default_sel)

            if selected:
                sub = batting_pressure_df[batting_pressure_df["Batter"].isin(selected)].copy()
                if not sub.empty:
                    st.write("### Batting Metrics by Pressure")
                    st.dataframe(sub)

                    fig = px.bar(
                        sub,
                        x="Pressure",
                        y="StrikeRate",
                        color="Batter",
                        barmode="group",
                        title="Strike Rate by Pressure (Batting)",
                        color_discrete_sequence=px.colors.sequential.Teal
                    )
                    fig.update_layout(
                        xaxis=dict(title="Pressure Situation"),
                        yaxis=dict(title="Strike Rate"),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)"
                    )
                    st.plotly_chart(fig, use_container_width=True)

    ######################
    # Tab 2: Bowling Pressure
    ######################
    with tab2:
        st.subheader("Bowling Under Different Pressure Situations")

        bowling_pressure_df = compute_bowling_metrics_by_pressure(df_all)
        bowling_games_df = compute_bowling_games_played(df_all)
        bowling_pressure_df = bowling_pressure_df.merge(bowling_games_df, on="Bowler", how="left")
        bowling_pressure_df = bowling_pressure_df[bowling_pressure_df["GamesPlayed"] >= min_games]

        if bowling_pressure_df.empty:
            st.info("No bowlers match the min games criteria or data is empty.")
        else:
            bowlers = sorted(bowling_pressure_df["Bowler"].unique().tolist())
            default_sel_bowl = bowlers[:2]
            selected_bowl = st.multiselect("Select bowlers:", bowlers, default=default_sel_bowl)

            if selected_bowl:
                sub_bowl = bowling_pressure_df[bowling_pressure_df["Bowler"].isin(selected_bowl)].copy()
                if not sub_bowl.empty:
                    st.write("### Bowling Metrics by Pressure")
                    st.dataframe(sub_bowl)

                    fig2 = px.bar(
                        sub_bowl,
                        x="Pressure",
                        y="Economy",
                        color="Bowler",
                        barmode="group",
                        title="Economy by Pressure (Bowling)",
                        color_discrete_sequence=px.colors.sequential.Tealgrn
                    )
                    fig2.update_layout(
                        xaxis=dict(title="Pressure Situation"),
                        yaxis=dict(title="Economy (runs per over)"),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)"
                    )
                    st.plotly_chart(fig2, use_container_width=True)

    ######################
    # Tab 3: Advanced Radar
    ######################
    with tab3:
        st.subheader("Advanced Metrics Radar Charts")

        mode = st.radio("Choose a discipline:", ["Batting", "Bowling"], horizontal=True)

        if mode == "Batting":
            # 1) aggregator
            batting_advanced = compute_advanced_batting_metrics(df_all)
            # 2) games played
            batting_games_df = compute_batting_games_played(df_all)
            # 3) merge & filter
            batting_advanced = batting_advanced.merge(batting_games_df, on="Batter", how="left")
            batting_advanced = batting_advanced[batting_advanced["GamesPlayed"] >= min_games]

            if batting_advanced.empty:
                st.warning("No batting data after applying filters.")
            else:
                st.write("#### All Batters' Advanced Metrics (Raw)")
                st.dataframe(batting_advanced)

                all_batters = sorted(batting_advanced["Batter"].unique().tolist())
                default_sel_bat = ["AK Markram", "T Stubbs", "RD Rickelton"]
                default_sel_bat = [p for p in default_sel_bat if p in all_batters]
                sel_batters = st.multiselect("Select batters for radar:", all_batters, default=default_sel_bat)

                if sel_batters:
                    metrics_batting = ["Total_Runs", "Average", "StrikeRate", "Finisher", "BoundaryRate", "DotBallPct"]
                    fig_radar = create_radar_chart(
                        batting_advanced,
                        player_col="Batter",
                        player_list=sel_batters,
                        metrics=metrics_batting,
                        title="Batting Radar (6 Metrics)"
                    )
                    if fig_radar:
                        st.plotly_chart(fig_radar, use_container_width=True)

        else:
            bowling_advanced = compute_advanced_bowling_metrics(df_all)
            bowling_games_df = compute_bowling_games_played(df_all)
            bowling_advanced = bowling_advanced.merge(bowling_games_df, on="Bowler", how="left")
            bowling_advanced = bowling_advanced[bowling_advanced["GamesPlayed"] >= min_games]

            if bowling_advanced.empty:
                st.warning("No bowling data after applying filters.")
            else:
                st.write("#### All Bowlers' Advanced Metrics (Raw)")
                st.dataframe(bowling_advanced)

                all_bowlers = sorted(bowling_advanced["Bowler"].unique().tolist())
                default_sel_bowlers = all_bowlers[:2]
                sel_bowlers = st.multiselect("Select bowlers for radar:", all_bowlers, default=default_sel_bowlers)

                if sel_bowlers:
                    metrics_bowling = ["Wickets", "Economy", "StrikeRate", "BounceBackRate", "KeyWicketIndex",
                                       "DeathOversEconomy"]
                    fig_radar_bowl = create_radar_chart(
                        bowling_advanced,
                        player_col="Bowler",
                        player_list=sel_bowlers,
                        metrics=metrics_bowling,
                        title="Bowling Radar (6 Metrics)"
                    )
                    if fig_radar_bowl:
                        st.plotly_chart(fig_radar_bowl, use_container_width=True)


if __name__ == "__main__":
    main()
