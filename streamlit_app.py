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
    all_dfs = []
    for fpath in all_files:
        df_match = convert_match_json_to_ball_df(fpath)
        if not df_match.empty:
            all_dfs.append(df_match)

    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)


##############################
# 2) CALCULATE DYNAMIC PRESSURE (1ST + 2ND INNINGS)
##############################

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
    # We'll scale parDeficit by 35 for 1st innings
    # We'll scale required RR by 7 in 2nd innings (since typical ~8 => near 1.14 baseline)
    # We'll define a small overs-based multiplier that grows from ~1.0 to ~1.5 as oversUsed goes from 0..20:
    # time_factor = 1 + (oversUsed/40). => e.g. oversUsed=20 => factor=1.5
    # This ensures death overs => higher base pressure.

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

        # time factor
        time_factor = 1.0 + (overs_used / 40.0)  # ranges ~1.0 -> 1.5

        if inn_idx == 1:
            # -------- FIRST INNINGS --------
            expected_runs_so_far = TYPICAL_RUN_RATE * overs_used
            par_deficit = expected_runs_so_far - runs_cumulative
            if par_deficit < 0:
                par_deficit = 0  # if you're above par, no "extra" pressure from the deficit

            if wickets_in_hand <= 0 or overs_used >= 20:
                # innings basically done
                pressure_score = 0
            else:
                # scale parDeficit / 35
                part1 = alpha * (par_deficit / 35.0)
                part2 = beta * (1.0 - (wickets_in_hand / 10.0))
                base_score = part1 + part2
                pressure_score = base_score * time_factor

        else:
            # -------- SECOND INNINGS --------
            if not first_innings_total:
                # if we don't know the target, skip
                pressure_score = np.nan
            else:
                runs_needed = first_innings_total - runs_cumulative + 1
                balls_left = 120 - legal_balls_count  # total T20 = 120 legal deliveries

                if runs_needed <= 0 or balls_left <= 0 or wickets_in_hand <= 0:
                    pressure_score = 0
                else:
                    req_run_rate = runs_needed / (balls_left / 6.0)
                    part1 = alpha * (req_run_rate / 7.0)  # scale by 7
                    part2 = beta * (1.0 - (wickets_in_hand / 10.0))
                    base_score = part1 + part2
                    pressure_score = base_score * time_factor

        scores.append(pressure_score)

    df_innings["DynamicPressureScore"] = scores

    # Label
    def label_pressure(val):
        if pd.isna(val):
            return None
        # Adjust the thresholds so we get more variety
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


##############################
# 3) CONVERT MATCH JSON => BALL DF
##############################

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

    # Static old logic if you want it
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


##############################
# 4) GAMES PLAYED HELPERS
##############################

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


##############################
# 5) METRICS BY PRESSURE
##############################

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


##############################
# 6) ADVANCED BATTING + BOWLING
##############################

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
        return pd.DataFrame(columns=["Bowler", "Wickets", "Economy", "StrikeRate", "BounceBackRate", "KeyWicketIndex",
                                     "DeathOversEconomy"])

    sub = df.dropna(subset=["Bowler"]).copy()
    sub["Balls_Bowled"] = sub["IsLegalDelivery"]
    sub.sort_values(["Bowler", "Match_File", "Innings_Index", "Over", "Ball_In_Over"], inplace=True)

    # Bounce Back Rate
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
    # Death overs => Over >=16
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


##############################
# 7) RADAR CHART
##############################

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


##############################
# 8) STREAMLIT APP
##############################

def main():
    st.set_page_config(page_title="Cric Assist - Enhanced Pressure", layout="wide")
    st.title("Cric Assist - Enhanced Pressure for Both Innings")

    df_all = load_and_combine_data("data")
    if df_all.empty:
        st.warning("No data found. Place JSON files in 'data' folder.")
        return

    # Filter by season
    seasons = sorted(df_all["Season"].dropna().unique().tolist())
    if seasons:
        selected_seasons = st.multiselect("Select Seasons", seasons, default=seasons)
        df_all = df_all[df_all["Season"].isin(selected_seasons)]
    if df_all.empty:
        st.warning("No data after filtering by season(s).")
        return

    # Minimum games
    st.write("### Minimum Games Played Filter")
    min_games = st.number_input("Minimum number of games", min_value=1, value=3)

    tabs = st.tabs([
        "Batting - Dynamic Pressure",
        "Bowling - Dynamic Pressure",
        "Advanced Metrics + Radar",
        "Raw Data Preview"
    ])

    # BATTING - DYNAMIC PRESSURE
    with tabs[0]:
        st.subheader("Batting Metrics By Enhanced Pressure Label")
        bat_pressure = compute_batting_metrics_by_pressure(df_all, pressure_col="DynamicPressureLabel")
        if bat_pressure.empty:
            st.info("No batting data or no dynamic pressure calculated.")
        else:
            bat_games = compute_batting_games_played(df_all)
            merged_bat = bat_pressure.merge(bat_games, on="Batter", how="left")
            merged_bat = merged_bat[merged_bat["GamesPlayed"] >= min_games]
            if merged_bat.empty:
                st.info("No batters meet the min games criteria.")
            else:
                all_batters = sorted(merged_bat["Batter"].unique().tolist())
                selected_batters = st.multiselect("Select Batters to Compare", all_batters, default=all_batters[:3])

                sub_bat = merged_bat[merged_bat["Batter"].isin(selected_batters)]
                st.dataframe(sub_bat)

                fig_bat = px.bar(
                    sub_bat,
                    x="DynamicPressureLabel",
                    y="StrikeRate",
                    color="Batter",
                    barmode="group",
                    title="Batting Strike Rate by Enhanced Pressure",
                )
                fig_bat.update_layout(xaxis_title="Pressure", yaxis_title="Strike Rate")
                st.plotly_chart(fig_bat, use_container_width=True)

                fig_bat2 = px.bar(
                    sub_bat,
                    x="DynamicPressureLabel",
                    y="DotBallPct",
                    color="Batter",
                    barmode="group",
                    title="Dot Ball % by Enhanced Pressure",
                )
                fig_bat2.update_layout(xaxis_title="Pressure", yaxis_title="Dot Ball Percentage")
                st.plotly_chart(fig_bat2, use_container_width=True)

    # BOWLING - DYNAMIC PRESSURE
    with tabs[1]:
        st.subheader("Bowling Metrics By Enhanced Pressure Label")
        bowl_pressure = compute_bowling_metrics_by_pressure(df_all, pressure_col="DynamicPressureLabel")
        if bowl_pressure.empty:
            st.info("No bowling data or no dynamic pressure calculated.")
        else:
            bowl_games = compute_bowling_games_played(df_all)
            merged_bowl = bowl_pressure.merge(bowl_games, on="Bowler", how="left")
            merged_bowl = merged_bowl[merged_bowl["GamesPlayed"] >= min_games]
            if merged_bowl.empty:
                st.info("No bowlers meet the min games criteria.")
            else:
                all_bowlers = sorted(merged_bowl["Bowler"].unique().tolist())
                selected_bowlers = st.multiselect("Select Bowlers to Compare", all_bowlers, default=all_bowlers[:3])

                sub_bowl = merged_bowl[merged_bowl["Bowler"].isin(selected_bowlers)]
                st.dataframe(sub_bowl)

                fig_bowl = px.bar(
                    sub_bowl,
                    x="DynamicPressureLabel",
                    y="Economy",
                    color="Bowler",
                    barmode="group",
                    title="Bowling Economy by Enhanced Pressure"
                )
                fig_bowl.update_layout(xaxis_title="Pressure", yaxis_title="Economy (RPO)")
                st.plotly_chart(fig_bowl, use_container_width=True)

                fig_bowl2 = px.bar(
                    sub_bowl,
                    x="DynamicPressureLabel",
                    y="DotBallPct",
                    color="Bowler",
                    barmode="group",
                    title="Bowling Dot Ball % by Enhanced Pressure"
                )
                fig_bowl2.update_layout(xaxis_title="Pressure", yaxis_title="Dot Ball Percentage")
                st.plotly_chart(fig_bowl2, use_container_width=True)

    # ADVANCED METRICS + RADAR
    with tabs[2]:
        st.subheader("Advanced Batting & Bowling + Radar Charts")
        col1, col2 = st.columns(2)

        with col1:
            st.write("### Batting")
            adv_bat = compute_advanced_batting_metrics(df_all)
            bat_games = compute_batting_games_played(df_all)
            adv_bat = adv_bat.merge(bat_games, on="Batter", how="left")
            adv_bat = adv_bat[adv_bat["GamesPlayed"] >= min_games]
            st.dataframe(adv_bat)

            all_bat = sorted(adv_bat["Batter"].unique().tolist())
            sel_bat = st.multiselect("Select Batters for Radar", all_bat, default=all_bat[:3])
            if sel_bat:
                metrics_batting = ["Total_Runs", "Average", "StrikeRate", "Finisher", "BoundaryRate", "DotBallPct"]
                fig_radar_bat = create_radar_chart(adv_bat, "Batter", sel_bat, metrics_batting, "Batting Radar")
                if fig_radar_bat:
                    st.plotly_chart(fig_radar_bat, use_container_width=True)

        with col2:
            st.write("### Bowling")
            adv_bowl = compute_advanced_bowling_metrics(df_all)
            bowl_games = compute_bowling_games_played(df_all)
            adv_bowl = adv_bowl.merge(bowl_games, on="Bowler", how="left")
            adv_bowl = adv_bowl[adv_bowl["GamesPlayed"] >= min_games]
            st.dataframe(adv_bowl)

            all_bowl = sorted(adv_bowl["Bowler"].unique().tolist())
            sel_bowl = st.multiselect("Select Bowlers for Radar", all_bowl, default=all_bowl[:3])
            if sel_bowl:
                metrics_bowling = ["Wickets", "Economy", "StrikeRate", "BounceBackRate", "KeyWicketIndex",
                                   "DeathOversEconomy"]
                fig_radar_bowl = create_radar_chart(adv_bowl, "Bowler", sel_bowl, metrics_bowling, "Bowling Radar")
                if fig_radar_bowl:
                    st.plotly_chart(fig_radar_bowl, use_container_width=True)

    # RAW DATA
    with tabs[3]:
        st.subheader("Raw Data (with Enhanced Dynamic Pressure Score/Label)")
        st.write(df_all.head(300))


if __name__ == "__main__":
    main()
