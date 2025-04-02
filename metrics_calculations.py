import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

# Import constants
from pressure_calculations import PRESSURE_LEVEL_ORDER


def compute_batting_games_played(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the number of games played by each batter.

    Parameters:
    -----------
    df : pd.DataFrame
        Ball-by-ball data

    Returns:
    --------
    pd.DataFrame
        DataFrame with batter and games played
    """
    sub = df.dropna(subset=["Batter"])[["Batter", "Match_File"]].drop_duplicates()
    out = sub.groupby("Batter")["Match_File"].nunique().reset_index()
    out.rename(columns={"Match_File": "GamesPlayed"}, inplace=True)
    return out


def compute_bowling_games_played(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the number of games played by each bowler.

    Parameters:
    -----------
    df : pd.DataFrame
        Ball-by-ball data

    Returns:
    --------
    pd.DataFrame
        DataFrame with bowler and games played
    """
    sub = df.dropna(subset=["Bowler"])[["Bowler", "Match_File"]].drop_duplicates()
    out = sub.groupby("Bowler")["Match_File"].nunique().reset_index()
    out.rename(columns={"Match_File": "GamesPlayed"}, inplace=True)
    return out


def compute_batting_metrics_by_pressure(df: pd.DataFrame, pressure_col: str = "BattingPressureLabel",
                                        aggregate: bool = False) -> pd.DataFrame:
    """
    Calculate batting performance metrics grouped by pressure level.
    Uses the BattingPressureLabel instead of legacy DynamicPressureLabel.

    Parameters:
    -----------
    df : pd.DataFrame
        Ball-by-ball data
    pressure_col : str
        Column name containing pressure labels
    aggregate : bool
        If True, aggregate across all batters

    Returns:
    --------
    pd.DataFrame
        DataFrame with batting metrics by pressure level
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
    # pressure_metrics = pressure_metrics.rename(columns={pressure_col: "Pressure"})

    return pressure_metrics


def compute_bowling_metrics_by_pressure(df: pd.DataFrame, pressure_col: str = "BowlingPressureLabel",
                                        aggregate: bool = False) -> pd.DataFrame:
    """
    Calculate bowling performance metrics grouped by pressure level.
    Uses the BowlingPressureLabel instead of legacy DynamicPressureLabel.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Ball-by-ball data
    pressure_col : str
        Column name containing pressure labels
    aggregate : bool
        If True, aggregate across all bowlers
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with bowling metrics by pressure level
    """
    # Implementation using the new labels
    if df.empty or pressure_col not in df.columns:
        return pd.DataFrame()
    
    # Filter to striker deliveries only
    df_striker = df[df["DeliveryType"] == "striker"].copy()
    
    # Group by criteria - either by bowler and pressure or just pressure
    if not aggregate:
        group_cols = ["Bowler", pressure_col]
    else:
        group_cols = [pressure_col]
    
    # Calculate dot balls for bowling (before grouping)
    df_striker["DotBall"] = ((df_striker["Runs_Batter"] == 0) & 
                            (df_striker["IsLegalDelivery"] == 1)).astype(int)
    
    # Compute metrics using the correct column names
    bowling_metrics = df_striker.groupby(group_cols).agg(
        Balls=pd.NamedAgg(column="IsLegalDelivery", aggfunc=lambda x: (x == 1).sum()),
        Runs_Conceded=("Bowler_Runs", "sum"),
        Wickets=pd.NamedAgg(column="Wicket", aggfunc=lambda x: (x == 1).sum()),
        Dots=("DotBall", "sum")
    ).reset_index()

    # Calculate derived metrics
    bowling_metrics["Economy"] = (bowling_metrics["Runs_Conceded"] / bowling_metrics["Balls"]) * 6
    bowling_metrics["DotBallPct"] = (bowling_metrics["Dots"] / bowling_metrics["Balls"]) * 100
    bowling_metrics["Strike_Rate"] = bowling_metrics["Balls"] / bowling_metrics["Wickets"].replace(0, float('nan'))
    bowling_metrics["Average"] = bowling_metrics["Runs_Conceded"] / bowling_metrics["Wickets"].replace(0, float('nan'))
    
    # Keep the original pressure column name instead of renaming it
    # bowling_metrics = bowling_metrics.rename(columns={pressure_col: "Pressure"})
    
    return bowling_metrics


def compute_overall_batting_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute batting metrics across all matches with correct dismissal tracking

    Parameters:
    -----------
    df : pd.DataFrame
        Ball-by-ball data

    Returns:
    --------
    pd.DataFrame
        DataFrame with overall batting metrics
    """
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


def compute_advanced_batting_metrics(df: pd.DataFrame, bat_ppi_agg: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Compute advanced batting metrics

    Parameters:
    -----------
    df : pd.DataFrame
        Ball-by-ball data
    bat_ppi_agg : Optional[pd.DataFrame]
        Aggregated batting PPI data

    Returns:
    --------
    pd.DataFrame
        DataFrame with advanced batting metrics
    """
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
        high_pressure = batter_df[batter_df["BattingPressureLabel"].isin(["High", "Extreme"])]
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


def compute_advanced_bowling_metrics(df: pd.DataFrame, bowl_ppi_agg: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Compute advanced bowling metrics

    Parameters:
    -----------
    df : pd.DataFrame
        Ball-by-ball data
    bowl_ppi_agg : Optional[pd.DataFrame]
        Aggregated bowling PPI data

    Returns:
    --------
    pd.DataFrame
        DataFrame with advanced bowling metrics
    """
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


def compute_match_ppi_batting(df: pd.DataFrame, min_balls: int = 5) -> pd.DataFrame:
    """
    For each match_file + Batter, calculate a Batting Performance Index
    scaled 0-100 among qualified batters in that match.

    Parameters:
    -----------
    df : pd.DataFrame
        Ball-by-ball data
    min_balls : int
        Minimum balls faced to qualify

    Returns:
    --------
    pd.DataFrame
        DataFrame with batting PPI for each match and batter
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


def compute_match_ppi_bowling(df: pd.DataFrame, min_balls: int = 6, cameo_overs: float = 2.5) -> pd.DataFrame:
    """
    For each match_file + Bowler, calculate a Bowling Performance Index
    scaled 0-100 among qualified bowlers in that match.

    Parameters:
    -----------
    df : pd.DataFrame
        Ball-by-ball data
    min_balls : int
        Minimum balls bowled to qualify
    cameo_overs : float
        Threshold for applying cameo penalty

    Returns:
    --------
    pd.DataFrame
        DataFrame with bowling PPI for each match and bowler
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


def get_default_batters_pressure(df: pd.DataFrame, n: int = 5) -> List[str]:
    """
    Get default batters for pressure analysis
    
    Parameters:
    -----------
    df : pd.DataFrame
        Ball-by-ball data
    n : int
        Number of batters to return
        
    Returns:
    --------
    List[str]
        List of batter names
    """
    if df.empty:
        return []
    grouped = df.groupby("Batter", dropna=True)["Runs_Batter"].sum().reset_index()
    grouped = grouped.sort_values("Runs_Batter", ascending=False)
    top = grouped.head(n)["Batter"].tolist()
    return top


def get_default_bowlers_pressure(df: pd.DataFrame, n: int = 5) -> List[str]:
    """
    Get default bowlers for pressure analysis
    
    Parameters:
    -----------
    df : pd.DataFrame
        Ball-by-ball data
    n : int
        Number of bowlers to return
        
    Returns:
    --------
    List[str]
        List of bowler names
    """
    if df.empty:
        return []
    grouped = df.groupby("Bowler", dropna=True)["Wicket"].sum().reset_index()
    grouped = grouped.sort_values("Wicket", ascending=False)
    top = grouped.head(n)["Bowler"].tolist()
    return top


def get_default_batters_adv(df: pd.DataFrame, n: int = 5) -> List[str]:
    """
    Get default batters for advanced metrics
    
    Parameters:
    -----------
    df : pd.DataFrame
        Advanced batting metrics DataFrame
    n : int
        Number of batters to return
        
    Returns:
    --------
    List[str]
        List of batter names
    """
    if df.empty:
        return []
    top = df.sort_values("Total_Runs", ascending=False).head(n)["Batter"].tolist()
    return top


def get_default_bowlers_adv(df: pd.DataFrame, n: int = 5) -> List[str]:
    """
    Get default bowlers for advanced metrics
    
    Parameters:
    -----------
    df : pd.DataFrame
        Advanced bowling metrics DataFrame
    n : int
        Number of bowlers to return
        
    Returns:
    --------
    List[str]
        List of bowler names
    """
    if df.empty:
        return []
    top = df.sort_values("Wickets", ascending=False).head(n)["Bowler"].tolist()
    return top