import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any


def build_batting_scorecard(df_inn: pd.DataFrame, bat_ppi_df: Optional[pd.DataFrame] = None,
                            min_balls_for_ppi: int = 5) -> pd.DataFrame:
    """
    Generate a batting scorecard from innings data

    Parameters:
    -----------
    df_inn : pd.DataFrame
        Ball-by-ball data for a single innings
    bat_ppi_df : Optional[pd.DataFrame]
        Batting PPI data
    min_balls_for_ppi : int
        Minimum balls faced to qualify for PPI

    Returns:
    --------
    pd.DataFrame
        Batting scorecard
    """
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


def build_bowling_scorecard(df_inn: pd.DataFrame, bowl_ppi_df: Optional[pd.DataFrame] = None,
                            min_balls_for_ppi: int = 6) -> pd.DataFrame:
    """
    Generate a bowling scorecard from innings data

    Parameters:
    -----------
    df_inn : pd.DataFrame
        Ball-by-ball data for a single innings
    bowl_ppi_df : Optional[pd.DataFrame]
        Bowling PPI data
    min_balls_for_ppi : int
        Minimum balls bowled to qualify for PPI

    Returns:
    --------
    pd.DataFrame
        Bowling scorecard
    """
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

    # Determine bowling order based on when they first bowled
    bowling_order = {}
    seen_bowlers = set()

    # Go through each ball in the innings in chronological order
    for _, row in df_inn.sort_values(["Over", "Ball_In_Over"]).iterrows():
        bowler = row.get("Bowler")
        if bowler and bowler not in seen_bowlers:
            bowling_order[bowler] = len(seen_bowlers)
            seen_bowlers.add(bowler)

    # Sort by bowling order
    bowl_stats["BowlingOrder"] = bowl_stats["Bowler"].map(bowling_order)
    bowl_stats = bowl_stats.sort_values("BowlingOrder")

    return bowl_stats


def build_innings_summary(df_inn: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a summary of an innings with key statistics
    
    Parameters:
    -----------
    df_inn : pd.DataFrame
        Ball-by-ball data for a single innings
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary with innings summary statistics
    """
    if df_inn.empty:
        return {}
    
    # Filter to striker deliveries only
    striker_df = df_inn[df_inn["DeliveryType"] == "striker"].copy()
    
    # Get basic innings info
    batting_team = striker_df["Batting_Team"].iloc[0] if "Batting_Team" in striker_df.columns else "Unknown Team"
    bowling_team = striker_df["Bowling_Team"].iloc[0] if "Bowling_Team" in striker_df.columns else "Unknown Team"
    match_type = striker_df["Match_Type"].iloc[0] if "Match_Type" in striker_df.columns else "Unknown"
    
    # Calculate innings totals
    total_runs = striker_df["Runs_Total"].sum()
    legal_balls = striker_df["IsLegalDelivery"].sum()
    wickets = striker_df["Wicket"].sum()
    
    # Calculate overs
    complete_overs = legal_balls // 6
    remaining_balls = legal_balls % 6
    overs_display = f"{complete_overs}.{remaining_balls}"
    
    # Calculate run rate
    run_rate = (total_runs / (legal_balls / 6)).round(2) if legal_balls > 0 else 0
    
    # Count boundaries
    fours = striker_df[striker_df["Runs_Batter"] == 4].shape[0]
    sixes = striker_df[striker_df["Runs_Batter"] == 6].shape[0]
    
    # Calculate dot ball percentage
    dot_balls = ((striker_df["Runs_Batter"] == 0) & (striker_df["IsLegalDelivery"] == 1)).sum()
    dot_ball_pct = (dot_balls / legal_balls * 100).round(1) if legal_balls > 0 else 0
    
    # Find highest partnership
    if "Current_Partnership" in striker_df.columns:
        highest_partnership = striker_df["Current_Partnership"].max()
    else:
        highest_partnership = None
    
    # Find highest individual score
    batter_scores = striker_df.groupby("Batter")["Runs_Batter"].sum()
    if not batter_scores.empty:
        highest_scorer = batter_scores.idxmax()
        highest_score = batter_scores.max()
        top_score_display = f"{highest_scorer} ({highest_score})"
    else:
        top_score_display = "None"
    
    # Find best bowler
    if "Bowler" in striker_df.columns:
        bowler_wickets = striker_df.groupby("Bowler")["Wicket"].sum()
        if not bowler_wickets.empty and bowler_wickets.max() > 0:
            best_bowler = bowler_wickets.idxmax()
            best_bowler_wickets = bowler_wickets.max()
            best_bowler_runs = striker_df[striker_df["Bowler"] == best_bowler]["Bowler_Runs"].sum()
            best_bowler_display = f"{best_bowler} ({best_bowler_wickets}/{best_bowler_runs})"
        else:
            best_bowler_display = "None"
    else:
        best_bowler_display = "None"
    
    # Return summary dictionary
    return {
        "BattingTeam": batting_team,
        "BowlingTeam": bowling_team,
        "MatchType": match_type,
        "TotalRuns": total_runs,
        "Wickets": wickets,
        "Overs": overs_display,
        "RunRate": run_rate,
        "Fours": fours,
        "Sixes": sixes,
        "DotBallPct": dot_ball_pct,
        "HighestPartnership": highest_partnership,
        "TopScorer": top_score_display,
        "BestBowler": best_bowler_display
    }


def format_innings_summary_html(summary: Dict[str, Any]) -> str:
    """
    Format innings summary as HTML for display
    
    Parameters:
    -----------
    summary : Dict[str, Any]
        Dictionary with innings summary statistics
        
    Returns:
    --------
    str
        HTML formatted summary
    """
    if not summary:
        return "<p>No innings data available</p>"
    
    # Fix the HTML structure to ensure proper rendering
    html = f"""
    <div class="innings-summary">
        <h3>{summary['BattingTeam']} - {summary['TotalRuns']}/{summary['Wickets']} ({summary['Overs']})</h3>
        <div class="summary-stats">
            <div class="stat-item">
                <span class="stat-label">Run Rate:</span>
                <span class="stat-value">{summary['RunRate']}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Boundaries:</span>
                <span class="stat-value">{summary['Fours']}×4s, {summary['Sixes']}×6s</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Dot Ball %:</span>
                <span class="stat-value">{summary['DotBallPct']}%</span>
            </div>
    """
    
    if summary.get('HighestPartnership'):
        html += f"""
            <div class="stat-item">
                <span class="stat-label">Highest Partnership:</span>
                <span class="stat-value">{summary['HighestPartnership']}</span>
            </div>
        """
    
    html += f"""
            <div class="stat-item">
                <span class="stat-label">Top Scorer:</span>
                <span class="stat-value">{summary['TopScorer']}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Best Bowler:</span>
                <span class="stat-value">{summary['BestBowler']}</span>
            </div>
        </div>
    </div>
    """
    
    return html


def format_batting_scorecard_html(scorecard: pd.DataFrame) -> str:
    """
    Format batting scorecard as HTML for display
    
    Parameters:
    -----------
    scorecard : pd.DataFrame
        Batting scorecard data
        
    Returns:
    --------
    str
        HTML formatted scorecard
    """
    if scorecard.empty:
        return "<p>No batting data available</p>"
    
    # Create a copy to avoid modifying the original
    df = scorecard.copy()
    
    # Format strike rate
    df["StrikeRate"] = df["StrikeRate"].apply(lambda x: f"{x:.2f}")
    
    # Check if PPI is available
    has_ppi = "BatPPI" in df.columns
    
    # Start HTML table
    html = """
    <table class="scorecard batting-scorecard">
        <thead>
            <tr>
                <th>Batter</th>
                <th>Dismissal</th>
                <th>Runs</th>
                <th>Balls</th>
                <th>4s</th>
                <th>6s</th>
                <th>SR</th>
    """
    
    if has_ppi:
        html += "<th>PPI</th>"
    
    html += """
            </tr>
        </thead>
        <tbody>
    """
    
    # Add rows for each batter
    for _, row in df.iterrows():
        html += f"""
            <tr>
                <td>{row['Batter']}</td>
                <td>{row['Dismissal']}</td>
                <td>{row['Runs']}</td>
                <td>{row['Balls']}</td>
                <td>{row['Fours']}</td>
                <td>{row['Sixes']}</td>
                <td>{row['StrikeRate']}</td>
        """
        
        if has_ppi:
            ppi_val = row['BatPPI'] if pd.notna(row['BatPPI']) else "-"
            html += f"<td>{ppi_val}</td>"
        
        html += "</tr>"
    
    # Close the table
    html += """
        </tbody>
    </table>
    """
    
    return html


def format_bowling_scorecard_html(scorecard: pd.DataFrame) -> str:
    """
    Format bowling scorecard as HTML for display
    
    Parameters:
    -----------
    scorecard : pd.DataFrame
        Bowling scorecard data
        
    Returns:
    --------
    str
        HTML formatted scorecard
    """
    if scorecard.empty:
        return "<p>No bowling data available</p>"
    
    # Create a copy to avoid modifying the original
    df = scorecard.copy()
    
    # Format overs and economy
    df["Overs"] = df["Overs"].apply(lambda x: f"{int(x)}.{int(x*10)%10}")
    df["Economy"] = df["Economy"].apply(lambda x: f"{x:.2f}")
    
    # Check if PPI is available
    has_ppi = "BowlPPI" in df.columns
    
    # Start HTML table
    html = """
    <table class="scorecard bowling-scorecard">
        <thead>
            <tr>
                <th>Bowler</th>
                <th>Overs</th>
                <th>Maidens</th>
                <th>Runs</th>
                <th>Wickets</th>
                <th>Economy</th>
    """
    
    if has_ppi:
        html += "<th>PPI</th>"
    
    html += """
            </tr>
        </thead>
        <tbody>
    """
    
    # Add rows for each bowler
    for _, row in df.iterrows():
        html += f"""
            <tr>
                <td>{row['Bowler']}</td>
                <td>{row['Overs']}</td>
                <td>{row['Maidens']}</td>
                <td>{row['Runs']}</td>
                <td>{row['Wickets']}</td>
                <td>{row['Economy']}</td>
        """
        
        if has_ppi:
            ppi_val = row['BowlPPI'] if pd.notna(row['BowlPPI']) else "-"
            html += f"<td>{ppi_val}</td>"
        
        html += "</tr>"
    
    # Close the table
    html += """
        </tbody>
    </table>
    """
    
    return html