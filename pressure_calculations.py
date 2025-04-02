import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

# Define global constants
PRESSURE_LEVEL_ORDER = ["Low", "Medium", "High", "Extreme"]

def calculate_dynamic_pressure_for_innings(df_innings: pd.DataFrame, first_innings_total: Optional[int] = None) -> pd.DataFrame:
    """
    Calculate dynamic pressure for each ball in an innings.
    
    Parameters:
    -----------
    df_innings : pd.DataFrame
        Ball-by-ball data for a single innings
    first_innings_total : Optional[int]
        Total runs scored in the first innings (for second innings pressure calculation)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added pressure metrics
    """
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
                    base_score = (part1 + part2 + part3 + part4 + powerplay_bowling_factor) * partnership_factor * chase_factor
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

def assign_separate_t20_pressure_labels(df_innings: pd.DataFrame, first_innings_total: Optional[int] = None) -> pd.DataFrame:
    """
    For T20 cricket, assign separate BattingPressureLabel and BowlingPressureLabel
    using a 3-level scale (Low, Medium, High) with smoothing to create organic transitions.
    
    Parameters:
    -----------
    df_innings : pd.DataFrame
        Ball-by-ball data for a single innings
    first_innings_total : Optional[int]
        Total runs scored in the first innings (for second innings pressure calculation)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added batting and bowling pressure labels
    """
    if df_innings.empty:
        return df_innings
    if df_innings["Match_Type"].iloc[0] != "T20":
        return df_innings

    df = df_innings.copy()
    raw_bat = []
    raw_bowl = []
    for _, row in df.iterrows():
        raw_bat.append(compute_continuous_batting_pressure(row, first_innings_total))
        raw_bowl.append(compute_continuous_bowling_pressure(row, first_innings_total))
    df["RawBatPressureScore"] = raw_bat
    df["RawBowlPressureScore"] = raw_bowl

    # Apply a rolling window smoothing (3-ball window)
    window_size = 3
    # We'll smooth separately for striker and non-striker deliveries so that we don't mix values.
    smoothed_bat = df.groupby("DeliveryType")["RawBatPressureScore"].transform(
        lambda x: x.rolling(window=window_size, min_periods=1).mean()
    )
    smoothed_bowl = df.groupby("DeliveryType")["RawBowlPressureScore"].transform(
        lambda x: x.rolling(window=window_size, min_periods=1).mean()
    )
    df["BatPressureScore"] = smoothed_bat
    df["BowlPressureScore"] = smoothed_bowl

    # Protect high raw scores from being dragged down:
    # If the raw score is >= 2.3 and the smoothed value is lower, use the raw value.
    df["BatPressureScore"] = df.apply(
        lambda r: r["RawBatPressureScore"] if r["RawBatPressureScore"] >= 2.3 and r["BatPressureScore"] < r["RawBatPressureScore"] else r["BatPressureScore"],
        axis=1
    )
    df["BowlPressureScore"] = df.apply(
        lambda r: r["RawBowlPressureScore"] if r["RawBowlPressureScore"] >= 2.3 and r["BowlPressureScore"] < r["RawBowlPressureScore"] else r["BowlPressureScore"],
        axis=1
    )

    # Map continuous scores to discrete labels
    df["BattingPressureLabel"] = df["BatPressureScore"].apply(get_pressure_label)
    df["BowlingPressureLabel"] = df["BowlPressureScore"].apply(get_pressure_label)

    # For backward compatibility:
    df["DynamicPressureLabel"] = df["BattingPressureLabel"]
    df["DynamicPressureScore"] = df["BatPressureScore"]

    return df

def compute_continuous_batting_pressure(row: pd.Series, first_innings_total: Optional[int] = None) -> float:
    """
    Calculate a continuous batting pressure score (scale 1.0=Low, 2.0=Medium, 3.0=High)
    for a ball in a T20 match.
    
    Parameters:
    -----------
    row : pd.Series
        Data for a single ball
    first_innings_total : Optional[int]
        Total runs scored in the first innings (for second innings pressure calculation)
        
    Returns:
    --------
    float
        Continuous batting pressure score
    """
    # Extract key variables
    ball_index = row["Over"] + (row["Ball_In_Over"] - 1) / 6.0
    innings_idx = row["Innings_Index"]
    wickets_fallen = row["Wickets_Fallen"] if pd.notna(row["Wickets_Fallen"]) else 0
    wickets_in_hand = 10 - wickets_fallen
    crr = row["Current_Run_Rate"] if pd.notna(row["Current_Run_Rate"]) else 0.0
    rrr = row["Required_Run_Rate"] if pd.notna(row["Required_Run_Rate"]) else 0.0
    wickets_in_3 = row["Wickets_In_Last_3_Overs"] if pd.notna(row["Wickets_In_Last_3_Overs"]) else 0
    
    # Default baseline pressure (Medium)
    base_pressure = 2.0

    if innings_idx == 1:
        # --- FIRST INNINGS (Batting First) ---
        # Blanket rule: after 3 overs, if CRR is less than 6, force high pressure.
        if ball_index > 3.0 and crr < 6:
            base_pressure = 3.0
        else:
            if ball_index < 6.0:
                # In the powerplay: if in the first 2 overs and CRR > 10 with no wicket, very low pressure.
                if ball_index < 2.0 and crr > 10 and wickets_fallen == 0:
                    base_pressure = 1.0
                else:
                    base_pressure = 3.0 if ball_index < 3.0 else 2.5
                # If 2 or more wickets have fallen early, pressure increases significantly
                if wickets_fallen >= 2:
                    base_pressure = max(base_pressure, 2.8)
            elif ball_index < 16.0:
                # In the middle overs:
                # If 3 or 4 wickets fall in the first 10 overs, force at least medium/high pressure.
                if ball_index < 10:
                    if wickets_fallen >= 4:
                        base_pressure = 3.0
                    elif wickets_fallen >= 3:
                        base_pressure = 2.5
                    else:
                        base_pressure = 1.5
                else:
                    # After 10 overs, if batsmen are in control, pressure can be low.
                    if crr >= 8 and wickets_in_hand >= 6:
                        base_pressure = 1.0
                    else:
                        base_pressure = 1.5
            else:
                # Death overs:
                if crr >= 9 and wickets_in_hand >= 6:
                    base_pressure = 1.0
                else:
                    base_pressure = 2.0
    else:
        # --- SECOND INNINGS (Chase) ---
        base_pressure = 2.5
        if rrr > 12:
            base_pressure += 0.5
        if (rrr - crr) < 1.5:
            base_pressure -= 0.3
        if wickets_in_3 >= 2:
            base_pressure += 0.4

    return max(1.0, min(3.0, base_pressure))

def compute_continuous_bowling_pressure(row: pd.Series, first_innings_total: Optional[int] = None) -> float:
    """
    Calculate a continuous bowling pressure score (scale 1.0=Low, 2.0=Medium, 3.0=High)
    for a ball in a T20 match.
    
    Parameters:
    -----------
    row : pd.Series
        Data for a single ball
    first_innings_total : Optional[int]
        Total runs scored in the first innings (for second innings pressure calculation)
        
    Returns:
    --------
    float
        Continuous bowling pressure score
    """
    ball_index = row["Over"] + (row["Ball_In_Over"] - 1) / 6.0
    innings_idx = row["Innings_Index"]
    wickets_fallen = row["Wickets_Fallen"] if pd.notna(row["Wickets_Fallen"]) else 0
    wickets_in_hand = 10 - wickets_fallen
    crr = row["Current_Run_Rate"] if pd.notna(row["Current_Run_Rate"]) else 0.0
    rrr = row["Required_Run_Rate"] if pd.notna(row["Required_Run_Rate"]) else 0.0

    base_pressure = 1.0

    if innings_idx == 1:
        # --- FIRST INNINGS (Batting First) ---
        if ball_index < 6.0:
            base_pressure = 3.0
            # If wickets fall early, bowlers experience less pressure.
            if ball_index < 10 and wickets_fallen >= 3:
                base_pressure = 1.5
        elif ball_index < 16.0:
            # In middle overs, if batsmen are in full flow (high CRR & many wickets in hand),
            # bowlers experience lower pressure.
            if crr >= 9 and wickets_in_hand >= 6:
                base_pressure = 1.0
            else:
                base_pressure = 1.5
        else:
            if crr >= 9 and wickets_in_hand >= 6:
                base_pressure = 1.5
            else:
                base_pressure = 2.5
    else:
        # --- SECOND INNINGS (Chase) ---
        base_pressure = 2.5
        if 7 <= rrr <= 10:
            base_pressure = 2.8
        if rrr > 15:
            base_pressure -= 0.5
        elif rrr < 6:
            base_pressure -= 0.3
        if ball_index >= 16 and 7 <= rrr <= 12:
            base_pressure = 3.0

    return max(1.0, min(3.0, base_pressure))

def get_pressure_label(score: float) -> str:
    """
    Map a continuous pressure score (1.0 to 3.0) to a discrete label.
    
    Parameters:
    -----------
    score : float
        Continuous pressure score
        
    Returns:
    --------
    str
        Pressure label (Low, Medium, High)
    """
    if score < 1.67:
        return "Low"
    elif score < 2.33:
        return "Medium"
    else:
        return "High"

def ensure_pressure_labels_robust(df: pd.DataFrame) -> pd.DataFrame:
    """
    More robust version of ensure_pressure_labels that guarantees pressure labels exist.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Ball-by-ball data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with guaranteed pressure labels
    """
    if df is None or df.empty:
        return df
    
    df = df.copy()  # Create a copy to avoid modifying the original
    
    # ALWAYS ensure DynamicPressureLabel exists
    if "DynamicPressureLabel" not in df.columns:
        if "BattingPressureLabel" in df.columns:
            # Use batting pressure if available
            df["DynamicPressureLabel"] = df["BattingPressureLabel"]
        else:
            # Otherwise, assign a default value
            df["DynamicPressureLabel"] = "Medium"
    
    # Also ensure DynamicPressureScore exists
    if "DynamicPressureScore" not in df.columns:
        if "BatPressureScore" in df.columns:
            df["DynamicPressureScore"] = df["BatPressureScore"]
        else:
            df["DynamicPressureScore"] = 2.0  # Default value
    
    # Ensure BattingPressureLabel exists
    if "BattingPressureLabel" not in df.columns:
        df["BattingPressureLabel"] = df["DynamicPressureLabel"]
    
    # Ensure BowlingPressureLabel exists
    if "BowlingPressureLabel" not in df.columns:
        # Try to calculate it, but if that fails, just use a default
        try:
            # Only calculate if we have the necessary data
            if all(col in df.columns for col in ["Over", "Ball_In_Over", "Innings_Index", "Wickets_Fallen"]):
                temp_df = assign_separate_t20_pressure_labels(df)
                df["BowlingPressureLabel"] = temp_df["BowlingPressureLabel"]
            else:
                df["BowlingPressureLabel"] = "Medium"
        except Exception:
            df["BowlingPressureLabel"] = "Medium"
    
    return df

def order_pressure_levels(df: pd.DataFrame, pressure_col: str = "DynamicPressureLabel") -> pd.DataFrame:
    """
    Apply consistent ordering to pressure level categories
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing pressure labels
    pressure_col : str
        Column name containing pressure labels
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with ordered pressure levels
    """
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