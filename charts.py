import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple, Union, Any

# Import constants
from pressure_calculations import PRESSURE_LEVEL_ORDER


def create_radar_chart(df: pd.DataFrame, player_col: str, player_list: List[str], 
                       metrics: List[str], title: str) -> Optional[go.Figure]:
    """
    Create a radar chart comparing multiple players across different metrics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing player metrics
    player_col : str
        Column name containing player names
    player_list : List[str]
        List of players to include in the chart
    metrics : List[str]
        List of metrics to include in the chart
    title : str
        Chart title
        
    Returns:
    --------
    Optional[go.Figure]
        Radar chart figure or None if no data
    """
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
    
    # Add explicit height and width
    fig.update_layout(
        height=500,
        width=800
    )
    
    return fig


def plot_pressure_timeline(df_innings: pd.DataFrame) -> go.Figure:
    """
    Create a timeline visualization of pressure throughout an innings.
    
    Parameters:
    -----------
    df_innings : pd.DataFrame
        Ball-by-ball data for a single innings
        
    Returns:
    --------
    go.Figure
        Plotly figure with pressure timeline
    """
    # Create a basic figure to return in case of errors
    empty_fig = go.Figure()
    empty_fig.update_layout(
        title="Pressure Timeline (No Data Available)",
        template="plotly_dark"
    )
    
    if df_innings.empty:
        return empty_fig
    
    # Print debugging info
    print(f"DataFrame shape: {df_innings.shape}")
    print(f"Columns available: {df_innings.columns.tolist()}")
    
    # Check for required columns
    required_cols = ["Over", "Ball_In_Over"]
    missing_cols = [col for col in required_cols if col not in df_innings.columns]
    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        return empty_fig
    
    # Check for pressure columns - we need at least one
    pressure_cols = ["BattingPressureLabel", "BowlingPressureLabel", "DynamicPressureLabel"]
    available_pressure_cols = [col for col in pressure_cols if col in df_innings.columns]
    
    if not available_pressure_cols:
        print("No pressure columns found in the data")
        return empty_fig
    
    print(f"Available pressure columns: {available_pressure_cols}")
    
    # Create working copy with necessary columns
    df_timeline = df_innings.copy()
    
    # Ensure we have both batting and bowling pressure labels
    if "BattingPressureLabel" not in df_timeline.columns:
        if "DynamicPressureLabel" in df_timeline.columns:
            df_timeline["BattingPressureLabel"] = df_timeline["DynamicPressureLabel"]
        else:
            # Create a default "Medium" label if no pressure data available
            df_timeline["BattingPressureLabel"] = "Medium"
    
    if "BowlingPressureLabel" not in df_timeline.columns:
        if "DynamicPressureLabel" in df_timeline.columns:
            df_timeline["BowlingPressureLabel"] = df_timeline["DynamicPressureLabel"]
        else:
            # Create a default "Medium" label if no pressure data available
            df_timeline["BowlingPressureLabel"] = "Medium"
    
    # Filter to striker deliveries only if DeliveryType exists
    if "DeliveryType" in df_timeline.columns:
        df_timeline = df_timeline[df_timeline["DeliveryType"] == "striker"]
    
    if df_timeline.empty:
        print("No striker deliveries found")
        return empty_fig
    
    # Create a ball number for x-axis
    df_timeline["BallNumber"] = df_timeline["Over"] * 6 + df_timeline["Ball_In_Over"]
    
    # Print some stats about the pressure labels
    print(f"Batting pressure distribution: {df_timeline['BattingPressureLabel'].value_counts().to_dict()}")
    print(f"Bowling pressure distribution: {df_timeline['BowlingPressureLabel'].value_counts().to_dict()}")
    
    # Create a combined column for hover text
    hover_cols = ["Batter", "Bowler", "Runs_Total"]
    available_hover_cols = [col for col in hover_cols if col in df_timeline.columns]
    
    hover_text = []
    for _, row in df_timeline.iterrows():
        text = f"Over: {row['Over']}.{row['Ball_In_Over']}<br>"
        for col in available_hover_cols:
            text += f"{col}: {row[col]}<br>"
        text += f"Batting Pressure: {row['BattingPressureLabel']}<br>"
        text += f"Bowling Pressure: {row['BowlingPressureLabel']}"
        hover_text.append(text)
    
    df_timeline["HoverText"] = hover_text
    
    # Create figure
    fig = go.Figure()
    
    # Define pressure mapping with fallback
    pressure_map = {
        "Low": 1,
        "Medium": 2,
        "High": 3,
        "Extreme": 4
    }
    
    # Function to safely map pressure labels
    def safe_pressure_map(label):
        if pd.isna(label):
            return 2  # Default to Medium
        return pressure_map.get(label, 2)  # Default to Medium if not found
    
    # Add batting pressure line
    batting_y = df_timeline["BattingPressureLabel"].apply(safe_pressure_map)
    fig.add_trace(go.Scatter(
        x=df_timeline["BallNumber"],
        y=batting_y,
        mode="lines+markers",
        name="Batting Pressure",
        line=dict(color="#4CAF50", width=3),
        marker=dict(size=8, symbol="circle"),
        hovertext=df_timeline["HoverText"],
        hoverinfo="text"
    ))
    
    # Add bowling pressure line
    bowling_y = df_timeline["BowlingPressureLabel"].apply(safe_pressure_map)
    fig.add_trace(go.Scatter(
        x=df_timeline["BallNumber"],
        y=bowling_y,
        mode="lines+markers",
        name="Bowling Pressure",
        line=dict(color="#F44336", width=3),
        marker=dict(size=8, symbol="circle"),
        hovertext=df_timeline["HoverText"],
        hoverinfo="text"
    ))
    
    # Add wicket markers if Wicket column exists
    if "Wicket" in df_timeline.columns:
        wickets = df_timeline[df_timeline["Wicket"] == 1]
        if not wickets.empty:
            wicket_texts = []
            for _, row in wickets.iterrows():
                text = "Wicket!<br>"
                if "BatterDismissed" in row and pd.notna(row["BatterDismissed"]):
                    text += f"{row['BatterDismissed']}"
                else:
                    text += "Batter"
                
                if "Bowler" in row and pd.notna(row["Bowler"]):
                    text += f" dismissed by {row['Bowler']}"
                wicket_texts.append(text)
            
            fig.add_trace(go.Scatter(
                x=wickets["BallNumber"],
                y=[4.5] * len(wickets),  # Position above the pressure lines
                mode="markers",
                name="Wicket",
                marker=dict(
                    symbol="x",
                    size=12,
                    color="red",
                    line=dict(width=2, color="white")
                ),
                hovertext=wicket_texts,
                hoverinfo="text"
            ))
    
    # Add boundary markers if Runs_Batter column exists
    if "Runs_Batter" in df_timeline.columns:
        boundaries = df_timeline[(df_timeline["Runs_Batter"] == 4) | (df_timeline["Runs_Batter"] == 6)]
        if not boundaries.empty:
            boundary_texts = []
            for _, row in boundaries.iterrows():
                text = "Six!" if row["Runs_Batter"] == 6 else "Four!"
                text += "<br>"
                if "Batter" in row and pd.notna(row["Batter"]):
                    text += f"{row['Batter']}"
                if "Bowler" in row and pd.notna(row["Bowler"]):
                    text += f" hit off {row['Bowler']}"
                boundary_texts.append(text)
            
            fig.add_trace(go.Scatter(
                x=boundaries["BallNumber"],
                y=[0.5] * len(boundaries),  # Position below the pressure lines
                mode="markers",
                name="Boundary",
                marker=dict(
                    symbol="star",
                    size=12,
                    color=boundaries["Runs_Batter"].map({4: "blue", 6: "purple"}),
                    line=dict(width=1, color="white")
                ),
                hovertext=boundary_texts,
                hoverinfo="text"
            ))
    
    # Update layout
    fig.update_layout(
        title="Pressure Timeline",
        xaxis_title="Ball",
        yaxis=dict(
            title="Pressure Level",
            tickvals=[1, 2, 3, 4],
            ticktext=["Low", "Medium", "High", "Extreme"],
            range=[0, 5]  # Extend range to accommodate markers
        ),
        template="plotly_dark",
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(t=50, l=0, r=0, b=0),
        height=400,  # Explicitly set height
        width=800    # Explicitly set width
    )
    
    # Add over markers
    overs = sorted(df_timeline["Over"].unique())
    for over in overs:
        if over % 5 == 0 or over == 1:  # Add vertical line every 5 overs and at over 1
            fig.add_vline(
                x=over * 6,
                line_width=1,
                line_dash="dash",
                line_color="rgba(255, 255, 255, 0.3)"
            )
            fig.add_annotation(
                x=over * 6,
                y=0,
                text=f"Over {over}",
                showarrow=False,
                yshift=-20,
                font=dict(size=10, color="white")
            )
    
    return fig


def plot_pressure_timeline_simple(df_innings: pd.DataFrame) -> go.Figure:
    """
    Create a simplified timeline visualization of pressure throughout an innings.
    
    Parameters:
    -----------
    df_innings : pd.DataFrame
        Ball-by-ball data for a single innings
        
    Returns:
    --------
    go.Figure
        Plotly figure with pressure timeline
    """
    # Create a basic figure
    fig = go.Figure()
    
    # Add a simple trace even if there's no data
    fig.add_trace(go.Scatter(
        x=[1, 2, 3, 4, 5],
        y=[1, 2, 3, 2, 1],
        mode="lines+markers",
        name="Sample Data",
        line=dict(color="#4CAF50", width=3)
    ))
    
    # Update layout
    fig.update_layout(
        title="Pressure Timeline (Sample Data)",
        xaxis_title="Ball",
        yaxis_title="Pressure Level",
        template="plotly_dark",
        height=400,
        width=800
    )
    
    # If we have actual data, try to plot it
    if not df_innings.empty and "Over" in df_innings.columns and "Ball_In_Over" in df_innings.columns:
        try:
            # Create a simplified timeline with just the basic data
            df = df_innings.copy()
            
            # Filter to striker deliveries only if DeliveryType exists
            if "DeliveryType" in df.columns:
                df = df[df["DeliveryType"] == "striker"]
            
            if df.empty:
                return fig  # Return sample data if no striker deliveries
            
            df["BallNumber"] = df["Over"] * 6 + df["Ball_In_Over"]
            
            # Define pressure mapping
            pressure_map = {
                "Low": 1,
                "Medium": 2,
                "High": 3,
                "Extreme": 4
            }
            
            # Clear the figure for real data
            fig = go.Figure()
            
            # Add batting pressure if available
            if "BattingPressureLabel" in df.columns:
                # Map pressure labels to numeric values, defaulting to Medium (2) if not found
                batting_y = df["BattingPressureLabel"].map(pressure_map).fillna(2)
                
                fig.add_trace(go.Scatter(
                    x=df["BallNumber"],
                    y=batting_y,
                    mode="lines+markers",
                    name="Batting Pressure",
                    line=dict(color="#4CAF50", width=3),
                    marker=dict(size=8, symbol="circle")
                ))
            
            # Add bowling pressure if available
            if "BowlingPressureLabel" in df.columns:
                # Map pressure labels to numeric values, defaulting to Medium (2) if not found
                bowling_y = df["BowlingPressureLabel"].map(pressure_map).fillna(2)
                
                fig.add_trace(go.Scatter(
                    x=df["BallNumber"],
                    y=bowling_y,
                    mode="lines+markers",
                    name="Bowling Pressure",
                    line=dict(color="#F44336", width=3),
                    marker=dict(size=8, symbol="circle")
                ))
            
            # Add wicket markers if Wicket column exists
            if "Wicket" in df.columns:
                wickets = df[df["Wicket"] == 1]
                if not wickets.empty:
                    fig.add_trace(go.Scatter(
                        x=wickets["BallNumber"],
                        y=[4.5] * len(wickets),  # Position above the pressure lines
                        mode="markers",
                        name="Wicket",
                        marker=dict(
                            symbol="x",
                            size=12,
                            color="red",
                            line=dict(width=2, color="white")
                        )
                    ))
            
            # Add boundary markers if Runs_Batter column exists
            if "Runs_Batter" in df.columns:
                boundaries = df[(df["Runs_Batter"] == 4) | (df["Runs_Batter"] == 6)]
                if not boundaries.empty:
                    fig.add_trace(go.Scatter(
                        x=boundaries["BallNumber"],
                        y=[0.5] * len(boundaries),  # Position below the pressure lines
                        mode="markers",
                        name="Boundary",
                        marker=dict(
                            symbol="star",
                            size=12,
                            color=boundaries["Runs_Batter"].map({4: "blue", 6: "purple"}),
                            line=dict(width=1, color="white")
                        )
                    ))
            
            # Update layout
            fig.update_layout(
                title="Pressure Timeline",
                xaxis_title="Ball",
                yaxis=dict(
                    title="Pressure Level",
                    tickvals=[1, 2, 3, 4],
                    ticktext=["Low", "Medium", "High", "Extreme"],
                    range=[0, 5]  # Extend range to accommodate markers
                ),
                template="plotly_dark",
                hovermode="closest",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(t=50, l=0, r=0, b=0),
                height=400,
                width=800
            )
            
            # Add over markers
            overs = sorted(df["Over"].unique())
            for over in overs:
                if over % 5 == 0 or over == 1:  # Add vertical line every 5 overs and at over 1
                    fig.add_vline(
                        x=over * 6,
                        line_width=1,
                        line_dash="dash",
                        line_color="rgba(255, 255, 255, 0.3)"
                    )
                    fig.add_annotation(
                        x=over * 6,
                        y=0,
                        text=f"Over {over}",
                        showarrow=False,
                        yshift=-20,
                        font=dict(size=10, color="white")
                    )
            
        except Exception as e:
            print(f"Error plotting pressure timeline: {e}")
            # Keep the sample data if there's an error
    
    return fig


def plot_economy_comparison(df: pd.DataFrame, bowler: Optional[str] = None) -> go.Figure:
    """
    Create a bar chart comparing bowler economy rates across different pressure levels.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Ball-by-ball data
    bowler : Optional[str]
        Bowler name to highlight
        
    Returns:
    --------
    go.Figure
        Bar chart figure
    """
    # Apply robust fix
    from pressure_calculations import ensure_pressure_labels_robust
    df = ensure_pressure_labels_robust(df)
    
    # Now it's safe to compute metrics - BowlingPressureLabel is guaranteed to exist
    from metrics_calculations import compute_bowling_metrics_by_pressure
    pressure_metrics = compute_bowling_metrics_by_pressure(df, pressure_col="BowlingPressureLabel", aggregate=True)
    
    # If no data available, return empty figure
    if pressure_metrics.empty:
        return go.Figure()
    
    # Create visualization
    fig = px.bar(
        pressure_metrics, 
        x="Pressure", 
        y="Economy",
        color="Pressure",
        title="Economy Rate by Pressure Level",
        labels={"Economy": "Economy Rate (runs per over)"},
        color_discrete_map={"Low": "#4CAF50", "Medium": "#FFC107", "High": "#F44336"}
    )
    
    # Apply consistent order to pressure levels
    fig.update_xaxes(categoryorder="array", categoryarray=PRESSURE_LEVEL_ORDER[:3])
    
    # Add text labels on bars
    fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')
    
    # Styling
    fig.update_layout(
        template="plotly_dark",
        legend_title_text="Pressure Level",
        margin=dict(t=50, l=0, r=0, b=0)
    )
    
    return fig


def plot_dot_ball_comparison(df: pd.DataFrame, bowler: Optional[str] = None) -> go.Figure:
    """
    Create a bar chart comparing dot ball percentages across different pressure levels.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Ball-by-ball data
    bowler : Optional[str]
        Bowler name to highlight
        
    Returns:
    --------
    go.Figure
        Bar chart figure
    """
    # Apply robust fix
    from pressure_calculations import ensure_pressure_labels_robust
    df = ensure_pressure_labels_robust(df)
    
    # Now it's safe to compute metrics - BowlingPressureLabel is guaranteed to exist
    from metrics_calculations import compute_bowling_metrics_by_pressure
    pressure_metrics = compute_bowling_metrics_by_pressure(df, pressure_col="BowlingPressureLabel", aggregate=True)
    
    # If no data available, return empty figure
    if pressure_metrics.empty:
        return go.Figure()
    
    # Create visualization
    fig = px.bar(
        pressure_metrics, 
        x="Pressure", 
        y="DotBallPct",
        color="Pressure",
        title="Dot Ball % by Pressure Level",
        labels={"DotBallPct": "Dot Ball %"},
        color_discrete_map={"Low": "#4CAF50", "Medium": "#FFC107", "High": "#F44336"}
    )
    
    # Apply consistent order to pressure levels
    fig.update_xaxes(categoryorder="array", categoryarray=PRESSURE_LEVEL_ORDER[:3])
    
    # Add text labels on bars
    fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
    
    # Styling
    fig.update_layout(
        template="plotly_dark",
        legend_title_text="Pressure Level",
        margin=dict(t=50, l=0, r=0, b=0)
    )
    
    return fig


def plot_batting_pressure_metrics(df: pd.DataFrame) -> Dict[str, go.Figure]:
    """
    Create visualizations of batting performance metrics by pressure level.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Ball-by-ball data
        
    Returns:
    --------
    Dict[str, go.Figure]
        Dictionary of Plotly figures
    """
    # Always use BattingPressureLabel
    return plot_metrics_by_pressure(df, "BattingPressureLabel", "batting")


def plot_bowling_pressure_metrics(df: pd.DataFrame) -> Dict[str, go.Figure]:
    """
    Create visualizations of bowling performance metrics by pressure level.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Ball-by-ball data
        
    Returns:
    --------
    Dict[str, go.Figure]
        Dictionary of Plotly figures
    """
    # Always use BowlingPressureLabel
    return plot_metrics_by_pressure(df, "BowlingPressureLabel", "bowling")


def plot_metrics_by_pressure(df: pd.DataFrame, pressure_col: str, metric_type: str = "batting") -> Dict[str, go.Figure]:
    """
    Create visualizations of performance metrics by pressure level.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Ball-by-ball data
    pressure_col : str
        Column name containing pressure labels
    metric_type : str
        Either "batting" or "bowling"
        
    Returns:
    --------
    Dict[str, go.Figure]
        Dictionary of Plotly figures
    """
    # Ensure we have pressure labels
    from pressure_calculations import ensure_pressure_labels_robust
    df = ensure_pressure_labels_robust(df)
    
    # Calculate metrics based on metric_type
    if metric_type == "batting":
        from metrics_calculations import compute_batting_metrics_by_pressure
        metrics = compute_batting_metrics_by_pressure(df, pressure_col=pressure_col, aggregate=True)
        primary_metric = "StrikeRate"
        secondary_metric = "DotBallPct"
        primary_label = "Strike Rate"
        secondary_label = "Dot Ball %"
    else:  # bowling
        from metrics_calculations import compute_bowling_metrics_by_pressure
        metrics = compute_bowling_metrics_by_pressure(df, pressure_col=pressure_col, aggregate=True)
        primary_metric = "Economy"
        secondary_metric = "DotBallPct"
        primary_label = "Economy Rate"
        secondary_label = "Dot Ball %"
    
    if metrics.empty:
        # Return empty figures if no data
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No Data Available",
            template="plotly_dark",
            height=400,
            width=800
        )
        return {
            "primary": empty_fig,
            "secondary": empty_fig
        }
    
    # Define color map
    color_map = {
        "Low": "#4CAF50",
        "Medium": "#FFC107", 
        "High": "#F44336",
        "Extreme": "#9C27B0"
    }
    
    # Determine which pressure column name is actually in the metrics DataFrame
    if pressure_col in metrics.columns:
        actual_pressure_col = pressure_col
    elif "Pressure" in metrics.columns:
        actual_pressure_col = "Pressure"
    else:
        # If neither exists, we can't create the charts
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No Pressure Data Available",
            template="plotly_dark",
            height=400,
            width=800
        )
        return {
            "primary": empty_fig,
            "secondary": empty_fig
        }
    
    # Create primary metric figure
    fig_primary = px.bar(
        metrics,
        x=actual_pressure_col,
        y=primary_metric,
        color=actual_pressure_col,
        title=f"{primary_label} by Pressure Level",
        color_discrete_map=color_map,
        labels={primary_metric: primary_label, actual_pressure_col: "Pressure"}
    )
    
    # Apply styling to primary figure
    fig_primary.update_xaxes(categoryorder="array", categoryarray=PRESSURE_LEVEL_ORDER)
    fig_primary.update_traces(texttemplate='%{y:.1f}', textposition='outside')
    fig_primary.update_layout(
        template="plotly_dark",
        legend_title_text="Pressure Level",
        margin=dict(t=50, l=0, r=0, b=0),
        height=400,  # Explicitly set height
        width=800    # Explicitly set width
    )
    
    # Create secondary metric figure
    fig_secondary = px.bar(
        metrics,
        x=actual_pressure_col,
        y=secondary_metric,
        color=actual_pressure_col,
        title=f"{secondary_label} by Pressure Level",
        color_discrete_map=color_map,
        labels={secondary_metric: secondary_label, actual_pressure_col: "Pressure"}
    )
    
    # Apply styling to secondary figure
    fig_secondary.update_xaxes(categoryorder="array", categoryarray=PRESSURE_LEVEL_ORDER)
    fig_secondary.update_traces(texttemplate='%{y:.1f}%' if secondary_metric == "DotBallPct" else '%{y:.2f}', 
                               textposition='outside')
    fig_secondary.update_layout(
        template="plotly_dark",
        legend_title_text="Pressure Level",
        margin=dict(t=50, l=0, r=0, b=0),
        height=400,  # Explicitly set height
        width=800    # Explicitly set width
    )
    
    return {
        "primary": fig_primary,
        "secondary": fig_secondary
    }


def plot_batsman_runs_by_bowler(df_innings: pd.DataFrame, batter: Optional[str] = None) -> go.Figure:
    """
    Create a horizontal stacked bar chart showing runs scored by each batsman against different bowlers.
    
    Parameters:
    -----------
    df_innings : pd.DataFrame
        Ball-by-ball data for a single innings
    batter : Optional[str]
        Specific batter to highlight, if None show all batters
        
    Returns:
    --------
    go.Figure
        Plotly figure with stacked bar chart
    """
    if df_innings.empty:
        return go.Figure()
    
    # Filter to striker deliveries only
    df = df_innings[df_innings["DeliveryType"] == "striker"].copy()
    
    # Filter to specific batter if provided
    if batter is not None:
        df = df[df["Batter"] == batter]
    
    # Check if we have the necessary columns
    if "Batter" not in df.columns or "Bowler" not in df.columns or "Runs_Batter" not in df.columns:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Missing required columns for batsman runs analysis",
            template="plotly_dark",
            height=400,
            width=800
        )
        return empty_fig
    
    # Calculate runs by batter and bowler
    runs_by_bowler = df.groupby(["Batter", "Bowler"]).agg(
        Runs=("Runs_Batter", "sum"),
        Balls=("IsBattingDelivery", "sum")
    ).reset_index()
    
    # Calculate strike rate
    runs_by_bowler["StrikeRate"] = (runs_by_bowler["Runs"] / runs_by_bowler["Balls"] * 100).round(2)
    
    # Create hover text
    runs_by_bowler["HoverText"] = runs_by_bowler.apply(
        lambda row: f"{row['Batter']} vs {row['Bowler']}<br>Runs: {row['Runs']}<br>Balls: {row['Balls']}<br>SR: {row['StrikeRate']}",
        axis=1
    )
    
    # Determine batting order based on first appearance
    batting_order = {}
    seen_batters = set()
    
    # Go through each ball in the innings in chronological order
    for _, row in df.sort_values(["Over", "Ball_In_Over"]).iterrows():
        batter = row.get("Batter")
        if batter and batter not in seen_batters:
            batting_order[batter] = len(seen_batters)
            seen_batters.add(batter)
    
    # Get batters in batting order (limit to top 10 for readability)
    ordered_batters = sorted(batting_order.keys(), key=lambda x: batting_order[x])[:10]
    
    # Filter to ordered batters
    runs_by_bowler_filtered = runs_by_bowler[runs_by_bowler["Batter"].isin(ordered_batters)]
    
    # Create figure
    fig = go.Figure()
    
    # Get unique bowlers
    bowlers = sorted(runs_by_bowler_filtered["Bowler"].unique())
    
    # Create a color scale for bowlers
    colors = px.colors.qualitative.Plotly
    color_map = {bowler: colors[i % len(colors)] for i, bowler in enumerate(bowlers)}
    
    # Add bars for each bowler
    for bowler in bowlers:
        bowler_data = runs_by_bowler_filtered[runs_by_bowler_filtered["Bowler"] == bowler]
        
        # Create a DataFrame with all batters to ensure proper ordering
        full_data = pd.DataFrame({"Batter": ordered_batters})
        merged_data = full_data.merge(bowler_data, on="Batter", how="left")
        merged_data["Runs"] = merged_data["Runs"].fillna(0)
        merged_data["HoverText"] = merged_data.apply(
            lambda row: (f"{row['Batter']} vs {bowler}<br>Runs: {row['Runs']}<br>" +
                        (f"Balls: {row['Balls']}<br>SR: {row['StrikeRate']}" if pd.notna(row.get('Balls')) else "No balls faced")),
            axis=1
        )
        
        fig.add_trace(go.Bar(
            y=merged_data["Batter"],  # Use y instead of x for horizontal bars
            x=merged_data["Runs"],    # Use x instead of y for horizontal bars
            name=bowler,
            marker_color=color_map[bowler],
            hovertext=merged_data["HoverText"],
            hoverinfo="text",
            orientation='h'  # Horizontal bars
        ))
    
    # Update layout
    fig.update_layout(
        title="Runs Scored by Batsman Against Each Bowler",
        yaxis_title="Batsman",  # Swap x and y titles
        xaxis_title="Runs",
        barmode="stack",
        template="plotly_dark",
        legend_title="Bowler",
        height=600,  # Increase height for better readability
        width=800
    )
    
    # Reverse y-axis to show batters in batting order (top to bottom)
    fig.update_layout(yaxis={'categoryorder': 'array', 'categoryarray': ordered_batters[::-1]})
    
    return fig


def plot_partnerships(df_innings: pd.DataFrame) -> Dict[str, go.Figure]:
    """
    Create a visualization of batsman contributions for an innings.
    
    Parameters:
    -----------
    df_innings : pd.DataFrame
        Ball-by-ball data for a single innings
        
    Returns:
    --------
    Dict[str, go.Figure]
        Dictionary of Plotly figures
    """
    # Check if we have the necessary columns
    df = df_innings.copy()
    
    # Filter to striker deliveries only
    if "DeliveryType" in df.columns:
        df = df[df["DeliveryType"] == "striker"]
    
    # Check for required columns
    if "Batter" not in df.columns:
        # Create a simple message figure
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Missing required columns for batsman analysis",
            template="plotly_dark",
            height=400,
            width=800
        )
        return {"partnership_viz": empty_fig}
    
    # Sort by ball progression
    if "Over" in df.columns and "Ball_In_Over" in df.columns:
        df = df.sort_values(["Over", "Ball_In_Over"])
    
    # Determine which runs column to use
    runs_col = None
    for col in ["Runs_Total", "Runs_Batter", "Runs"]:
        if col in df.columns:
            runs_col = col
            break
    
    if not runs_col:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No runs column found for batsman analysis",
            template="plotly_dark",
            height=400,
            width=800
        )
        return {"partnership_viz": empty_fig}
    
    # Create a simple batter contribution chart
    batter_runs = df.groupby("Batter")[runs_col].sum().reset_index()
    batter_balls = df.groupby("Batter").size().reset_index(name="Balls")
    
    # Merge runs and balls
    batter_stats = pd.merge(batter_runs, batter_balls, on="Batter")
    
    # Calculate strike rate
    batter_stats["StrikeRate"] = (batter_stats[runs_col] / batter_stats["Balls"] * 100).round(1)
    
    # Sort by batting order (based on first appearance)
    batter_order = {}
    for i, batter in enumerate(df["Batter"].drop_duplicates()):
        batter_order[batter] = i
    
    batter_stats["Order"] = batter_stats["Batter"].map(batter_order)
    batter_stats = batter_stats.sort_values("Order")
    
    # Create a horizontal bar chart
    fig = go.Figure()
    
    # Add bars for each batter
    for i, row in batter_stats.iterrows():
        fig.add_trace(go.Bar(
            y=[row["Batter"]],
            x=[row[runs_col]],
            orientation="h",
            name=row["Batter"],
            text=f"{row[runs_col]} runs ({row['Balls']} balls, SR: {row['StrikeRate']})",
            textposition="outside",
            hoverinfo="text",
            marker_color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
        ))
    
    # Update layout
    fig.update_layout(
        title="Batsman Contributions",
        xaxis_title="Runs",
        yaxis_title="Batsman",
        template="plotly_dark",
        height=500,
        width=800,
        showlegend=False,
        barmode="stack"
    )
    
    return {"partnership_viz": fig} 