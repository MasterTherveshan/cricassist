import streamlit as st
import pandas as pd
import numpy as np
import json
import glob
import os
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any

# Import our modules
from data_loaders import load_and_combine_data
from pressure_calculations import ensure_pressure_labels_robust, order_pressure_levels, PRESSURE_LEVEL_ORDER
from metrics_calculations import (
    compute_batting_metrics_by_pressure,
    compute_bowling_metrics_by_pressure,
    compute_overall_batting_metrics,
    compute_advanced_batting_metrics,
    compute_advanced_bowling_metrics,
    compute_match_ppi_batting,
    compute_match_ppi_bowling,
    get_default_batters_pressure,
    get_default_bowlers_pressure,
    get_default_batters_adv,
    get_default_bowlers_adv
)
from charts import (
    create_radar_chart,
    plot_pressure_timeline,
    plot_batting_pressure_metrics,
    plot_bowling_pressure_metrics,
    plot_economy_comparison,
    plot_dot_ball_comparison,
    plot_pressure_timeline_simple,
    plot_batsman_runs_by_bowler,
    plot_partnerships
)
from scorecards import (
    build_batting_scorecard,
    build_bowling_scorecard,
    build_innings_summary,
    format_innings_summary_html,
    format_batting_scorecard_html,
    format_bowling_scorecard_html
)


# Define custom CSS for styling the entire app
def inject_custom_css():
    """Inject custom CSS to match the blue theme from the image."""
    st.markdown("""
    <style>
    /* Main app background and text */
    .stApp {
        background-color: #1e5383;
        color: #FFFFFF;
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-12oz5g7 {
        background-color: #1a4a76;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF;
    }
    
    h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
    }
    
    h2 {
        font-size: 2rem;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #1a4a76;
        padding: 0px 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1a4a76;
        border-radius: 4px 4px 0 0;
        padding: 10px 20px;
        border: none;
        color: #FFFFFF;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1e5383;
        border-bottom: 2px solid #8cdcfc;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #8cdcfc;
        color: #1a4a76;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    
    .stButton > button:hover {
        background-color: #65c6f3;
    }
    
    /* Dataframe styling */
    .dataframe {
        border: 1px solid #8cdcfc;
        border-radius: 4px;
    }
    
    .dataframe th {
        background-color: #1a4a76;
        color: white;
        padding: 8px;
        border-bottom: 1px solid #8cdcfc;
    }
    
    .dataframe td {
        padding: 8px;
        border-bottom: 1px solid #8cdcfc;
        background-color: #1e5383;
    }
    
    /* Helper text styling */
    .helper-text {
        background-color: rgba(26, 74, 118, 0.7);
        border-left: 4px solid #8cdcfc;
        padding: 10px 15px;
        margin-bottom: 20px;
        font-size: 0.9rem;
        color: #FFFFFF;
        border-radius: 0 4px 4px 0;
    }
    
    /* Match info container styling */
    .match-info-container {
        background-color: #1a4a76;
        border-radius: 8px;
        border: 1px solid #8cdcfc;
        padding: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .match-info-header h3 {
        margin: 0;
        padding-bottom: 0.2rem;
        color: #FFFFFF;
    }
    
    .match-file {
        font-size: 0.8rem;
        color: #b8d4f5;
        margin-bottom: 0.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #8cdcfc;
    }
    
    .match-info-details {
        display: flex;
        flex-wrap: wrap;
        margin-top: 0.5rem;
    }
    
    .match-info-item {
        flex: 1;
        min-width: 150px;
        padding: 0.5rem;
    }
    
    .match-info-label {
        display: block;
        font-size: 0.8rem;
        color: #b8d4f5;
        margin-bottom: 0.2rem;
    }
    
    .match-info-value {
        display: block;
        font-size: 1.1rem;
        font-weight: 600;
        color: #FFFFFF;
    }
    
    /* Metric cards styling to match the image */
    [data-testid="stMetric"] {
        background-color: #1a4a76;
        border-radius: 8px;
        padding: 10px 15px;
        border: 1px solid #8cdcfc;
        margin-bottom: 10px;
    }
    
    div[data-testid="stMetricValue"] {
        color: white;
        font-size: 2rem;
        font-weight: bold;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #b8d4f5;
    }
    
    /* Section headers styling */
    .section-header {
        color: #FFFFFF;
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 30px;
        margin-bottom: 15px;
        padding-bottom: 8px;
        border-bottom: 1px solid #8cdcfc;
    }
    
    /* Dashboard title styling */
    .dashboard-title {
        font-size: 2rem;
        font-weight: bold;
        color: white;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="Cricket Analytics Dashboard",
        page_icon="🏏",
        layout="wide"
    )

    # Inject custom CSS
    inject_custom_css()

    # App title
    st.title("Cricket Analytics Dashboard")

    # Sidebar for data loading
    with st.sidebar:
        st.header("Data Source")
        data_dir = st.text_input("Data Directory", "data")

        if st.button("Load Data"):
            with st.spinner("Loading match data..."):
                df_all = load_and_combine_data(data_dir)
                st.session_state["df_all"] = df_all
                st.session_state["data_loaded"] = True
                st.success(f"Loaded {len(df_all['Match_File'].unique())} matches with {len(df_all)} deliveries")

        # Filters (only show if data is loaded)
        if "data_loaded" in st.session_state and st.session_state["data_loaded"]:
            st.header("Filters")

            # Season filter
            if "Season" in st.session_state["df_all"].columns:
                seasons = ["All"] + sorted(st.session_state["df_all"]["Season"].unique().tolist())
                selected_season = st.selectbox("Season", seasons)

            # Team filter
            if "Batting_Team" in st.session_state["df_all"].columns:
                teams = ["All"] + sorted(st.session_state["df_all"]["Batting_Team"].unique().tolist())
                selected_team = st.selectbox("Team", teams)

    # Main content area
    if "data_loaded" not in st.session_state or not st.session_state["data_loaded"]:
        st.info("Please load data from the sidebar to begin analysis.")
        return

    # Get the data
    df_all = st.session_state["df_all"]

    # Apply filters
    filtered_df = df_all.copy()
    if "selected_season" in locals() and selected_season != "All":
        filtered_df = filtered_df[filtered_df["Season"] == selected_season]
    if "selected_team" in locals() and selected_team != "All":
        filtered_df = filtered_df[(filtered_df["Batting_Team"] == selected_team) |
                                  (filtered_df["Bowling_Team"] == selected_team)]

    # Ensure pressure labels exist
    filtered_df = ensure_pressure_labels_robust(filtered_df)

    # Create tabs for different analysis views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Match Analysis",
        "Batting Pressure",
        "Bowling Pressure",
        "Advanced Metrics",
        "Raw Data Explorer"
    ])

    # Tab 1: Match Analysis
    with tab1:
        st.header("Match Analysis")
        
        st.markdown("""
        <div class="helper-text">
            Analyze individual matches with detailed scorecards and pressure timelines.
            Select a match to view detailed ball-by-ball analysis.
        </div>
        """, unsafe_allow_html=True)
        
        # Match selector
        match_files = sorted(filtered_df["Match_File"].unique())
        
        if not match_files:
            st.warning("No matches available with the current filters.")
        else:
            selected_match = st.selectbox("Select Match", match_files)
            
            # Filter data for selected match
            match_df = filtered_df[filtered_df["Match_File"] == selected_match]
            
            # Get match metadata with better fallbacks
            match_date = match_df["Date"].iloc[0] if "Date" in match_df.columns and pd.notna(match_df["Date"].iloc[0]) else "Not Available"
            match_venue = match_df["Venue"].iloc[0] if "Venue" in match_df.columns and pd.notna(match_df["Venue"].iloc[0]) else "Not Available"
            match_type = match_df["Match_Type"].iloc[0] if "Match_Type" in match_df.columns and pd.notna(match_df["Match_Type"].iloc[0]) else "T20"
            
            # Extract teams from the match data
            teams = []
            for inn_idx in sorted(match_df["Innings_Index"].unique()):
                innings_df = match_df[match_df["Innings_Index"] == inn_idx]
                if "Batting_Team" in innings_df.columns and pd.notna(innings_df["Batting_Team"].iloc[0]):
                    teams.append(innings_df["Batting_Team"].iloc[0])
                else:
                    teams.append(f"Team {inn_idx}")
            
            # Create a more descriptive match title
            if len(teams) >= 2:
                match_title = f"{teams[0]} vs {teams[1]}"
            else:
                # Extract match name from the file name
                match_title = selected_match.replace(".json", "").replace("_", " ").title()
            
            # Display match info using Streamlit's native components with better styling
            st.subheader("Match Details")

            # Create a more informative display for match details
            match_info_html = f"""
            <div class="match-info-container">
                <div class="match-info-header">
                    <h3>{match_title}</h3>
                    <div class="match-file">{selected_match}</div>
                </div>
                <div class="match-info-details">
                    <div class="match-info-item">
                        <span class="match-info-label">Date</span>
                        <span class="match-info-value">{match_date}</span>
                    </div>
                    <div class="match-info-item">
                        <span class="match-info-label">Venue</span>
                        <span class="match-info-value">{match_venue}</span>
                    </div>
                    <div class="match-info-item">
                        <span class="match-info-label">Format</span>
                        <span class="match-info-value">{match_type}</span>
                    </div>
                </div>
            </div>
            """

            # Display the match info
            st.markdown(match_info_html, unsafe_allow_html=True)
            
            # Create innings tabs
            innings_tabs = []
            for inn_idx in sorted(match_df["Innings_Index"].unique()):
                innings_df = match_df[match_df["Innings_Index"] == inn_idx]
                batting_team = innings_df["Batting_Team"].iloc[0] if "Batting_Team" in innings_df.columns else f"Team {inn_idx}"
                innings_tabs.append(f"Innings {inn_idx} ({batting_team})")
            
            if innings_tabs:
                inn_tab1, inn_tab2 = st.tabs(innings_tabs)
                
                # First Innings
                with inn_tab1:
                    innings1_df = match_df[match_df["Innings_Index"] == 1]
                    
                    # Compute PPI for this innings
                    bat_ppi_inn1 = compute_match_ppi_batting(innings1_df)
                    bowl_ppi_inn1 = compute_match_ppi_bowling(innings1_df)
                    
                    # Build scorecards
                    innings1_summary = build_innings_summary(innings1_df)
                    batting_card1 = build_batting_scorecard(innings1_df, bat_ppi_inn1)
                    bowling_card1 = build_bowling_scorecard(innings1_df, bowl_ppi_inn1)
                    
                    # Display innings summary directly without HTML
                    st.subheader(f"{innings1_summary['BattingTeam']} - {innings1_summary['TotalRuns']}/{innings1_summary['Wickets']} ({innings1_summary['Overs']})")
                    
                    # Create metrics for key stats
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Run Rate", f"{innings1_summary['RunRate']}")
                    with col2:
                        st.metric("Boundaries", f"{innings1_summary['Fours']}×4s, {innings1_summary['Sixes']}×6s")
                    with col3:
                        st.metric("Dot Ball %", f"{innings1_summary['DotBallPct']}%")
                    with col4:
                        st.metric("Top Scorer", f"{innings1_summary['TopScorer']}")
                    
                    # Display scorecards using Streamlit's native dataframe
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Batting Scorecard")
                        st.dataframe(
                            batting_card1[["Batter", "Dismissal", "Runs", "Balls", "Fours", "Sixes", "StrikeRate"]],
                            use_container_width=True
                        )
                    
                    with col2:
                        st.subheader("Bowling Scorecard")
                        st.dataframe(
                            bowling_card1[["Bowler", "Overs", "Maidens", "Runs", "Wickets", "Economy"]],
                            use_container_width=True
                        )
                    
                    # Pressure timeline
                    st.subheader("Pressure Timeline")
                    timeline_fig = plot_pressure_timeline_simple(innings1_df)
                    st.plotly_chart(timeline_fig, use_container_width=True, key="pressure_timeline_inn1")
                    
                    # Add batsman vs bowler analysis
                    st.markdown("<div class='section-header'>Batsman vs Bowler Analysis</div>", unsafe_allow_html=True)
                    st.markdown("""
                    <div class="helper-text">
                        This chart shows how many runs each batsman scored against each bowler. 
                        Hover over the bars to see strike rates for each batsman-bowler matchup.
                    </div>
                    """, unsafe_allow_html=True)

                    batsman_bowler_fig = plot_batsman_runs_by_bowler(innings1_df)

                    # Wrap the chart in a div with our custom class
                    st.markdown("<div class='batsman-bowler-chart'>", unsafe_allow_html=True)
                    st.plotly_chart(batsman_bowler_fig, use_container_width=True, key="batsman_bowler_inn1")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Second Innings (if exists)
                if len(innings_tabs) > 1:
                    with inn_tab2:
                        innings2_df = match_df[match_df["Innings_Index"] == 2]
                        
                        # Compute PPI for this innings
                        bat_ppi_inn2 = compute_match_ppi_batting(innings2_df)
                        bowl_ppi_inn2 = compute_match_ppi_bowling(innings2_df)
                        
                        # Build scorecards
                        innings2_summary = build_innings_summary(innings2_df)
                        batting_card2 = build_batting_scorecard(innings2_df, bat_ppi_inn2)
                        bowling_card2 = build_bowling_scorecard(innings2_df, bowl_ppi_inn2)
                        
                        # Display innings summary directly without HTML
                        st.subheader(f"{innings2_summary['BattingTeam']} - {innings2_summary['TotalRuns']}/{innings2_summary['Wickets']} ({innings2_summary['Overs']})")
                        
                        # Create metrics for key stats
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Run Rate", f"{innings2_summary['RunRate']}")
                        with col2:
                            st.metric("Boundaries", f"{innings2_summary['Fours']}×4s, {innings2_summary['Sixes']}×6s")
                        with col3:
                            st.metric("Dot Ball %", f"{innings2_summary['DotBallPct']}%")
                        with col4:
                            st.metric("Top Scorer", f"{innings2_summary['TopScorer']}")
                        
                        # Display scorecards using Streamlit's native dataframe
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Batting Scorecard")
                            st.dataframe(
                                batting_card2[["Batter", "Dismissal", "Runs", "Balls", "Fours", "Sixes", "StrikeRate"]],
                                use_container_width=True
                            )
                        
                        with col2:
                            st.subheader("Bowling Scorecard")
                            st.dataframe(
                                bowling_card2[["Bowler", "Overs", "Maidens", "Runs", "Wickets", "Economy"]],
                                use_container_width=True
                            )
                        
                        # Pressure timeline
                        st.subheader("Pressure Timeline")
                        timeline_fig = plot_pressure_timeline_simple(innings2_df)
                        st.plotly_chart(timeline_fig, use_container_width=True, key="pressure_timeline_inn2")
                        
                        # Add batsman vs bowler analysis
                        st.markdown("<div class='section-header'>Batsman vs Bowler Analysis</div>", unsafe_allow_html=True)
                        st.markdown("""
                        <div class="helper-text">
                            This chart shows how many runs each batsman scored against each bowler. 
                            Hover over the bars to see strike rates for each batsman-bowler matchup.
                        </div>
                        """, unsafe_allow_html=True)

                        batsman_bowler_fig = plot_batsman_runs_by_bowler(innings2_df)

                        # Wrap the chart in a div with our custom class
                        st.markdown("<div class='batsman-bowler-chart'>", unsafe_allow_html=True)
                        st.plotly_chart(batsman_bowler_fig, use_container_width=True, key="batsman_bowler_inn2")
                        st.markdown("</div>", unsafe_allow_html=True)

    # Tab 2: Batting Pressure Analysis
    with tab2:
        st.header("Batting Pressure Analysis")

        st.markdown("""
        <div class="helper-text">
            Analyze how batters perform under different pressure levels. 
            Pressure is calculated based on match situation, required run rate, wickets fallen, and other factors.
        </div>
        """, unsafe_allow_html=True)

        # Overall batting metrics by pressure
        st.subheader("Overall Batting Performance by Pressure")

        # Get batting metrics by pressure
        batting_figs = plot_batting_pressure_metrics(filtered_df)

        # Display the charts
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(batting_figs["primary"], use_container_width=True, height=400)
        with col2:
            st.plotly_chart(batting_figs["secondary"], use_container_width=True, height=400)

        # Individual batter analysis
        st.subheader("Individual Batter Analysis")

        # Get default batters
        default_batters = get_default_batters_pressure(filtered_df)

        # Batter selector
        all_batters = sorted(filtered_df["Batter"].dropna().unique())
        selected_batters = st.multiselect(
            "Select Batters",
            all_batters,
            default=default_batters[:min(3, len(default_batters))]
        )

        if selected_batters:
            # Filter data for selected batters
            batter_df = filtered_df[filtered_df["Batter"].isin(selected_batters)]

            # Compute metrics by pressure for each batter
            batter_metrics = compute_batting_metrics_by_pressure(batter_df, aggregate=False)

            # Create a pivot table for easier comparison
            pressure_col = "BattingPressureLabel"  # Use the original column name
            pivot_sr = batter_metrics.pivot(index=pressure_col, columns="Batter", values="StrikeRate")
            pivot_dot = batter_metrics.pivot(index=pressure_col, columns="Batter", values="DotBallPct")

            # Display the pivot tables
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Strike Rate by Pressure")
                st.dataframe(pivot_sr.style.format("{:.2f}"), use_container_width=True)

            with col2:
                st.subheader("Dot Ball % by Pressure")
                st.dataframe(pivot_dot.style.format("{:.1%}"), use_container_width=True)
        else:
            st.warning("No data available for the selected batters.")

    # Tab 3: Bowling Pressure Analysis
    with tab3:
        st.header("Bowling Pressure Analysis")

        st.markdown("""
        <div class="helper-text">
            Analyze how bowlers perform under different pressure levels.
            Bowling pressure is calculated based on match situation, required run rate, and batting aggression.
        </div>
        """, unsafe_allow_html=True)

        # Overall bowling metrics by pressure
        st.subheader("Overall Bowling Performance by Pressure")

        # Get bowling metrics by pressure
        bowling_figs = plot_bowling_pressure_metrics(filtered_df)

        # Display the charts
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(bowling_figs["primary"], use_container_width=True, height=400)
        with col2:
            st.plotly_chart(bowling_figs["secondary"], use_container_width=True, height=400)

        # Individual bowler analysis
        st.subheader("Individual Bowler Analysis")

        # Get default bowlers
        default_bowlers = get_default_bowlers_pressure(filtered_df)

        # Bowler selector
        all_bowlers = sorted(filtered_df["Bowler"].dropna().unique())
        selected_bowlers = st.multiselect(
            "Select Bowlers",
            all_bowlers,
            default=default_bowlers[:min(3, len(default_bowlers))]
        )

        if selected_bowlers:
            # Filter data for selected bowlers
            bowler_df = filtered_df[filtered_df["Bowler"].isin(selected_bowlers)]

            # Compute metrics by pressure for each bowler
            bowler_metrics = compute_bowling_metrics_by_pressure(bowler_df, aggregate=False)

            # Use the actual column name that exists in the DataFrame
            if "BowlingPressureLabel" in bowler_metrics.columns:
                pressure_col = "BowlingPressureLabel"
            elif "DynamicPressureLabel" in bowler_metrics.columns:
                pressure_col = "DynamicPressureLabel"
            else:
                # If neither exists, we can't create the pivot
                st.warning("No pressure label column found in the data.")
                pressure_col = None

            if pressure_col:
                pivot_econ = bowler_metrics.pivot(index=pressure_col, columns="Bowler", values="Economy")
                pivot_dot = bowler_metrics.pivot(index=pressure_col, columns="Bowler", values="DotBallPct")

            # Display the pivot tables
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Economy Rate by Pressure")
                st.dataframe(pivot_econ.style.format("{:.2f}"), use_container_width=True)

            with col2:
                st.subheader("Dot Ball % by Pressure")
                st.dataframe(pivot_dot.style.format("{:.1f}"), use_container_width=True)
        else:
            st.warning("No data available for the selected bowlers.")

    # Tab 4: Advanced Metrics
    with tab4:
        st.header("Advanced Performance Metrics")

        # Create subtabs for batting and bowling
        adv_tab1, adv_tab2 = st.tabs(["Batting Metrics", "Bowling Metrics"])

        # Advanced Batting Metrics
        with adv_tab1:
            st.subheader("Advanced Batting Metrics")

            st.markdown("""
            <div class="helper-text">
                These metrics provide deeper insights into batting performance beyond traditional statistics.
                They include pressure performance, strike rotation efficiency, and boundary rates.
            </div>
            """, unsafe_allow_html=True)

            # Add minimum runs filter
            min_runs = st.slider("Minimum Total Runs", min_value=0, max_value=500, value=100, step=10)

            # Compute advanced batting metrics
            bat_ppi_all = compute_match_ppi_batting(filtered_df)
            # Aggregate PPI across matches
            if not bat_ppi_all.empty:
                bat_ppi_agg = bat_ppi_all.groupby("Batter")["BatPPI"].mean().reset_index()
                bat_ppi_agg.rename(columns={"BatPPI": "AvgBatPPI"}, inplace=True)
            else:
                bat_ppi_agg = None

            adv_batting = compute_advanced_batting_metrics(filtered_df, bat_ppi_agg)

            if not adv_batting.empty:
                # Apply minimum runs filter
                filtered_adv_batting = adv_batting[adv_batting["Total_Runs"] >= min_runs]
                
                if filtered_adv_batting.empty:
                    st.warning(f"No batters found with at least {min_runs} runs. Try lowering the threshold.")
                else:
                    # Display the metrics table
                    st.dataframe(
                        filtered_adv_batting.sort_values("Total_Runs", ascending=False)
                        .style.format({
                            "Total_Runs": "{:.0f}",
                            "Average": "{:.2f}",
                            "StrikeRate": "{:.2f}",
                            "BoundaryRate": "{:.1%}",
                            "DotBallPct": "{:.1%}",
                            "StrikeRotationEfficiency": "{:.1%}",
                            "Finisher": "{:.1%}",
                            "PressurePerformanceIndex": "{:.2f}",
                            "AvgBatPPI": "{:.1f}"
                        }),
                        use_container_width=True
                    )

                    # Radar chart for comparing batters
                    st.subheader("Batter Comparison")

                    # Get default batters for comparison (from filtered data)
                    default_adv_batters = get_default_batters_adv(filtered_adv_batting)

                    # Batter selector for radar chart
                    selected_adv_batters = st.multiselect(
                        "Select Batters for Comparison",
                        filtered_adv_batting["Batter"].tolist(),
                        default=default_adv_batters[:min(3, len(default_adv_batters))]
                    )

                    if selected_adv_batters:
                        # Define metrics to include in radar chart
                        radar_metrics = [
                            "StrikeRate", "Average", "BoundaryRate",
                            "DotBallPct", "StrikeRotationEfficiency", "PressurePerformanceIndex"
                        ]

                        # Create radar chart
                        radar_fig = create_radar_chart(
                            filtered_adv_batting,
                            "Batter",
                            selected_adv_batters,
                            radar_metrics,
                            "Batter Performance Comparison"
                        )

                        if radar_fig:
                            st.plotly_chart(radar_fig, use_container_width=True)
                        else:
                            st.warning("Unable to create radar chart with the selected batters.")
            else:
                st.warning("No advanced batting metrics available with the current filters.")

        # Advanced Bowling Metrics
        with adv_tab2:
            st.subheader("Advanced Bowling Metrics")

            st.markdown("""
            <div class="helper-text">
                These metrics provide deeper insights into bowling performance beyond traditional statistics.
                They include bounce-back rate, key wicket index, and death overs economy.
            </div>
            """, unsafe_allow_html=True)

            # Add minimum wickets filter
            min_wickets = st.slider("Minimum Wickets Taken", min_value=0, max_value=20, value=5, step=1)

            # Compute advanced bowling metrics
            bowl_ppi_all = compute_match_ppi_bowling(filtered_df)
            # Aggregate PPI across matches
            if not bowl_ppi_all.empty:
                bowl_ppi_agg = bowl_ppi_all.groupby("Bowler")["BowlPPI"].mean().reset_index()
                bowl_ppi_agg.rename(columns={"BowlPPI": "AvgBowlPPI"}, inplace=True)
            else:
                bowl_ppi_agg = None

            adv_bowling = compute_advanced_bowling_metrics(filtered_df, bowl_ppi_agg)

            if not adv_bowling.empty:
                # Apply minimum wickets filter
                filtered_adv_bowling = adv_bowling[adv_bowling["Wickets"] >= min_wickets]
                
                if filtered_adv_bowling.empty:
                    st.warning(f"No bowlers found with at least {min_wickets} wickets. Try lowering the threshold.")
                else:
                    # Display the metrics table
                    st.dataframe(
                        filtered_adv_bowling.sort_values("Wickets", ascending=False)
                        .style.format({
                            "Economy": "{:.2f}",
                            "StrikeRate": "{:.2f}",
                            "BounceBackRate": "{:.1%}",
                            "KeyWicketIndex": "{:.2f}",
                            "DeathOversEconomy": "{:.2f}",
                            "AvgBowlPPI": "{:.1f}"
                        }),
                        use_container_width=True
                    )

                    # Radar chart for comparing bowlers
                    st.subheader("Bowler Comparison")

                    # Get default bowlers for comparison (from filtered data)
                    default_adv_bowlers = get_default_bowlers_adv(filtered_adv_bowling)

                    # Bowler selector for radar chart
                    selected_adv_bowlers = st.multiselect(
                        "Select Bowlers for Comparison",
                        filtered_adv_bowling["Bowler"].tolist(),
                        default=default_adv_bowlers[:min(3, len(default_adv_bowlers))]
                    )

                    if selected_adv_bowlers:
                        # Define metrics to include in radar chart
                        radar_metrics = [
                            "Economy", "StrikeRate", "BounceBackRate",
                            "KeyWicketIndex", "DeathOversEconomy"
                        ]

                        # Create radar chart
                        radar_fig = create_radar_chart(
                            filtered_adv_bowling,
                            "Bowler",
                            selected_adv_bowlers,
                            radar_metrics,
                            "Bowler Performance Comparison"
                        )

                        if radar_fig:
                            st.plotly_chart(radar_fig, use_container_width=True)
                        else:
                            st.warning("Unable to create radar chart with the selected bowlers.")
            else:
                st.warning("No advanced bowling metrics available with the current filters.")

    # Tab 5: Raw Data Explorer
    with tab5:
        st.header("Raw Data Explorer")

        st.markdown("""
        <div class="helper-text">
            Explore the raw ball-by-ball data with custom filters and view the underlying data used for analysis.
        </div>
        """, unsafe_allow_html=True)

        # Column selector
        all_columns = filtered_df.columns.tolist()
        default_columns = [
            "Match_File", "Innings_Index", "Over", "Ball_In_Over",
            "Batter", "Bowler", "Runs_Batter", "Wicket",
            "BattingPressureLabel", "BowlingPressureLabel"
        ]

        # Only include default columns that actually exist
        default_columns = [col for col in default_columns if col in all_columns]

        selected_columns = st.multiselect(
            "Select Columns to Display",
            all_columns,
            default=default_columns
        )

        # Additional filters
        with st.expander("Advanced Filters"):
            col1, col2 = st.columns(2)

            with col1:
                # Filter by innings
                if "Innings_Index" in filtered_df.columns:
                    innings_options = [1, 2]
                    selected_innings_filter = st.multiselect(
                        "Innings",
                        innings_options,
                        default=innings_options
                    )

                # Filter by delivery type
                if "DeliveryType" in filtered_df.columns:
                    delivery_options = filtered_df["DeliveryType"].unique().tolist()
                    selected_delivery = st.multiselect(
                        "Delivery Type",
                        delivery_options,
                        default=["striker"]
                    )

            with col2:
                # Filter by pressure level
                if "BattingPressureLabel" in filtered_df.columns:
                    pressure_options = PRESSURE_LEVEL_ORDER
                    selected_pressure = st.multiselect(
                        "Pressure Level",
                        pressure_options,
                        default=pressure_options
                    )

                # Filter by wicket
                if "Wicket" in filtered_df.columns:
                    wicket_options = [0, 1]
                    selected_wicket = st.multiselect(
                        "Wicket",
                        wicket_options,
                        default=wicket_options
                    )

        # Apply additional filters
        display_df = filtered_df.copy()

        if "selected_innings_filter" in locals() and selected_innings_filter:
            display_df = display_df[display_df["Innings_Index"].isin(selected_innings_filter)]

        if "selected_delivery" in locals() and selected_delivery:
            display_df = display_df[display_df["DeliveryType"].isin(selected_delivery)]

        if "selected_pressure" in locals() and selected_pressure:
            if "BattingPressureLabel" in display_df.columns:
                display_df = display_df[display_df["BattingPressureLabel"].isin(selected_pressure)]
            elif "DynamicPressureLabel" in display_df.columns:
                display_df = display_df[display_df["DynamicPressureLabel"].isin(selected_pressure)]
        
        if "selected_wicket" in locals() and selected_wicket:
            display_df = display_df[display_df["Wicket"].isin(selected_wicket)]
        
        # Display the filtered data
        if selected_columns:
            st.dataframe(display_df[selected_columns], use_container_width=True)
        else:
            st.warning("Please select at least one column to display.")
        
        # Download button for CSV export
        if not display_df.empty and selected_columns:
            csv = display_df[selected_columns].to_csv(index=False)
            st.download_button(
                label="Download Data as CSV",
                data=csv,
                file_name="cricket_data.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()