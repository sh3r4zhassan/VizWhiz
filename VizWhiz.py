import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from matplotlib.colors import to_hex
import scipy.stats as stats
import seaborn as sns


st.set_page_config(page_title="VizWhiz", layout="wide")

def format_var_name(name):
    return ' '.join(word.capitalize() for word in name.split('_'))

st.title("VizWhiz - A Visualization Wizard")

st.sidebar.header("Step 1: Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data")
    st.dataframe(df)

    # Data Cleaning
    st.sidebar.header("Step 2: Data Cleaning")

    if st.sidebar.checkbox("Drop rows with null values", value=True):
        df = df.dropna()
    else:
        fill_method = st.sidebar.radio("Fill missing values with", ("Mean", "Median", "Mode"))
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype in ['float64', 'int']:
                    fill_value = df[col].mean() if fill_method == "Mean" else df[col].median()
                else:
                    fill_value = df[col].mode()[0]
                df[col].fillna(fill_value, inplace=True)

    # Outlier Removal (IQR method with adjustable threshold)
    if st.sidebar.checkbox("Remove outliers (IQR method)", value=True):
        float_cols = df.select_dtypes(include=['float64', 'int']).columns
        iqr_threshold = st.sidebar.slider("Outlier Threshold (IQR)", min_value=1.0, max_value=3.0, value=1.5)
        Q1 = df[float_cols].quantile(0.25)
        Q3 = df[float_cols].quantile(0.75)
        IQR = Q3 - Q1
        mask = ~((df[float_cols] < (Q1 - iqr_threshold * IQR)) | (df[float_cols] > (Q3 + iqr_threshold * IQR))).any(axis=1)
        df = df[mask]

    # Duplicate Removal
    if st.sidebar.checkbox("Remove duplicate rows", value=True):
        subset_columns = st.sidebar.multiselect("Columns to check for duplicates", df.columns.tolist(), default=df.columns.tolist())
        df = df.drop_duplicates(subset=subset_columns)

    st.subheader("Cleaned Data")
    st.dataframe(df)

    st.sidebar.header("Step 3: Choose Plot Types")
    plot_types = st.sidebar.multiselect(
        "Which plots do you want?",
        ["Histogram", "Box Plot", "Violin Plot", "Scatter Plot", "Line Plot"]
    )


    st.subheader("Dashboard Visualizations")

    numeric_cols = df.select_dtypes(include=['float64', 'int']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    display_names = [format_var_name(col) for col in numeric_cols]
    col_display_map = dict(zip(display_names, numeric_cols))

    def groupby_selector(key_prefix):
        if categorical_cols:
            return st.selectbox("Group by (optional)", ["None"] + categorical_cols, key=f"{key_prefix}_groupby")
        return "None"

    # --- Histogram ---
    if "Histogram" in plot_types:
        st.markdown("### Histograms")
        hist_count = st.number_input("How many histograms to show?", min_value=1, max_value=10, step=1, key="hist_count")
        for i in range(hist_count):
            with st.expander(f"Histogram #{i+1}"):

                col1, col2 = st.columns([1, 2])  # Create two columns: options on the left and plot on the right
                
                with col1:  # Options column
                    display_col = st.selectbox("Select column", display_names, key=f"hist_col_{i}")
                    col = col_display_map[display_col]
                    show_kde = st.checkbox("Show KDE (curve)", value=True, key=f"kde_{i}")
                    log_scale = st.checkbox("Log scale", value=False, key=f"log_hist_{i}")
                    groupby_col = groupby_selector(f"hist_{i}")

                    color_palette = {}
                    if groupby_col != "None":
                        unique_groups = df[groupby_col].dropna().unique()
                        default_colors = px.colors.qualitative.Plotly[:len(unique_groups)]
                        for idx, grp in enumerate(unique_groups):
                            color_palette[grp] = st.color_picker(f"Color for {grp}", value=default_colors[idx], key=f"hist_color_{i}_{grp}")
                    else:
                        default_colors = px.colors.qualitative.Plotly[:1]
                        color = st.color_picker("Color for plot", value=default_colors[0], key=f"hist_color_{i}_single")


                    

                with col2:  # Plot column
                    # Initialize the figure
                    fig = go.Figure()

                    # Apply groupby_col if it's selected
                    if groupby_col != "None":
                        for grp in df[groupby_col].dropna().unique():
                            subset = df[df[groupby_col] == grp]
                            values = np.log(subset[col]) if log_scale else subset[col]
                            fig.add_trace(go.Histogram(x=values, name=str(grp), marker_color=color_palette[grp], histnorm='probability density'))

                            # Add KDE if selected
                            if show_kde:
                                kde = stats.gaussian_kde(values)  # Smooth KDE
                                kde_x = np.linspace(min(values), max(values), 1000)  # Generate x values
                                kde_y = kde(kde_x)  # Evaluate KDE at the x values
                                fig.add_trace(go.Scatter(x=kde_x, y=kde_y, mode='lines', name=f'{grp} KDE', line=dict(color=color_palette[grp], dash='dash')))
                    else:
                        values = np.log(df[col]) if log_scale else df[col]
                        fig.add_trace(go.Histogram(x=values, name=display_col, marker_color=color, histnorm='probability density'))

                        # Add KDE if selected
                        if show_kde:
                            kde = stats.gaussian_kde(values)  # Smooth KDE
                            kde_x = np.linspace(min(values), max(values), 1000)  # Generate x values
                            kde_y = kde(kde_x)  # Evaluate KDE at the x values
                            fig.add_trace(go.Scatter(x=kde_x, y=kde_y, mode='lines', name='KDE', line=dict(color=color, dash='dash')))
                    
                    fig.update_layout(title=f"Histogram of {display_col}", 
                                    xaxis_title=display_col, 
                                    yaxis_title="Density", 
                                    bargap=0.2)

                    st.plotly_chart(fig)





    # --- Box Plot ---
    if "Box Plot" in plot_types:
        st.markdown("### Box Plots")
        box_count = st.number_input("How many box plots to show?", min_value=1, max_value=10, step=1, key="box_count")
        for i in range(box_count):
            with st.expander(f"Box Plot #{i+1}"):

                col1, col2 = st.columns([1, 2])  # Options on the left, plot on the right

                with col1:
                    selected_displays = st.multiselect(
                        "Select column(s)",
                        display_names,
                        default=[display_names[0]],
                        key=f"box_col_{i}"
                    )
                    selected_cols = [col_display_map[name] for name in selected_displays]

                    # Dropdown for point type
                    point_display = st.selectbox(
                        "Which sample points to show?",
                        options=["None", "All", "Outliers", "Suspected Outliers"],
                        index=1,
                        key=f"box_points_mode_{i}"
                    )
                    point_map = {
                        "None": False,
                        "All": "all",
                        "Outliers": "outliers",
                        "Suspected Outliers": "suspectedoutliers"
                    }
                    selected_points = point_map[point_display]

                    groupby_col = groupby_selector(f"box_{i}")

                    # Choose a color palette
                    color_palette = {}
                    if groupby_col != "None":
                        unique_groups = df[groupby_col].dropna().unique()
                        default_colors = px.colors.qualitative.Plotly[:len(unique_groups)]

                        # Create color pickers for each group in the groupby column
                        for idx, grp in enumerate(unique_groups):
                            color_palette[grp] = st.color_picker(
                                f"Color for {grp}",
                                value=to_hex(default_colors[idx]),
                                key=f"box_color_{i}_{grp}"
                            )

                    else:
                        variable_colors = {}
                        for idx, var in enumerate(selected_displays):
                            default_color = to_hex(px.colors.qualitative.Plotly[idx % 10])
                            variable_colors[var] = st.color_picker(
                                f"Color for {var}",
                                value=default_color,
                                key=f"box_color_{i}_{var}"
                            )

                with col2:
                    fig = go.Figure()

                    # Calculate the x-axis positions to separate the boxes
                    x_pos = np.arange(len(selected_cols))  # Space boxes apart by assigning unique x positions

                    if groupby_col != "None":
                        for idx, col in enumerate(selected_cols):
                            for grp_idx, grp in enumerate(df[groupby_col].dropna().unique()):
                                subset = df[df[groupby_col] == grp]
                                fig.add_trace(go.Box(
                                    y=subset[col],
                                    x=[x_pos[idx] + (grp_idx * 0.2)] * len(subset),  # Adjust x position for group
                                    name=f"{grp}",
                                    fillcolor=color_palette.get(grp, "blue"),
                                    line_color=color_palette.get(grp, "blue"),
                                    boxmean="sd",  # Show mean and standard deviation
                                    jitter=0.05,
                                    pointpos=0,
                                    showlegend=(idx == 0)  # Only show legend for the first box
                                ))
                    else:
                        for idx, col in enumerate(selected_cols):
                            fig.add_trace(go.Box(
                                y=df[col],
                                x=[x_pos[idx]] * len(df),
                                name=selected_displays[idx],
                                fillcolor=variable_colors.get(selected_displays[idx], "blue"),
                                line_color=variable_colors.get(selected_displays[idx], "blue"),
                                boxmean="sd",  # Show mean and standard deviation
                                jitter=0.05,
                                pointpos=0
                            ))

                    fig.update_layout(
                        title=f"Box Plot for {', '.join(selected_displays)}",
                        xaxis_title="Variable",
                        yaxis_title="Value",
                        showlegend=True,
                        xaxis=dict(tickvals=x_pos, ticktext=selected_displays)  # Custom x-axis labels
                    )

                    st.plotly_chart(fig)






    # --- Violin Plot ---
    if "Violin Plot" in plot_types:
        st.markdown("### Violin Plots")
        violin_count = st.number_input("How many violin plots to show?", min_value=1, max_value=10, step=1, key="violin_count")
        for i in range(violin_count):
            with st.expander(f"Violin Plot #{i+1}"):

                col1, col2 = st.columns([1, 2])  # Options on the left, plot on the right

                with col1:
                    selected_displays = st.multiselect(
                        "Select column(s)",
                        display_names,
                        default=[display_names[0]],
                        key=f"violin_col_{i}"
                    )
                    selected_cols = [col_display_map[name] for name in selected_displays]

                    # Dropdown for point type
                    point_display = st.selectbox(
                        "Which sample points to show?",
                        options=["None", "All", "Outliers", "Suspected Outliers"],
                        index=1,
                        key=f"violin_points_mode_{i}"
                    )
                    point_map = {
                        "None": False,
                        "All": "all",
                        "Outliers": "outliers",
                        "Suspected Outliers": "suspectedoutliers"
                    }
                    selected_points = point_map[point_display]

                    groupby_col = groupby_selector(f"violin_{i}")

                    # Choose a color palette
                    color_palette = {}
                    if groupby_col != "None":
                        unique_groups = df[groupby_col].dropna().unique()
                        default_colors = px.colors.qualitative.Plotly[:len(unique_groups)]

                        # Create color pickers for each group in the groupby column
                        for idx, grp in enumerate(unique_groups):
                            color_palette[grp] = st.color_picker(
                                f"Color for {grp}",
                                value=to_hex(default_colors[idx]),
                                key=f"violin_color_{i}_{grp}"
                            )

                    else:
                        variable_colors = {}
                        for idx, var in enumerate(selected_displays):
                            default_color = to_hex(px.colors.qualitative.Plotly[idx % 10])
                            variable_colors[var] = st.color_picker(
                                f"Color for {var}",
                                value=default_color,
                                key=f"violin_color_{i}_{var}"
                            )

                with col2:
                    fig = go.Figure()

                    # Calculate the x-axis positions to separate the violins
                    x_pos = np.arange(len(selected_cols))  # Space violins apart by assigning unique x positions

                    if groupby_col != "None":
                        for idx, col in enumerate(selected_cols):
                            for grp_idx, grp in enumerate(df[groupby_col].dropna().unique()):
                                subset = df[df[groupby_col] == grp]
                                fig.add_trace(go.Violin(
                                    y=subset[col],
                                    x=[x_pos[idx] + (grp_idx * 0.2)] * len(subset),  # Adjust x position for group
                                    name=f"{grp}",
                                    fillcolor=color_palette.get(grp, "blue"),
                                    line_color=color_palette.get(grp, "blue"),
                                    points=selected_points,
                                    opacity=0.8,
                                    box_visible=False,
                                    legendgroup=str(grp),
                                    showlegend=(idx == 0)  # Only show legend for the first violin
                                ))
                    else:
                        for idx, col in enumerate(selected_cols):
                            fig.add_trace(go.Violin(
                                y=df[col],
                                x=[x_pos[idx]] * len(df),
                                name=selected_displays[idx],
                                fillcolor=variable_colors.get(selected_displays[idx], "blue"),
                                line_color=variable_colors.get(selected_displays[idx], "blue"),
                                points=selected_points,
                                opacity=0.8,
                                box_visible=False
                            ))

                    fig.update_layout(
                        title=f"Violin Plot for {', '.join(selected_displays)}",
                        xaxis_title="Variable",
                        yaxis_title="Value",
                        showlegend=True,
                        xaxis=dict(tickvals=x_pos, ticktext=selected_displays)  # Custom x-axis labels
                    )

                    st.plotly_chart(fig)



    # --- Scatter Plot ---
    if "Scatter Plot" in plot_types:
        st.markdown("### Scatter Plots")
        scatter_count = st.number_input("How many scatter plots to show?", min_value=1, max_value=10, step=1, key="scatter_count")
        for i in range(scatter_count):
            with st.expander(f"Scatter Plot #{i+1}"):

                col1, col2 = st.columns([1, 2])  # Create two columns: options on the left and plot on the right

                with col1:  # Options column
                    x_display = st.selectbox("X-axis", display_names, key=f"x_scatter_{i}")
                    x_col = col_display_map[x_display]
                    y_displays = st.multiselect("Y-axis columns", display_names, default=[display_names[0]], key=f"y_scatter_{i}")
                    y_cols = [col_display_map[y] for y in y_displays]
                    show_reg = st.checkbox("Add Regression Line", value=False, key=f"reg_scatter_{i}")
                    log_scale = st.checkbox("Log scale (x & y)?", value=False, key=f"log_scatter_{i}")
                    opacity = st.slider("Point Opacity", 0.0, 1.0, 0.5, key=f"opacity_scatter_{i}")  # Renamed to Opacity
                    size = st.slider("Point size", 1, 100, 10, key=f"size_scatter_{i}")  # Default size set to 10


                    color_palette = {}

                    variable_colors = {}
                    for idx, y_display in enumerate(y_displays):
                        default_color = to_hex(px.colors.qualitative.Plotly[idx % 10])
                        variable_colors[y_display] = st.color_picker(f"Color for {y_display}", value=default_color, key=f"scatter_color_{i}_{y_display}")

                with col2:  # Plot column
                    fig = go.Figure()

                    for idx, y_display in enumerate(y_displays):
                        y = col_display_map[y_display]
                        x_vals = np.log(df[x_col]) if log_scale else df[x_col]
                        y_vals = np.log(df[y]) if log_scale else df[y]



                        # Add points if show_points is selected
                        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="markers", name=f"{y_display} Points", 
                                                marker=dict(color=variable_colors.get(y_display, "blue"), opacity=opacity, size=size)))

                        # Add regression line if selected
                        if show_reg:
                            regression_line = np.polyfit(x_vals, y_vals, 1)
                            regression_line_fn = np.poly1d(regression_line)
                            fig.add_trace(go.Scatter(x=x_vals, y=regression_line_fn(x_vals), mode="lines", 
                                                    name=f"{y_display} Regression", line=dict(color=variable_colors.get(y_display, "blue"), dash="dash")))

                    # Update the layout
                    fig.update_layout(
                        title=f"Scatter Plot: {', '.join(y_displays)} vs {x_display}",
                        xaxis_title=x_display,
                        yaxis_title="Value",
                        showlegend=True
                    )

                    st.plotly_chart(fig)



    # --- Line Plot ---
    if "Line Plot" in plot_types:
        st.markdown("### Line Plots")
        line_count = st.number_input("How many line plots to show?", min_value=1, max_value=10, step=1, key="line_count")
        for i in range(line_count):
            with st.expander(f"Line Plot #{i+1}"):

                col1, col2 = st.columns([1, 2])
                with col1:
                    x_display = st.selectbox("X-axis", display_names, key=f"x_line_{i}")
                    x_col = col_display_map[x_display]
                    y_displays = st.multiselect("Y-axis columns", display_names, default=[display_names[0]], key=f"y_line_{i}")
                    y_cols = [col_display_map[y] for y in y_displays]
                    show_points = st.checkbox("Points as circles", value=True, key=f"show_points_line_{i}")
                    log_scale = st.checkbox("Log scale (x & y)?", value=False, key=f"log_line_{i}")
                    opacity = st.slider("Point Opacity (if shown)", 0.0, 1.0, 0.5, key=f"opacity_line_{i}")  # Renamed to Opacity
                    size = st.slider("Point size (if shown)", 1, 100, 10, key=f"size_line_{i}")  # Default size set to 10

                    variable_colors = {}
                    for idx, y_display in enumerate(y_displays):
                        default_color = to_hex(px.colors.qualitative.Plotly[idx % 10])
                        variable_colors[y_display] = st.color_picker(f"Color for {y_display}", value=default_color, key=f"line_color_{i}_{y_display}")

                with col2:
                    fig = go.Figure()

                    for idx, y_display in enumerate(y_displays):
                        y = col_display_map[y_display]
                        x_vals = np.log(df[x_col]) if log_scale else df[x_col]
                        y_vals = np.log(df[y]) if log_scale else df[y]

                        # Only plot points if 'show_points' is selected
                        mode = "lines+markers" if show_points else "lines"  # Adjust mode to either include markers or not

                        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode=mode, name=y_display, 
                                                line=dict(color=variable_colors.get(y_display, "blue")), 
                                                marker=dict(size=size, opacity=opacity)))

                    fig.update_layout(
                        title=f"Line Plot: {', '.join(y_displays)} vs {x_display}",
                        xaxis_title=x_display,
                        yaxis_title="Value",
                        showlegend=True
                    )

                    st.plotly_chart(fig)



else:
    st.info("Upload a CSV file to get started.")
