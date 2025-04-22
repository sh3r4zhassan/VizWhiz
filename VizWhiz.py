import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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

    if st.sidebar.checkbox("Remove outliers (IQR method)", value=True):
        float_cols = df.select_dtypes(include=['float64', 'int']).columns
        Q1 = df[float_cols].quantile(0.25)
        Q3 = df[float_cols].quantile(0.75)
        IQR = Q3 - Q1
        mask = ~((df[float_cols] < (Q1 - 1.5 * IQR)) | (df[float_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
        df = df[mask]

    if st.sidebar.checkbox("Remove duplicate rows", value=True):
        df = df.drop_duplicates()

    st.subheader("Cleaned Data")
    st.dataframe(df)

    st.sidebar.header("Step 3: Choose Plot Types")
    plot_types = st.sidebar.multiselect(
    "Which plots do you want?",
    ["Histogram", "Box Plot", "Violin Plot", "Scatter Plot", "Line Plot"]
    )



    st.sidebar.header("Figure Settings")
    fig_width = st.sidebar.slider("Figure Width", 4, 12, 6)
    fig_height = st.sidebar.slider("Figure Height", 3, 8, 4)

    st.subheader("Dashboard Visualizations")

    numeric_cols = df.select_dtypes(include=['float64', 'int']).columns.tolist()
    display_names = [format_var_name(col) for col in numeric_cols]
    col_display_map = dict(zip(display_names, numeric_cols))

    if "Histogram" in plot_types:
        st.markdown("### Histograms")
        hist_count = st.number_input("How many histograms to show?", min_value=1, max_value=10, step=1, key="hist_count")
        for i in range(hist_count):
            with st.expander(f"Histogram #{i+1}"):
                col1, col2 = st.columns([1, 2])
                with col1:
                    display_col = st.selectbox("Select column", display_names, key=f"hist_col_{i}")
                    col = col_display_map[display_col]
                    color = st.color_picker("Color", value="#1f77b4", key=f"hist_color_{i}")
                    show_kde = st.checkbox("Show KDE (curve)", value=True, key=f"kde_{i}")
                    log_scale = st.checkbox("Log scale", value=False, key=f"log_hist_{i}")
                with col2:
                    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                    values = np.log(df[col]) if log_scale else df[col]
                    sns.histplot(values, kde=show_kde, ax=ax, color=color)
                    ax.set_title(f"Histogram of {display_col}", loc="left", weight="bold")
                    sns.despine(ax=ax)
                    ax.legend([],[], frameon=False)
                    fig.tight_layout()
                    st.pyplot(fig)

    if "Box Plot" in plot_types:
        st.markdown("### Box Plots")
        box_count = st.number_input("How many box plots to show?", min_value=1, max_value=10, step=1, key="box_count")

        for i in range(box_count):
            with st.expander(f"Box Plot #{i+1}"):
                col1, col2 = st.columns([1, 2])
                with col1:
                    selected_displays = st.multiselect("Select column(s)", display_names, default=[display_names[0]], key=f"box_col_{i}")
                    selected_cols = [col_display_map[name] for name in selected_displays]
                    show_mean = st.checkbox("Show Mean", value=False, key=f"box_mean_{i}")
                    show_fliers = st.checkbox("Show Outliers", value=True, key=f"box_fliers_{i}")
                    whis = st.selectbox("Whisker Extent", ["1.5 IQR", "Min-Max"], key=f"box_whis_{i}")
                    whis_val = 1.5 if whis == "1.5 IQR" else [0, 100]

                    color_palette = {}
                    for name in selected_displays:
                        color_palette[col_display_map[name]] = st.color_picker(f"Color for {name}", value="#1f77b4", key=f"box_color_{i}_{name}")

                with col2:
                    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                    
                    if len(selected_cols) == 1:
                        # Single box plot
                        col = selected_cols[0]
                        sns.boxplot(
                            x=df[col],
                            ax=ax,
                            color=color_palette.get(col, "#1f77b4"),
                            showmeans=show_mean,
                            showfliers=show_fliers,
                            whis=whis_val
                        )
                        ax.set_title(f"Box Plot of {selected_displays[0]}", loc="left", weight="bold")
                    else:
                        # Multi-variable box plot
                        melted = df[selected_cols].melt(var_name="Variable", value_name="Value")
                        palette = {col: color_palette[col] for col in selected_cols}
                        sns.boxplot(
                            x="Variable", y="Value", data=melted,
                            ax=ax,
                            palette=palette,
                            showmeans=show_mean,
                            showfliers=show_fliers,
                            whis=whis_val
                        )
                        ax.set_title("Combined Box Plot", loc="left", weight="bold")

                    sns.despine(ax=ax)
                    ax.legend([], [], frameon=False)
                    fig.tight_layout()
                    st.pyplot(fig)

    if "Scatter Plot" in plot_types:
        st.markdown("### Scatter Plots")
        scatter_count = st.number_input("How many scatter plots to show?", min_value=1, max_value=10, step=1, key="scatter_count")
        for i in range(scatter_count):
            with st.expander(f"Scatter Plot #{i+1}"):
                col1, col2 = st.columns([1, 2])
                with col1:
                    x_display = st.selectbox("X-axis", display_names, key=f"x_scatter_{i}")
                    x_col = col_display_map[x_display]
                    y_displays = st.multiselect("Y-axis columns", display_names, default=[display_names[0]], key=f"y_scatter_{i}")
                    y_cols = [col_display_map[y] for y in y_displays]
                    show_reg = st.checkbox("Add Regression Line", value=False, key=f"reg_scatter_{i}")
                    log_scale = st.checkbox("Log scale (x & y)?", value=False, key=f"log_scatter_{i}")
                    alpha = st.slider("Transparency", 0.1, 1.0, 0.7, key=f"alpha_scatter_{i}")
                    size = st.slider("Point size", 10, 200, 50, key=f"size_scatter_{i}")
                    
                    color_map = {}
                    for y_display in y_displays:
                        color = st.color_picker(f"Color for {y_display}", value="#1f77b4", key=f"scatter_color_{i}_{y_display}")
                        color_map[y_display] = color

                with col2:
                    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                    for y_display in y_displays:
                        y = col_display_map[y_display]
                        x_vals = np.log(df[x_col]) if log_scale else df[x_col]
                        y_vals = np.log(df[y]) if log_scale else df[y]
                        color = color_map[y_display]

                        sns.scatterplot(x=x_vals, y=y_vals, ax=ax, label=y_display, s=size, alpha=alpha, color=color)

                        if show_reg:
                            reg_line = sns.regplot(x=x_vals, y=y_vals, data=df, scatter=False, ax=ax, label=f"{y_display} (reg)", color=color)
                            reg_line.get_lines()[-1].set_linestyle("--")

                    ax.set_title(f"Scatter Plot: {', '.join(y_displays)} vs {x_display}", loc="left", weight="bold")
                    sns.despine(ax=ax)
                    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
                    fig.tight_layout()
                    st.pyplot(fig)

                    
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
                    show_points = st.checkbox("Show points as circles", value=True, key=f"show_points_line_{i}")
                    log_scale = st.checkbox("Log scale (x & y)?", value=False, key=f"log_line_{i}")
                    alpha = st.slider("Transparency", 0.1, 1.0, 0.7, key=f"alpha_line_{i}")
                    size = st.slider("Point size (if shown)", 10, 200, 50, key=f"size_line_{i}")

                    color_map = {}
                    for y_display in y_displays:
                        color = st.color_picker(f"Color for {y_display}", value="#1f77b4", key=f"line_color_{i}_{y_display}")
                        color_map[y_display] = color

                with col2:
                    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                    for y_display in y_displays:
                        y = col_display_map[y_display]
                        x_vals = np.log(df[x_col]) if log_scale else df[x_col]
                        y_vals = np.log(df[y]) if log_scale else df[y]
                        color = color_map[y_display]

                        sns.lineplot(x=x_vals, y=y_vals, ax=ax, label=y_display, alpha=alpha, color=color)
                        if show_points:
                            ax.scatter(x_vals, y_vals, s=size, alpha=alpha, color=color)

                    ax.set_title(f"Line Plot: {', '.join(y_displays)} vs {x_display}", loc="left", weight="bold")
                    sns.despine(ax=ax)
                    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
                    fig.tight_layout()
                    st.pyplot(fig)


    if "Violin Plot" in plot_types:
        st.markdown("### Violin Plots")
        violin_count = st.number_input("How many violin plots to show?", min_value=1, max_value=10, step=1, key="violin_count")

        for i in range(violin_count):
            with st.expander(f"Violin Plot #{i+1}"):
                col1, col2 = st.columns([1, 2])
                with col1:
                    selected_displays = st.multiselect("Select column(s)", display_names, default=[display_names[0]], key=f"violin_col_{i}")
                    selected_cols = [col_display_map[name] for name in selected_displays]
                    show_box = st.checkbox("Show Box", value=True, key=f"violin_box_{i}")
                    show_kde = st.checkbox("Show KDE", value=True, key=f"violin_kde_{i}")
                    scale = st.selectbox("Violin Width Scale", ["Area", "Count", "Width"], key=f"violin_scale_{i}")

                    color_palette = {}
                    for name in selected_displays:
                        color_palette[col_display_map[name]] = st.color_picker(f"Color for {name}", value="#1f77b4", key=f"violin_color_{i}_{name}")

                with col2:
                    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

                    if len(selected_cols) == 1:
                        col = selected_cols[0]
                        sns.violinplot(
                            x=df[col],
                            ax=ax,
                            inner="box" if show_box else None,
                            cut=0,
                            scale=scale.lower(),
                            bw=0.2,
                            color=color_palette.get(col, "#1f77b4")
                        )
                        ax.set_title(f"Violin Plot of {selected_displays[0]}", loc="left", weight="bold")
                    else:
                        melted = df[selected_cols].melt(var_name="Variable", value_name="Value")
                        palette = {col: color_palette[col] for col in selected_cols}
                        sns.violinplot(
                            x="Variable",
                            y="Value",
                            data=melted,
                            ax=ax,
                            inner="box" if show_box else None,
                            cut=0,
                            scale=scale.lower(),
                            bw=0.2,
                            palette=palette
                        )
                        ax.set_title("Combined Violin Plot", loc="left", weight="bold")

                    sns.despine(ax=ax)
                    ax.legend([], [], frameon=False)
                    fig.tight_layout()
                    st.pyplot(fig)


else:
    st.info("Upload a CSV file to get started.")
