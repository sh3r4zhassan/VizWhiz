import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex  # Place this import at the top of your file



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
                col1, col2 = st.columns([1, 2])
                with col1:
                    display_col = st.selectbox("Select column", display_names, key=f"hist_col_{i}")
                    col = col_display_map[display_col]
                    show_kde = st.checkbox("Show KDE (curve)", value=True, key=f"kde_{i}")
                    log_scale = st.checkbox("Log scale", value=False, key=f"log_hist_{i}")
                    groupby_col = groupby_selector(f"hist_{i}")
                    color_palette = {}
                    if groupby_col != "None":
                        unique_groups = df[groupby_col].dropna().unique()
                        default_colors = sns.color_palette("pastel", len(unique_groups))
                        for idx, grp in enumerate(unique_groups):
                            hex_color = to_hex(default_colors[idx])
                            color_palette[grp] = st.color_picker(f"Color for {grp}", value=hex_color, key=f"hist_color_{i}_{grp}")
                    else:
                        color = st.color_picker("Color for plot", value="#a1c9f4", key=f"hist_color_{i}_single")
                with col2:
                    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                    if groupby_col != "None":
                        for grp in df[groupby_col].dropna().unique():
                            subset = df[df[groupby_col] == grp]
                            values = np.log(subset[col]) if log_scale else subset[col]
                            sns.histplot(values, kde=show_kde, ax=ax, color=color_palette[grp], label=str(grp))
                        ax.legend(title=groupby_col, loc='upper left', bbox_to_anchor=(1.02, 1))
                    else:
                        values = np.log(df[col]) if log_scale else df[col]
                        sns.histplot(values, kde=show_kde, ax=ax, color=color, label=display_col)
                        ax.legend(title="Legend", loc='upper left', bbox_to_anchor=(1.02, 1))
                    ax.set_title(f"Histogram of {display_col}", loc="left", weight="bold")
                    sns.despine(ax=ax)
                    fig.tight_layout()
                    st.pyplot(fig)


    # --- Box Plot ---
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
                    groupby_col = groupby_selector(f"box_{i}")

                    color_palette = {}
                    if groupby_col != "None":
                        unique_groups = df[groupby_col].dropna().unique()
                        default_colors = sns.color_palette("pastel", len(unique_groups))
                        for idx, grp in enumerate(unique_groups):
                            hex_color = to_hex(default_colors[idx])
                            color_palette[grp] = st.color_picker(f"Color for {grp}", value=hex_color, key=f"box_color_{i}_{grp}")
                    else:
                        variable_colors = {}
                        for idx, name in enumerate(selected_displays):
                            default_color = to_hex(sns.color_palette("pastel")[idx % 10])
                            variable_colors[name] = st.color_picker(f"Color for {name}", value=default_color, key=f"box_color_{i}_{name}")

                with col2:
                    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                    if len(selected_cols) == 1:
                        col = selected_cols[0]
                        if groupby_col != "None":
                            sns.boxplot(data=df, x=groupby_col, y=col, ax=ax,
                                        palette=color_palette, showmeans=show_mean,
                                        showfliers=show_fliers, whis=whis_val)
                            ax.legend(title=groupby_col, bbox_to_anchor=(1.02, 1), loc="upper left")
                        else:
                            sns.boxplot(x=df[col], ax=ax, color=variable_colors[selected_displays[0]],
                                        showmeans=show_mean, showfliers=show_fliers, whis=whis_val)
                            ax.legend([], [], frameon=False)
                        ax.set_title(f"Box Plot of {selected_displays[0]}", loc="left", weight="bold")
                    else:
                        melted = df[selected_cols].melt(var_name="Variable", value_name="Value")
                        if groupby_col != "None":
                            melted[groupby_col] = pd.concat([df[groupby_col]] * len(selected_cols), ignore_index=True)
                            sns.boxplot(data=melted, x="Variable", y="Value", hue=groupby_col, ax=ax,
                                        palette=color_palette, showmeans=show_mean, showfliers=show_fliers, whis=whis_val)
                            ax.legend(title=groupby_col, bbox_to_anchor=(1.02, 1), loc="upper left")
                        else:
                            # Correct palette mapping: real column name → color
                            col_to_display = {col_display_map[name]: name for name in selected_displays}
                            palette = {col: variable_colors[col_to_display[col]] for col in selected_cols}
                            sns.boxplot(x="Variable", y="Value", data=melted, ax=ax,
                                        palette=palette, showmeans=show_mean, showfliers=show_fliers, whis=whis_val)
                            ax.legend([], [], frameon=False)

                        ax.set_title("Combined Box Plot", loc="left", weight="bold")
                    sns.despine(ax=ax)
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
                    groupby_col = groupby_selector(f"scatter_{i}")

                    if groupby_col != "None":
                        color_palette = {}
                        unique_groups = df[groupby_col].dropna().unique()
                        default_colors = sns.color_palette("pastel", len(unique_groups))
                        for idx, grp in enumerate(unique_groups):
                            hex_color = to_hex(default_colors[idx])
                            color_palette[grp] = st.color_picker(f"Color for {grp}", value=hex_color, key=f"scatter_color_{i}_{grp}")
                    else:
                        variable_colors = {}
                        for idx, y_display in enumerate(y_displays):
                            default_color = to_hex(sns.color_palette("pastel")[idx % 10])
                            variable_colors[y_display] = st.color_picker(f"Color for {y_display}", value=default_color, key=f"scatter_color_{i}_{y_display}")

                with col2:
                    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                    for idx, y_display in enumerate(y_displays):
                        y = col_display_map[y_display]
                        x_vals = np.log(df[x_col]) if log_scale else df[x_col]
                        y_vals = np.log(df[y]) if log_scale else df[y]

                        if groupby_col != "None":
                            sns.scatterplot(data=df, x=x_col, y=y, hue=groupby_col, ax=ax,
                                            alpha=alpha, s=size, palette=color_palette)
                        else:
                            sns.scatterplot(x=x_vals, y=y_vals, ax=ax, label=y_display,
                                            s=size, alpha=alpha, color=variable_colors[y_display])
                            if show_reg:
                                sns.regplot(x=x_vals, y=y_vals, scatter=False, ax=ax,
                                            label=f"{y_display} (reg)", color=variable_colors[y_display], linestyle="--")

                    ax.set_title(f"Scatter Plot: {', '.join(y_displays)} vs {x_display}", loc="left", weight="bold")
                    sns.despine(ax=ax)
                    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
                    fig.tight_layout()
                    st.pyplot(fig)

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
                    show_points = st.checkbox("Show points as circles", value=True, key=f"show_points_line_{i}")
                    log_scale = st.checkbox("Log scale (x & y)?", value=False, key=f"log_line_{i}")
                    alpha = st.slider("Transparency", 0.1, 1.0, 0.7, key=f"alpha_line_{i}")
                    size = st.slider("Point size (if shown)", 10, 200, 50, key=f"size_line_{i}")
                    groupby_col = groupby_selector(f"line_{i}")

                    if groupby_col != "None":
                        color_palette = {}
                        unique_groups = df[groupby_col].dropna().unique()
                        default_colors = sns.color_palette("pastel", len(unique_groups))
                        for idx, grp in enumerate(unique_groups):
                            hex_color = to_hex(default_colors[idx])
                            color_palette[grp] = st.color_picker(f"Color for {grp}", value=hex_color, key=f"line_color_{i}_{grp}")
                    else:
                        variable_colors = {}
                        for idx, y_display in enumerate(y_displays):
                            default_color = to_hex(sns.color_palette("pastel")[idx % 10])
                            variable_colors[y_display] = st.color_picker(f"Color for {y_display}", value=default_color, key=f"line_color_{i}_{y_display}")

                with col2:
                    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                    for idx, y_display in enumerate(y_displays):
                        y = col_display_map[y_display]
                        x_vals = np.log(df[x_col]) if log_scale else df[x_col]
                        y_vals = np.log(df[y]) if log_scale else df[y]

                        if groupby_col != "None":
                            sns.lineplot(data=df, x=x_col, y=y, hue=groupby_col, ax=ax,
                                        palette=color_palette, alpha=alpha)
                            if show_points:
                                for grp, subdf in df.groupby(groupby_col):
                                    ax.scatter(subdf[x_col], subdf[y], s=size, alpha=alpha, label=grp)
                        else:
                            sns.lineplot(x=x_vals, y=y_vals, ax=ax, label=y_display,
                                        alpha=alpha, color=variable_colors[y_display])
                            if show_points:
                                ax.scatter(x_vals, y_vals, s=size, alpha=alpha, color=variable_colors[y_display])

                    ax.set_title(f"Line Plot: {', '.join(y_displays)} vs {x_display}", loc="left", weight="bold")
                    sns.despine(ax=ax)
                    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
                    fig.tight_layout()
                    st.pyplot(fig)

    # --- Violin Plot ---
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
                    groupby_col = groupby_selector(f"violin_{i}")

                    color_palette = {}
                    if groupby_col != "None":
                        unique_groups = df[groupby_col].dropna().unique()
                        default_colors = sns.color_palette("pastel", len(unique_groups))
                        for idx, grp in enumerate(unique_groups):
                            hex_color = to_hex(default_colors[idx])
                            color_palette[grp] = st.color_picker(f"Color for {grp}", value=hex_color, key=f"violin_color_{i}_{grp}")
                    else:
                        color_picker_single = st.color_picker("Color for plot", value=to_hex(sns.color_palette("pastel")[0]), key=f"violin_color_{i}_single")

                with col2:
                    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

                    if len(selected_cols) == 1:
                        col = selected_cols[0]
                        if groupby_col != "None":
                            sns.violinplot(data=df, x=groupby_col, y=col, ax=ax,
                                           palette=color_palette, inner="box" if show_box else None, scale=scale.lower())
                            ax.legend(title=groupby_col, bbox_to_anchor=(1.02, 1), loc="upper left")
                        else:
                            sns.violinplot(x=df[col], ax=ax,
                                           inner="box" if show_box else None, scale=scale.lower(),
                                           color=color_picker_single)
                            ax.legend([], [], frameon=False)
                        ax.set_title(f"Violin Plot of {selected_displays[0]}", loc="left", weight="bold")
                    else:
                        melted = df[selected_cols].melt(var_name="Variable", value_name="Value")
                        if groupby_col != "None":
                            melted[groupby_col] = pd.concat([df[groupby_col]] * len(selected_cols), ignore_index=True)
                            sns.violinplot(data=melted, x="Variable", y="Value", hue=groupby_col, ax=ax,
                                           palette=color_palette, inner="box" if show_box else None, scale=scale.lower(), cut=0)
                            ax.legend(title=groupby_col, bbox_to_anchor=(1.02, 1), loc="upper left")
                        else:
                            palette = [color_picker_single] * len(selected_cols)
                            sns.violinplot(x="Variable", y="Value", data=melted, ax=ax,
                                           inner="box" if show_box else None, palette=palette, scale=scale.lower(), cut=0)
                            ax.legend([], [], frameon=False)
                        ax.set_title("Combined Violin Plot", loc="left", weight="bold")
                    sns.despine(ax=ax)
                    fig.tight_layout()
                    st.pyplot(fig)


else:
    st.info("Upload a CSV file to get started.")
