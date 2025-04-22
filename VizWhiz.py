import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="VizWhiz", layout="wide")

def format_var_name(name):
    return ' '.join(word.capitalize() for word in name.split('_'))

st.title("📊 VizWhiz - A Visualization Wizard")

st.sidebar.header("Step 1: Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Raw Data")
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

    st.subheader("✅ Cleaned Data")
    st.dataframe(df)

    # Select Numeric Columns
    st.sidebar.header("Step 3: Select Numeric Columns")
    numeric_cols = df.select_dtypes(include=['float64', 'int']).columns.tolist()
    display_names = [format_var_name(col) for col in numeric_cols]
    col_display_map = dict(zip(display_names, numeric_cols))
    selected_display = st.sidebar.multiselect("Choose columns for visualization:", display_names)
    selected_columns = [col_display_map[name] for name in selected_display]

    if selected_columns:
        st.sidebar.header("Step 4: Choose Plot Types")
        plot_types = st.sidebar.multiselect("Which plots do you want?", ["Histogram", "Box Plot", "Scatter Plot"])

        st.sidebar.header("Figure Settings")
        fig_width = st.sidebar.slider("Figure Width", 4, 12, 6)
        fig_height = st.sidebar.slider("Figure Height", 3, 8, 4)

        st.subheader("📊 Dashboard Visualizations")

        # Histogram
        if "Histogram" in plot_types:
            st.markdown("### 📌 Histograms")
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


                # Box Plot
        if "Box Plot" in plot_types:
            st.markdown("### 📌 Box Plots")
            box_mode = st.radio("Box Plot Mode", ["One column per figure", "Multiple columns in one figure"], horizontal=True)

            # One-variable Box Plot
            if box_mode == "One column per figure":
                box_count = st.number_input("How many box plots to show?", min_value=1, max_value=10, step=1, key="box_count")
                for i in range(box_count):
                    with st.expander(f"Box Plot #{i+1}"):
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            display_col = st.selectbox("Column", display_names, key=f"box_col_{i}")
                            col = col_display_map[display_col]
                            color = st.color_picker("Box Color", "#1f77b4", key=f"box_color_{i}")
                            show_mean = st.checkbox("Show Mean", value=False, key=f"box_mean_{i}")
                            show_fliers = st.checkbox("Show Outliers", value=True, key=f"box_fliers_{i}")
                            whis = st.selectbox("Whisker Extent", ["1.5 IQR", "Min-Max"], key=f"box_whis_{i}")
                            whis_val = 1.5 if whis == "1.5 IQR" else [0, 100]
                        with col2:
                            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                            sns.boxplot(x=df[col], ax=ax, color=color, showmeans=show_mean, showfliers=show_fliers, whis=whis_val)
                            ax.set_title(f"Box Plot of {display_col}", loc="left", weight="bold")
                            sns.despine(ax=ax)
                            ax.legend([], [], frameon=False)
                            fig.tight_layout()
                            st.pyplot(fig)

            # Multi-variable Box Plot
            else:
                with st.expander("Multi-variable Box Plot"):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        multi_displays = st.multiselect("Columns", display_names, default=display_names[:2], key="multi_cols")
                        multi_cols = [col_display_map[name] for name in multi_displays]
                        show_mean = st.checkbox("Show Mean", value=False, key="multi_box_mean")
                        show_fliers = st.checkbox("Show Outliers", value=True, key="multi_box_fliers")
                        whis = st.selectbox("Whisker Extent", ["1.5 IQR", "Min-Max"], key="multi_whis")
                        whis_val = 1.5 if whis == "1.5 IQR" else [0, 100]

                        color_palette = {}
                        for name in multi_displays:
                            color_palette[col_display_map[name]] = st.color_picker(f"Color for {name}", value="#1f77b4", key=f"multi_color_{name}")

                    with col2:
                        if multi_cols:
                            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                            melted = df[multi_cols].melt(var_name="Variable", value_name="Value")
                            palette = color_palette if any(color_palette.values()) else sns.color_palette("Pastel2", len(multi_cols))
                            sns.boxplot(x="Variable", y="Value", data=melted, ax=ax,
                                        palette=palette, showmeans=show_mean, showfliers=show_fliers, whis=whis_val)
                            ax.set_title("Combined Box Plot", loc="left", weight="bold")
                            sns.despine(ax=ax)
                            ax.legend([], [], frameon=False)
                            fig.tight_layout()
                            st.pyplot(fig)


        # Scatter Plot
        if "Scatter Plot" in plot_types:
            st.markdown("### 📌 Scatter Plots")
            scatter_count = st.number_input("How many scatter plots to show?", min_value=1, max_value=10, step=1, key="scatter_count")
            for i in range(scatter_count):
                with st.expander(f"Scatter Plot #{i+1}"):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        x_display = st.selectbox("X-axis", display_names, key=f"x_{i}")
                        x_col = col_display_map[x_display]
                        y_displays = st.multiselect("Y-axis columns", display_names, default=[display_names[0]], key=f"y_{i}")
                        y_cols = [col_display_map[y] for y in y_displays]
                        plot_type = st.selectbox("Plot Type", ["Scatter", "Line"], key=f"scatter_type_{i}")
                        show_reg = st.checkbox("Add Regression Line", value=False, key=f"reg_{i}")
                        log_scale = st.checkbox("Log scale (x & y)?", value=False, key=f"log_{i}")
                        alpha = st.slider("Transparency", 0.1, 1.0, 0.7, key=f"alpha_{i}")
                        size = st.slider("Point size", 10, 200, 50, key=f"size_{i}")
                    with col2:
                        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                        for y_display in y_displays:
                            y = col_display_map[y_display]
                            x_vals = np.log(df[x_col]) if log_scale else df[x_col]
                            y_vals = np.log(df[y]) if log_scale else df[y]
                            color = st.color_picker(f"Color for {y_display}", value="#1f77b4", key=f"scatter_color_{i}_{y}")
                            if plot_type == "Scatter":
                                sns.scatterplot(x=x_vals, y=y_vals, ax=ax, label=y_display, s=size, alpha=alpha, color=color)
                            else:
                                sns.lineplot(x=x_vals, y=y_vals, ax=ax, label=y_display, alpha=alpha, color=color)
                            if show_reg:
                                reg_line = sns.regplot(x=x_vals, y=y_vals, data=df, scatter=False, ax=ax, label=f"{y_display} (reg)", color=color)
                                reg_line.get_lines()[-1].set_linestyle("--")
                        ax.set_title(f"{plot_type} Plot: {', '.join(y_displays)} vs {x_display}", loc="left", weight="bold")
                        sns.despine(ax=ax)
                        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
                        fig.tight_layout()
                        st.pyplot(fig)
else:
    st.info("👈 Upload a CSV file to get started.")
