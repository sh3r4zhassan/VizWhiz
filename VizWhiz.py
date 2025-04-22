import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# App Title
st.title("📊 VizWhiz - A Visualization Wizard")

# Sidebar: Upload File
st.sidebar.header("Step 1: Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=["csv"])

if uploaded_file:
    # Read the uploaded CSV
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Raw Data")
    st.dataframe(df)

    # Sidebar: Data Cleaning Options
    st.sidebar.header("Step 2: Data Cleaning Settings")

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

    # Show cleaned data
    st.subheader("✅ Cleaned Data")
    st.dataframe(df)

    # Sidebar: Visualization Options
    st.sidebar.header("Step 3: Choose Dashboard Visuals")
    numeric_cols = df.select_dtypes(include=['float64', 'int']).columns.tolist()

    if len(numeric_cols) == 0:
        st.warning("No numeric columns found for visualization.")
    else:
        plot_types = st.sidebar.multiselect(
            "Select plots to include:", 
            ["Histogram", "Box Plot", "Scatter Plot"], 
            default=["Histogram", "Box Plot"]
        )

        st.sidebar.header("Select Columns to Visualize")
        selected_columns = st.sidebar.multiselect(
            "Pick numeric columns:", numeric_cols, default=numeric_cols[:2]
        )

        if "Scatter Plot" in plot_types:
            st.sidebar.markdown("**Scatter Plot Axes**")
            selected_x = st.sidebar.selectbox("X-axis", selected_columns)
            selected_y = st.sidebar.selectbox("Y-axis", selected_columns, index=1 if len(selected_columns) > 1 else 0)

        st.subheader("📊 Dashboard Visualizations")

        # Histogram
        if "Histogram" in plot_types and selected_columns:
            st.write("### Histogram")
            for col in selected_columns:
                fig, ax = plt.subplots()
                sns.histplot(df[col], kde=True, ax=ax)
                ax.set_title(f"Histogram of {col}")
                st.pyplot(fig)

        # Box Plot
        if "Box Plot" in plot_types and selected_columns:
            st.write("### Box Plot")
            for col in selected_columns:
                fig, ax = plt.subplots()
                sns.boxplot(x=df[col], ax=ax)
                ax.set_title(f"Box Plot of {col}")
                st.pyplot(fig)

        # Scatter Plot
        if "Scatter Plot" in plot_types and len(selected_columns) >= 2:
            st.write("### Scatter Plot")
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=selected_x, y=selected_y, ax=ax)
            ax.set_title(f"{selected_y} vs {selected_x}")
            st.pyplot(fig)
else:
    st.info("👈 Please upload a CSV file to get started.")
