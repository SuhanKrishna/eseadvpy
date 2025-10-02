# app.py
# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os

# =================================================================================
# 1. GUI Design & Initial Setup
# =================================================================================

# Set the page configuration for a wider layout and a title
st.set_page_config(layout="wide", page_title="Interactive Data Analysis Dashboard")

# Add a title to the dashboard
st.title("📊 Interactive Data Analysis and Visualization Dashboard")
st.write("""
This application is designed to meet the assignment requirements by providing a comprehensive
GUI for data manipulation, analysis, and visualization using Streamlit.
""")

# --- Sidebar for Controls and Widgets ---
st.sidebar.header("⚙️ Controls & Options")

# Function to load data. Caching is used to improve performance.
@st.cache_data
def load_data():
    """
    Loads the Palmer Penguins dataset from a URL.
    This function is cached to prevent reloading data on every user interaction.
    """
    url = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv"
    df = pd.read_csv(url)
    # Drop rows with missing values for cleaner analysis
    df.dropna(inplace=True)
    return df

# Load the initial dataframe
df_original = load_data()

# Create a session state to hold the potentially modified dataframe
if 'df' not in st.session_state:
    st.session_state.df = df_original.copy()

# =================================================================================
# 2. File Read/Write/Append Operations
# =================================================================================

st.header("1. File I/O Operations")
st.markdown("""
Perform read, write, and append operations on Text (CSV), Binary (Pickle), and XML files. 
You can save the current state of the data or upload your own file.
""")

# --- File Writing ---
col1, col2, col3 = st.columns(3)

# i. Save to Text File (CSV)
with col1:
    if st.button("Save as CSV (Text)"):
        # The dataframe is converted to a CSV string
        csv = st.session_state.df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name='penguins_data.csv',
            mime='text/csv',
        )
        st.success("CSV file is ready for download.")

# ii. Save to Binary File (Pickle)
with col2:
    if st.button("Save as Pickle (Binary)"):
        # The dataframe is serialized into a binary format using pickle
        pkl = st.session_state.df.to_pickle()
        st.download_button(
            label="📥 Download Pickle",
            data=pkl,
            file_name="penguins_data.pkl"
        )
        st.success("Pickle file is ready for download.")

# iii. Save to XML File
with col3:
    if st.button("Save as XML"):
        # Function to convert DataFrame to XML
        def df_to_xml(df_to_convert):
            root = ET.Element("data")
            for _, row in df_to_convert.iterrows():
                record = ET.SubElement(root, "record")
                for col_name in df_to_convert.columns:
                    item = ET.SubElement(record, str(col_name).replace(" ", "_"))
                    item.text = str(row[col_name])
            # Pretty print the XML
            xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
            return xml_str

        xml_data = df_to_xml(st.session_state.df).encode('utf-8')
        st.download_button(
            label="📥 Download XML",
            data=xml_data,
            file_name="penguins_data.xml",
            mime='application/xml',
        )
        st.success("XML file is ready for download.")
        
# --- File Reading ---
st.subheader("Upload and Read a File")
uploaded_file = st.file_uploader("Choose a file (CSV or Pickle)", type=['csv', 'pkl'])
if uploaded_file is not None:
    try:
        # Get the file extension
        file_extension = os.path.splitext(uploaded_file.name)[1]
        
        # Read file based on extension
        if file_extension == ".csv":
            # read from text file
            new_df = pd.read_csv(uploaded_file)
        elif file_extension == ".pkl":
            # read from binary file
            new_df = pd.read_pickle(uploaded_file)
            
        st.session_state.df = new_df
        st.success(f"Successfully loaded {uploaded_file.name}")
    except Exception as e:
        st.error(f"Error reading file: {e}")

# =================================================================================
# 3. Data Analysis, Visualization & Interactive Dashboard
# =================================================================================

st.header("2. Data Analysis and Visualization")
st.markdown("Explore, manipulate, and visualize the dataset interactively.")

# --- Display Data Details ---
if st.checkbox("Show Data Details", value=True):
    st.subheader("Dataset Details")
    # Display the dimensions (shape) of the dataframe
    st.write("**Dataset Shape:**", st.session_state.df.shape)
    # Display the column names
    st.write("**Columns:**", list(st.session_state.df.columns))
    # Display the raw dataframe in an interactive table
    st.dataframe(st.session_state.df)

# --- Summary Statistics ---
if st.checkbox("Show Summary Statistics"):
    st.subheader("Summary Statistics")
    # The describe() function provides statistical details on numerical columns
    st.write(st.session_state.df.describe())

# --- Interactive Data Manipulation ---
st.sidebar.subheader("Data Manipulation")

# i. Convert data types
st.sidebar.markdown("**Convert Data Types**")
col_to_convert = st.sidebar.selectbox("Select column to convert", options=st.session_state.df.columns)
convert_to_type = st.sidebar.selectbox("Convert to type", options=['object', 'int64', 'float64'])
if st.sidebar.button("Convert Type"):
    try:
        # astype() function is used for type casting
        st.session_state.df[col_to_convert] = st.session_state.df[col_to_convert].astype(convert_to_type)
        st.experimental_rerun()
    except Exception as e:
        st.sidebar.error(f"Conversion failed: {e}")

# ii. Sorting
st.sidebar.markdown("**Sort Data**")
sort_column = st.sidebar.selectbox("Sort by column", options=st.session_state.df.columns)
sort_order = st.sidebar.radio("Sort order", options=["Ascending", "Descending"])
if st.sidebar.button("Sort"):
    # sort_values() function is used to sort the dataframe
    is_ascending = (sort_order == "Ascending")
    st.session_state.df = st.session_state.df.sort_values(by=sort_column, ascending=is_ascending)
    st.experimental_rerun()

# iii. Grouping
st.sidebar.markdown("**Group Data**")
group_column = st.sidebar.selectbox("Group by column", options=st.session_state.df.select_dtypes(include='object').columns)
agg_column = st.sidebar.selectbox("Aggregate column", options=st.session_state.df.select_dtypes(include=np.number).columns)
agg_func = st.sidebar.selectbox("Aggregation function", options=['mean', 'sum', 'count', 'max', 'min'])
if st.sidebar.button("Group and Aggregate"):
    st.subheader("Grouped Data Result")
    # groupby() is used to split data into groups based on some criteria
    grouped_df = st.session_state.df.groupby(group_column)[agg_column].agg(agg_func)
    st.dataframe(grouped_df)

# iv. Filtering / Slicing / Locating
st.sidebar.markdown("**Filter Data (Slicing)**")
filter_col = st.sidebar.selectbox("Select column to filter", st.session_state.df.columns)

# Depending on the column type, provide different filter widgets
if pd.api.types.is_numeric_dtype(st.session_state.df[filter_col]):
    # Use a slider for numerical columns
    min_val, max_val = float(st.session_state.df[filter_col].min()), float(st.session_state.df[filter_col].max())
    slider_range = st.sidebar.slider(f"Filter range for {filter_col}", min_val, max_val, (min_val, max_val))
    # Slicing the dataframe based on the condition
    st.session_state.df = df_original[st.session_state.df[filter_col].between(slider_range[0], slider_range[1])]
else:
    # Use a multiselect box for categorical columns
    unique_values = st.session_state.df[filter_col].unique()
    selected_values = st.sidebar.multiselect(f"Filter values for {filter_col}", options=unique_values, default=unique_values)
    # isin() is used for filtering based on a list of values
    st.session_state.df = df_original[st.session_state.df[filter_col].isin(selected_values)]

if st.sidebar.button("Reset Filters"):
    st.session_state.df = df_original.copy()
    st.experimental_rerun()

# --- Data Visualization ---
st.header("3. Data Visualization")

plot_type = st.selectbox("Select Plot Type", ["Histogram", "Bar Chart", "Scatter Plot"])

if plot_type == "Histogram":
    st.subheader("Histogram")
    st.markdown("Shows the distribution of a single numerical variable.")
    # Select a numerical column for the histogram
    hist_col = st.selectbox("Select a numerical column", options=st.session_state.df.select_dtypes(include=np.number).columns)
    # Plotting using Plotly Express
    fig = px.histogram(st.session_state.df, x=hist_col, title=f"Distribution of {hist_col}")
    st.plotly_chart(fig, use_container_width=True)

elif plot_type == "Bar Chart":
    st.subheader("Bar Chart")
    st.markdown("Compares a numerical value across different categories.")
    # Select categorical and numerical columns for the bar chart
    cat_col = st.selectbox("Select a categorical column (X-axis)", options=st.session_state.df.select_dtypes(include='object').columns)
    num_col_bar = st.selectbox("Select a numerical column (Y-axis)", options=st.session_state.df.select_dtypes(include=np.number).columns, key="bar_num")
    # Plotting using Plotly Express
    fig = px.bar(st.session_state.df, x=cat_col, y=num_col_bar, title=f"{num_col_bar} by {cat_col}")
    st.plotly_chart(fig, use_container_width=True)

elif plot_type == "Scatter Plot":
    st.subheader("Scatter Plot")
    st.markdown("Shows the relationship between two numerical variables.")
    # Select two numerical columns for the scatter plot
    x_axis = st.selectbox("Select X-axis", options=st.session_state.df.select_dtypes(include=np.number).columns)
    y_axis = st.selectbox("Select Y-axis", options=st.session_state.df.select_dtypes(include=np.number).columns, index=1)
    color_axis = st.selectbox("Select column for color", options=st.session_state.df.columns)
    # Plotting using Plotly Express
    fig = px.scatter(st.session_state.df, x=x_axis, y=y_axis, color=color_axis, title=f"Relationship between {x_axis} and {y_axis}")
    st.plotly_chart(fig, use_container_width=True)
