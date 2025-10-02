# app.py
"""
Streamlit GUI for Data I/O, Analysis, Visualization and Interactive Dashboard
Marks mapping:
1. GUI Design (15)      - Dataset load/save, display, widgets
2. Data Analysis (15)   - dataset view, summary stats, conversions, sorting, grouping, indexing, slicing, filters
3. Interactive Dashboard(15) - buttons/widgets to run numpy/pandas queries
4. Writing (5)          - export to CSV/pickle/XML
5. Viva (5)             - small 'notes' & explanation panel

Author: ChatGPT (GPT-5 Thinking mini)
Date: 2025-10-02
"""

import streamlit as st
import pandas as pd           # pandas: DataFrame ops, read_csv, to_csv, groupby, describe, astype, sort_values, set_index, iloc, loc, to_xml
import numpy as np            # numpy: numerical ops, mean/median/std, array creation
import matplotlib.pyplot as plt  # matplotlib: plotting (histogram, bar)
import altair as alt          # altair: interactive plotting
import io
import pickle                 # pickle: binary read/write/append for python objects
import xml.etree.ElementTree as ET  # xml.etree: create and write XML
from datetime import datetime

st.set_page_config(page_title="Data GUI (Streamlit) - I/O, Analysis, Visuals", layout="wide")

# ---------------------------
# Helper functions (I/O)
# ---------------------------

def load_sample_dataset(name: str) -> pd.DataFrame:
    """
    Load a sample dataset using seaborn's datasets via pandas (if available) or create synthetic.
    Used functions:
    - pd.DataFrame (constructor)
    - pd.read_csv (if an online sample is used later)
    """
    if name == "iris":
        # Use sklearn datasets to create dataframe (no network)
        from sklearn import datasets
        iris = datasets.load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['target'] = iris.target
        return df
    elif name == "tips":
        try:
            import seaborn as sns
            return sns.load_dataset("tips")
        except Exception:
            # fallback synthetic
            rng = np.random.default_rng(0)
            df = pd.DataFrame({
                "total_bill": rng.normal(20, 5, 100),
                "tip": rng.normal(3, 1, 100),
                "sex": np.random.choice(['Male','Female'], 100),
                "smoker": np.random.choice(['Yes','No'], 100),
                "day": np.random.choice(['Thur','Fri','Sat','Sun'], 100)
            })
            return df
    else:
        # Default synthetic dataset
        rng = np.random.default_rng(1)
        df = pd.DataFrame({
            "A": rng.integers(0,100, 50),
            "B": rng.normal(0,1,50),
            "C": np.random.choice(list("XYZ"), 50)
        })
        return df

# Binary read/write (pickle)
def save_binary_pickle(df: pd.DataFrame, path: str):
    """Save df object to binary (pickle) - uses pickle.dump"""
    with open(path, "ab+") as f:       # 'ab+' allows append in binary mode
        pickle.dump({"ts":datetime.now().isoformat(), "data": df}, f)

def read_binary_pickle_all(path: str):
    """Read all pickle objects from file. Uses pickle.load repeatedly."""
    objs = []
    try:
        with open(path, "rb") as f:
            while True:
                try:
                    objs.append(pickle.load(f))
                except EOFError:
                    break
    except FileNotFoundError:
        return []
    return objs

# Text read/write (CSV)
def write_text_csv(df: pd.DataFrame, path: str, mode="w", index=False):
    """Write to CSV. Uses df.to_csv"""
    df.to_csv(path, mode=mode, index=index)

def read_text_csv(path: str):
    """Read CSV with pandas.read_csv"""
    return pd.read_csv(path)

# XML read/write
def df_to_xml_string(df: pd.DataFrame, root_name="data", row_name="row"):
    """
    Convert DataFrame to simple XML string using xml.etree.
    Demonstrates: ET.Element, ET.SubElement, ET.tostring
    """
    root = ET.Element(root_name)
    for _, row in df.iterrows():
        r = ET.SubElement(root, row_name)
        for col in df.columns:
            # convert to str for XML content
            c = ET.SubElement(r, str(col))
            c.text = str(row[col])
    return ET.tostring(root, encoding="unicode")

def save_xml_string_to_file(xml_str: str, path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml_str)

def read_xml_to_df(path: str):
    """
    Read simple XML back into DataFrame assuming structure created by df_to_xml_string.
    Uses ET.parse, root.findall, iterating children
    """
    tree = ET.parse(path)
    root = tree.getroot()
    rows = []
    for child in root:
        d = {}
        for el in child:
            d[el.tag] = el.text
        rows.append(d)
    return pd.DataFrame(rows)

# ---------------------------
# UI: Sidebar - Data sources
# ---------------------------

st.sidebar.title("Dataset & I/O")

data_source = st.sidebar.radio("Select dataset source", ("Upload CSV", "Sample dataset", "Download URL (CSV)"))

uploaded_file = None
df = None

if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        # pd.read_csv supports file-like, uses read_csv
        df = pd.read_csv(uploaded_file)
elif data_source == "Sample dataset":
    sample = st.sidebar.selectbox("Choose sample", ["iris", "tips", "synthetic"])
    if st.sidebar.button("Load sample dataset"):
        df = load_sample_dataset(sample)
elif data_source == "Download URL (CSV)":
    url = st.sidebar.text_input("Enter CSV URL")
    if st.sidebar.button("Download from URL"):
        if url:
            try:
                df = pd.read_csv(url)   # read_csv used
                st.sidebar.success("Downloaded successfully")
            except Exception as e:
                st.sidebar.error(f"Failed to download/read: {e}")

# Display basic dataset if loaded
st.title("Data GUI — I/O, Analysis, Visualization, Interactive Dashboard")
st.markdown("#### Instructions / Notes")
st.markdown("""
- You can **upload a CSV**, **load a sample**, or **download a CSV** from a URL.
- The app demonstrates **binary (pickle) read/write/append**, **text (CSV)** read/write, and **XML** conversion/read/write.
- All major `pandas` and `numpy` manipulations are available through buttons/widgets below.
""")

# ---------------------------
# Section 1: Display & Basic I/O Buttons
# ---------------------------

st.header("1. Dataset details & Basic I/O (GUI Design — 15 marks)")

col1, col2 = st.columns([2,1])

with col1:
    if df is None:
        st.info("No dataset loaded yet. Use the sidebar to load/upload a dataset.")
    else:
        st.subheader("Preview dataset (first 10 rows)")
        # st.dataframe uses streamlit's display; demonstrates interactive rendering
        st.dataframe(df.head(10))

        st.subheader("Basic details")
        # functions: df.shape, df.columns, df.info (we capture into buffer), df.dtypes
        st.write("Shape:", df.shape)         # tuple
        st.write("Columns:", list(df.columns))
        st.write("dtypes:")
        st.write(pd.DataFrame(df.dtypes, columns=["dtype"]))

        # df.info -> write to string buffer
        buf = io.StringIO()
        df.info(buf=buf)
        info_str = buf.getvalue()
        st.text(info_str)

with col2:
    st.subheader("I/O actions (binary/text/XML)")
    # Default file paths (server-side)
    csv_path = st.text_input("CSV path to save (server)", value="saved_dataset.csv")
    pickle_path = st.text_input("Pickle path (binary append)", value="saved_data.pkl")
    xml_path = st.text_input("XML path to save", value="saved_dataset.xml")

    if df is not None:
        if st.button("Save to CSV (overwrite)"):
            # df.to_csv
            write_text_csv(df, csv_path, mode="w", index=False)
            st.success(f"Saved CSV to {csv_path}")

        if st.button("Append to CSV"):
            # df.to_csv with mode='a'
            write_text_csv(df, csv_path, mode="a", index=False)
            st.success(f"Appended dataset to {csv_path}")

        if st.button("Write to binary (pickle append)"):
            # append a pickled record
            save_binary_pickle(df, pickle_path)
            st.success(f"Appended dataframe object (pickle) to {pickle_path}")

        if st.button("Read all binary (pickle) objects"):
            objs = read_binary_pickle_all(pickle_path)
            st.write(f"Found {len(objs)} objects in pickle file")
            if objs:
                st.write("Timestamps of saved objects:")
                st.write([o.get("ts") for o in objs])
                # show first saved object's df preview
                first_df = objs[0]["data"]
                st.dataframe(first_df.head())

        if st.button("Save as XML"):
            xml_str = df_to_xml_string(df, root_name="dataset", row_name="record")
            save_xml_string_to_file(xml_str, xml_path)
            st.success(f"Saved XML to {xml_path}")

        if st.button("Read XML back into DataFrame"):
            try:
                df_xml = read_xml_to_df(xml_path)
                st.write("Read XML -> converted to DataFrame (all entries are strings).")
                st.dataframe(df_xml.head())
            except Exception as e:
                st.error(f"Failed to read XML: {e}")

# ---------------------------
# Section 2: Data Analysis & Visualization (15 marks)
# ---------------------------

st.header("2. Data Analysis & Visualization (15 marks)")
if df is None:
    st.info("Load a dataset to perform analysis.")
else:
    st.subheader("Dataset & Summary statistics")
    if st.checkbox("Show full dataset"):
        st.dataframe(df)

    if st.checkbox("Show summary statistics (describe())"):
        # df.describe() - numeric summary
        st.dataframe(df.describe(include='all').T)

    # Convert column types
    st.subheader("Type conversions & transformations")
    cols = list(df.columns)
    conv_col = st.selectbox("Select column to convert type", ["-- none --"] + cols)
    conv_target = st.selectbox("Convert to type", ["int", "float", "str", "category", "datetime"])
    if st.button("Apply conversion"):
        try:
            if conv_target == "int":
                df[conv_col] = df[conv_col].astype("int")
            elif conv_target == "float":
                df[conv_col] = df[conv_col].astype("float")
            elif conv_target == "str":
                df[conv_col] = df[conv_col].astype("str")
            elif conv_target == "category":
                df[conv_col] = df[conv_col].astype("category")
            elif conv_target == "datetime":
                df[conv_col] = pd.to_datetime(df[conv_col], errors="coerce")  # pd.to_datetime
            st.success(f"Converted {conv_col} to {conv_target}. New dtype: {df[conv_col].dtype}")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Failed to convert: {e}")

    # Sorting, grouping
    st.subheader("Sorting & Grouping")
    sort_cols = st.multiselect("Select columns to sort by", cols)
    asc = st.radio("Ascending?", ("Yes","No")) == "Yes"
    if st.button("Apply sort"):
        try:
            if sort_cols:
                df_sorted = df.sort_values(by=sort_cols, ascending=asc)   # pd.DataFrame.sort_values
                st.dataframe(df_sorted.head(20))
            else:
                st.warning("Choose at least one column to sort by.")
        except Exception as e:
            st.error(f"Sort failed: {e}")

    group_col = st.selectbox("Select column to group by for aggregation", ["-- none --"] + cols)
    agg_func = st.selectbox("Aggregation function", ["mean", "sum", "median", "count", "std"])
    if st.button("Group & Aggregate"):
        try:
            if group_col != "-- none --":
                grouped = getattr(df.groupby(group_col), agg_func)()   # df.groupby(...).mean()/sum()/median()
                st.dataframe(grouped)
            else:
                st.warning("Select a column to group by.")
        except Exception as e:
            st.error(f"Group/Aggregation failed: {e}")

    # Indexing, slicing, locating filters
    st.subheader("Indexing, Slicing, Loc/Iloc & Filters")
    st.write("Use iloc (position-based) slice:")
    iloc_start = st.number_input("iloc start (row)", min_value=0, max_value=max(0,len(df)-1), value=0, step=1)
    iloc_end = st.number_input("iloc end (row, exclusive)", min_value=1, max_value=max(1,len(df)), value=min(5, len(df)), step=1)
    if st.button("Show iloc slice"):
        st.dataframe(df.iloc[iloc_start:iloc_end])   # df.iloc used

    st.write("Use loc (label-based) filtering:")
    # offer a simple filter builder for one column
    filter_col = st.selectbox("Column for simple filter (loc/query)", ["-- none --"] + cols)
    operator = st.selectbox("Operator", ["==", "!=", ">", ">=", "<", "<=", "contains"])
    filter_value = st.text_input("Value to compare (text input)")
    if st.button("Apply filter"):
        try:
            if filter_col == "-- none --":
                st.warning("Select a column")
            else:
                if operator == "contains":
                    mask = df[filter_col].astype(str).str.contains(filter_value, case=False, na=False)
                    result = df.loc[mask]   # df.loc used
                else:
                    # build a query safely using pandas Series ops
                    if filter_value == "":
                        st.warning("Enter a value")
                    else:
                        # attempt numeric conversion, otherwise compare as string
                        try:
                            # try float then int
                            if "." in filter_value:
                                val = float(filter_value)
                            else:
                                val = int(filter_value)
                        except Exception:
                            val = filter_value
                        if operator == "==":
                            result = df.loc[df[filter_col] == val]
                        elif operator == "!=":
                            result = df.loc[df[filter_col] != val]
                        elif operator == ">":
                            result = df.loc[df[filter_col] > val]
                        elif operator == ">=":
                            result = df.loc[df[filter_col] >= val]
                        elif operator == "<":
                            result = df.loc[df[filter_col] < val]
                        elif operator == "<=":
                            result = df.loc[df[filter_col] <= val]
                st.write(f"Filtered rows: {len(result)}")
                st.dataframe(result.head(50))
        except Exception as e:
            st.error(f"Filter failed: {e}")

    # Plots
    st.subheader("Plots (Bar, Line, Histogram)")
    plot_type = st.selectbox("Choose plot type", ["Histogram", "Bar chart", "Scatter (relationship)"])
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    if plot_type == "Histogram":
        hist_col = st.selectbox("Numeric column for histogram", ["-- none --"] + numeric_cols)
        bins = st.slider("Bins", 5, 100, 20)
        if st.button("Plot histogram"):
            if hist_col != "-- none --":
                fig, ax = plt.subplots()
                ax.hist(df[hist_col].dropna(), bins=bins)   # matplotlib.hist
                ax.set_title(f"Histogram of {hist_col}")
                st.pyplot(fig)
            else:
                st.warning("Select a numeric column")

    elif plot_type == "Bar chart":
        cat_cols = list(df.select_dtypes(include=['object','category']).columns)
        bar_cat = st.selectbox("Categorical column for bar (count)", ["-- none --"] + cat_cols)
        if st.button("Plot bar chart"):
            if bar_cat != "-- none --":
                counts = df[bar_cat].value_counts()
                fig, ax = plt.subplots()
                ax.bar(counts.index.astype(str), counts.values)  # matplotlib.bar
                ax.set_xticklabels(counts.index.astype(str), rotation=45, ha="right")
                ax.set_title(f"Counts per {bar_cat}")
                st.pyplot(fig)
            else:
                st.warning("Select a categorical column")

    else:  # scatter / relationship using Altair for interactivity
        if len(numeric_cols) >= 2:
            xcol = st.selectbox("X column", numeric_cols, index=0)
            ycol = st.selectbox("Y column", numeric_cols, index=1)
            color = st.selectbox("Optional color (categorical)", ["-- none --"] + list(df.select_dtypes(include=['object','category']).columns))
            if st.button("Plot scatter"):
                chart_df = df[[xcol, ycol] + ([color] if color != "-- none --" else [])].dropna()
                chart = alt.Chart(chart_df).mark_circle(size=60).encode(
                    x=alt.X(xcol, type='quantitative'),
                    y=alt.Y(ycol, type='quantitative'),
                    color=alt.Color(color, type='nominal') if color != "-- none --" else alt.value('steelblue'),
                    tooltip=list(chart_df.columns)
                ).interactive()
                st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Need at least two numeric columns to show relationships.")

# ---------------------------
# Section 3: Interactive Dashboard (15 marks)
# ---------------------------

st.header("3. Interactive Dashboard (15 marks) — numpy & pandas functions via widgets")

if df is None:
    st.info("Load a dataset to use interactive dashboard.")
else:
    st.subheader("Quick numpy / pandas computations")

    colA, colB, colC = st.columns(3)
    with colA:
        st.write("Numeric column selection")
        num_col = st.selectbox("Numeric column for stats", ["-- none --"] + list(df.select_dtypes(include=[np.number]).columns))
        if st.button("Run basic numpy stats"):
            if num_col != "-- none --":
                arr = df[num_col].to_numpy()  # Series.to_numpy
                st.write("numpy array shape:", arr.shape)
                st.write("mean (np.mean):", np.nanmean(arr))    # np.nanmean
                st.write("median (np.median):", np.nanmedian(arr)) # np.median
                st.write("std (np.std):", np.nanstd(arr))       # np.std
                st.write("min, max:", np.nanmin(arr), np.nanmax(arr))
            else:
                st.warning("Select a numeric column")

    with colB:
        st.write("Pandas group / pivot / value_counts")
        pivot_index = st.selectbox("Pivot index (categorical)", ["-- none --"] + list(df.select_dtypes(include=['object','category']).columns))
        pivot_values = st.selectbox("Pivot value (numeric)", ["-- none --"] + list(df.select_dtypes(include=[np.number]).columns))
        if st.button("Create pivot_table"):
            if pivot_index != "-- none --" and pivot_values != "-- none --":
                pt = pd.pivot_table(df, index=pivot_index, values=pivot_values, aggfunc=np.mean)  # pd.pivot_table
                st.dataframe(pt)
            else:
                st.warning("Pick pivot index and value columns")

    with colC:
        st.write("Unique, counts, top values")
        c = st.selectbox("Column for value_counts / unique", ["-- none --"] + list(df.columns))
        if st.button("Show value_counts & unique"):
            if c != "-- none --":
                st.write("Unique values (pd.Series.unique):")
                st.write(df[c].unique())
                st.write("Value counts (pd.Series.value_counts):")
                st.dataframe(df[c].value_counts().reset_index().rename(columns={'index':c, c:'count'}))
            else:
                st.warning("Select a column")

    st.subheader("Complex query builder (multiple conditions)")
    # Build boolean masks combining up to 2 conditions
    cond1_col = st.selectbox("Condition 1 column", ["-- none --"] + list(df.columns), key="c1")
    cond1_op = st.selectbox("Condition 1 operator", ["==","!=","<",">","<=",">="], key="o1")
    cond1_val = st.text_input("Condition 1 value", key="v1")

    cond2_col = st.selectbox("Condition 2 column", ["-- none --"] + list(df.columns), key="c2")
    cond2_op = st.selectbox("Condition 2 operator", ["==","!=","<",">","<=",">="], key="o2")
    cond2_val = st.text_input("Condition 2 value", key="v2")

    combine = st.radio("Combine conditions with", ("AND","OR"))

    if st.button("Run complex query"):
        try:
            mask = pd.Series([True]*len(df))
            # helper to build mask for one condition
            def build_mask(col, op, val):
                if col == "-- none --" or val == "":
                    return pd.Series([True]*len(df))
                s = df[col]
                # attempt numeric conversion
                try:
                    if "." in val:
                        v = float(val)
                    else:
                        v = int(val)
                except Exception:
                    v = val
                if op == "==":
                    return s == v
                if op == "!=":
                    return s != v
                if op == "<":
                    return s.astype(float) < float(v)
                if op == ">":
                    return s.astype(float) > float(v)
                if op == "<=":
                    return s.astype(float) <= float(v)
                if op == ">=":
                    return s.astype(float) >= float(v)
                return pd.Series([True]*len(df))
            m1 = build_mask(cond1_col, cond1_op, cond1_val)
            m2 = build_mask(cond2_col, cond2_op, cond2_val)
            if combine == "AND":
                final = m1 & m2
            else:
                final = m1 | m2
            res = df.loc[final]
            st.write(f"Query returned {len(res)} rows")
            st.dataframe(res.head(100))
        except Exception as e:
            st.error(f"Query failed: {e}")

# ---------------------------
# Section 4: Writing (5 marks)
# ---------------------------

st.header("4. Writing & Export (5 marks)")
if df is None:
    st.info("Load dataset to export.")
else:
    st.write("Export modified/filtered dataset to multiple formats:")
    export_df = df.copy()
    # optional row selection
    if st.checkbox("Export only filtered rows from previous operations (not implemented persistent)"):
        st.info("Note: to export filtered results, run filter and then use `st.session_state` to store — for demo we export full df.")

    st.download_button("Download CSV (client-side)", data=export_df.to_csv(index=False).encode('utf-8'), file_name="exported.csv", mime="text/csv")
    # Binary download: allow download of pickled bytes
    pickled = pickle.dumps({"ts": datetime.now().isoformat(), "data": export_df})
    st.download_button("Download Pickle (binary)", data=pickled, file_name="exported.pkl", mime="application/octet-stream")
    # XML export
    xml_string = df_to_xml_string(export_df, root_name="dataset", row_name="row")
    st.download_button("Download XML", data=xml_string.encode('utf-8'), file_name="exported.xml", mime="application/xml")

    st.success("Exports above are generated from current dataset copy (server-side for path writes, and client-side using download buttons).")

# ---------------------------
# Section 5: Viva helper (5 marks)
# ---------------------------

st.header("5. Viva / Explanation (5 marks)")
st.markdown("""
This panel lists the key functions and methods used in this app (mention these in your viva):

**I/O & Formats**
- `pd.read_csv`, `DataFrame.to_csv`  (read/write CSV text)
- `pickle.dump`, `pickle.load`        (binary read/write/append)
- `xml.etree.ElementTree` functions (`Element`, `SubElement`, `tostring`, `parse`) (XML write/read)
- `st.download_button` (client-side download)

**Pandas / Data manipulation**
- `pd.DataFrame`, `df.head()`, `df.info()`, `df.describe()` (inspect)
- `df.astype()`, `pd.to_datetime()` (type conversion)
- `df.sort_values()`, `df.groupby().mean()` (sorting & grouping)
- `df.iloc[]`, `df.loc[]` (indexing/slicing/locating)
- `df.query()` (optional), `df.pivot_table()` (pivot)
- `df.value_counts()`, `Series.unique()` (counts & uniques)
- `pd.pivot_table`, `pd.concat` (merging/aggregation — not explicitly used but mentionable)

**Numpy**
- `np.mean`, `np.median`, `np.std`, `np.nanmean` (aggregations)
- `Series.to_numpy()` (conversion to numpy array)

**Visualization**
- `matplotlib.pyplot.hist`, `plt.bar` (histogram & bar)
- `altair.Chart` (interactive scatter/relationship)

**Streamlit widgets used**
- `st.sidebar.file_uploader`, `st.sidebar.radio`, `st.selectbox`, `st.multiselect`, `st.button`, `st.checkbox`, `st.number_input`, `st.text_input`, `st.download_button`, `st.dataframe`, `st.pyplot`, `st.altair_chart`

**Notes to mention in viva**
- Explain difference between text (CSV), binary (pickle) and XML formats: readability, portability, and security (pickle is binary and can execute arbitrary code — mention security risk).
- Demonstrate one example of converting types, sorting and grouping live on the GUI.
- Explain pros/cons of using pandas for in-memory analysis vs DB for large data.

Good luck with the viva!
""")

st.markdown("---")
st.caption("Everything above includes comments near code where functions are used. Modify paths if running on a remote server; file writes go to the server filesystem.")

