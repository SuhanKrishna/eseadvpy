import streamlit as st
import pandas as pd
import numpy as np
import os, pickle, xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Lab Exam App", layout="wide")
st.title("Lab Exam - File Handling + Data Analysis Dashboard")

# ------------------- FILE HANDLING SECTION -------------------
file_type = st.selectbox("Select File Type", ["Text", "Binary", "XML"])
mode = st.radio("Choose Operation", ["Read", "Write", "Append", "Search"])
filename = st.text_input("Enter file name", "records")

if file_type == "Text": filename += ".txt"
elif file_type == "Binary": filename += ".bin"
else: filename += ".xml"

name = st.text_input("Enter Student Name")
uid = st.text_input("Enter Student ID")
marks = st.number_input("Enter Marks", 0, 100, 50)

record = {"id": uid, "name": name, "marks": marks}

# ------------ File Operations ------------
def write_data():
    if file_type=="Text": open(filename,"w").write(str([record]))
    elif file_type=="Binary": pickle.dump([record], open(filename,"wb"))
    else:
        root=ET.Element("records"); r=ET.SubElement(root,"record")
        for k,v in record.items(): ET.SubElement(r,k).text=str(v)
        ET.ElementTree(root).write(filename)

def append_data():
    data=[]
    if os.path.exists(filename):
        if file_type=="Text": data=eval(open(filename).read())
        elif file_type=="Binary": data=pickle.load(open(filename,"rb"))
        else:
            tree=ET.parse(filename); root=tree.getroot()
            for r in root: data.append({c.tag:c.text for c in r})

    # Update existing record if ID exists
    updated = False
    for rec in data:
        if rec["id"] == uid:
            rec["name"] = name
            rec["marks"] = marks
            updated = True
            break

    if not updated:
        data.append(record)  # add new record if ID not found

    # Save back to file
    if file_type=="Text": open(filename,"w").write(str(data))
    elif file_type=="Binary": pickle.dump(data,open(filename,"wb"))
    else:
        root=ET.Element("records")
        for rec in data:
            r=ET.SubElement(root,"record")
            for k,v in rec.items(): ET.SubElement(r,k).text=str(v)
        ET.ElementTree(root).write(filename)

def read_data():
    if not os.path.exists(filename): return []
    if file_type=="Text": return eval(open(filename).read())
    elif file_type=="Binary": return pickle.load(open(filename,"rb"))
    else:
        tree=ET.parse(filename); root=tree.getroot()
        return [{c.tag:c.text for c in r} for r in root]

if st.button("Execute Operation"):
    if mode=="Write": write_data(); st.success("Data Written")
    elif mode=="Append": append_data(); st.success("Data Appended/Updated")
    elif mode=="Read": st.write(read_data())
    elif mode=="Search":
        data=read_data(); res=[r for r in data if r["id"]==uid]
        st.write(res if res else "Not Found")

# ------------------- DATA ANALYSIS SECTION -------------------
st.header("Data Analysis with NumPy & Pandas")
upload = st.file_uploader("Upload CSV Dataset", type=["csv"])

if upload:
    df = pd.read_csv(upload)
    st.subheader("Dataset Preview")
    st.write(df.head())

    # Info
    st.write("Shape:", df.shape)
    st.write("Columns:", df.columns.tolist())
    st.write("Description:", df.describe())

    # Handle Missing Values
    for col in df.columns:
        if df[col].isna().mean() > 0.5:
            df = df.drop(columns=[col])
    df = df.fillna(df.mean(numeric_only=True))

    # Convert combined_Id → numeric
    if "combined_Id" in df.columns:
        df["numeric_Id"] = df["combined_Id"].str.extract(r'(\d+)').astype(int)

    # Select numeric columns
    num_cols = df.select_dtypes(include=np.number).columns

    # Standardization (Z-score or Min-Max)
    st.subheader("Data Standardization")
    scaling_method = st.selectbox("Select Standardization Method", ["Z-score", "Min-Max"])
    if st.button("Apply Standardization"):
        df_scaled = df.copy()
        if scaling_method == "Z-score":
            df_scaled[num_cols] = (df[num_cols]-df[num_cols].mean())/df[num_cols].std()
            st.success("Z-score normalization applied!")
        else:
            df_scaled[num_cols] = (df[num_cols]-df[num_cols].min())/(df[num_cols].max()-df[num_cols].min())
            st.success("Min-Max scaling applied!")
        st.write(df_scaled.head())

    # Aggregations
    st.subheader("Aggregations")
    st.write("Mean:", df[num_cols].mean())
    st.write("Median:", df[num_cols].median())
    st.write("Std Dev:", df[num_cols].std())
    st.write("Min:", df[num_cols].min())
    st.write("Max:", df[num_cols].max())

    # Mask & Boolean filtering
    st.subheader("Mask Example")
    if len(num_cols):
        mask = df[num_cols[0]] > 0
        st.write(df[mask].head())

    # Visualization
    st.subheader("Visualization")
    col = st.selectbox("Select column for Histogram", num_cols)
    fig, ax = plt.subplots()
    sns.histplot(df[col], kde=True, ax=ax)
    st.pyplot(fig)

    # Dashboard Interactions
    st.subheader("Interactive Queries")
    query_col = st.selectbox("Select Column to Filter", df.columns)
    val = st.text_input("Enter Value to Filter")
    if st.button("Apply Filter"):
        st.write(df[df[query_col].astype(str)==val])
