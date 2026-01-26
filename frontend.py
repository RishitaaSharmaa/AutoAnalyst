import streamlit as st
import pandas as pd
from main import load_dataset, run
from registry import DATASET_REGISTRY, MODEL_REGISTRY

st.set_page_config(page_title="AutoML LangGraph", layout="wide")

st.title("AutoAnalyst")

uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset loaded successfully!")

    st.dataframe(df.head())

    dataset_id = "user_dataset" 
    load_dataset(dataset_id, df)

    st.info(f"Dataset registered as `{dataset_id}`")

task = st.text_area(
    "Enter your analysis instruction:",
)

if st.button("Run AutoML Agent"):

    if uploaded_file is None:
        st.error("Please upload a dataset first.")
    else:
        with st.spinner("Running AutoML Agent..."):
            response = run(task.replace("sales_data", dataset_id))

        st.subheader("Agent Output")
        st.write(response)

        import os
        for file in os.listdir():
            if file.endswith(".png") and dataset_id in file:
                st.image(file)
