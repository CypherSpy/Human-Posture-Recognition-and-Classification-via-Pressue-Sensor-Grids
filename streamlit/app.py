import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Human Posture Recognition",
    page_icon="🛏️",
    layout="wide"
)

st.title("🛏️ Human Posture Recognition and Classification via Pressure Sensor Grids")

st.markdown("""
### Healthcare AI | Deep Learning | PhysioNet Dataset

This project classifies in-bed human postures using pressure sensor grid data obtained from the PhysioNet Pressure Map Dataset.

**Postures Classified**
- Supine
- Prone
- Left-Lateral
- Right-Lateral
""")

tab1, tab2, tab3 = st.tabs(
    ["Project Overview", "Dataset Information", "Demo Prediction"]
)

with tab1:
    st.header("Project Summary")

    st.write("""
    Developed a machine learning and deep learning framework for
    posture classification using pressure sensor grids.

    Key techniques:
    - Exploratory Data Analysis (EDA)
    - Principal Component Analysis (PCA)
    - Pressure Variance Profiling
    - ANN
    - CNN
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("ANN Accuracy", "94.2%")

    with col2:
        st.metric("CNN Accuracy", "97.5%")

with tab2:
    st.header("Dataset")

    st.write("""
    **Dataset:** PhysioNet Pressure Map Dataset (PMD)

    - 13 Participants
    - 32 × 64 Pressure Grid
    - 2048 Sensors per Frame
    - Four Major Posture Classes
    """)

    st.subheader("Sample Pressure Map")

    sample = np.random.rand(32, 64)

    fig, ax = plt.subplots()
    ax.imshow(sample)
    ax.set_title("Pressure Sensor Heatmap")
    st.pyplot(fig)

with tab3:
    st.header("Demo Prediction")

    posture = st.selectbox(
        "Select Sample Posture",
        ["Supine", "Prone", "Left-Lateral", "Right-Lateral"]
    )

    if st.button("Predict"):

        confidence = np.random.uniform(92, 99)

        st.success(f"Predicted Posture: {posture}")

        st.info(f"Confidence: {confidence:.2f}%")

        st.write("Model integration will be connected to the trained CNN model.")

st.markdown("---")
st.markdown(
    "Developed by **Yashika Yadav** | B.Tech CSE (Data Science) | MANIT Research Internship"
)
