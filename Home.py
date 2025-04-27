# Home.py

import streamlit as st

st.set_page_config(page_title="Climate Trend Analysis", layout="wide")

st.image("assets/cover_image.jpg", use_column_width=True)

st.title("🌍 Climate Data Trend Analysis Web App")
st.markdown("""
Welcome to the Climate Data Trend Analysis Web Application!

- Upload your climate data 📈
- Perform multiple trend analyses 📊
- Download results for further study 💾

Navigate to the **Trend Analysis** page to get started.
""")
