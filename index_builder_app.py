import streamlit as st
import os
from build_index import main as build_index_main

st.set_page_config(page_title="Build Index", layout="centered")
st.title("Build FAISS Index for Kristalina Speeches")

st.write("This will create `index.faiss`, `meta.json`, and `chunks.json` in the app folder using your dataset.")
key = os.getenv("OPENAI_API_KEY")
if not key:
    st.error("OPENAI_API_KEY is not set in Streamlit Secrets. Set it under Settings → Secrets and redeploy.")
    st.stop()

if st.button("Build index now"):
    with st.spinner("Building index… this can take several minutes on first run. Please keep this page open."):
        try:
            build_index_main()
            st.success("Index built successfully! You can now switch the app back to app_llm.py.")
        except Exception as e:
            st.error(f"Error: {e}")
