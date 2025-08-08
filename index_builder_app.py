# index_builder_app.py (Streamlit helper to build index in the cloud)
import streamlit as st
import os
from build_index import main as build_index_main

st.set_page_config(page_title="Build FAISS Index", layout="centered")
st.title("Build FAISS Index for Kristalina Speeches")

st.write("This will create `index.faiss`, `meta.json`, and `chunks.json` in your app folder, using your dataset.")

# Check API key
if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is not set. In Streamlit Cloud, go to Settings → Secrets and add it, then redeploy.")
    st.stop()

# Check dataset presence
has_reclass = os.path.exists("speeches_data_reclassified.pkl")
has_base = os.path.exists("speeches_data.pkl")
if not (has_reclass or has_base):
    st.error("Dataset not found. Upload 'speeches_data_reclassified.pkl' or 'speeches_data.pkl' to the repo root and redeploy this page.")
    st.stop()

if st.button("Build index now"):
    with st.spinner("Building index… this can take several minutes. Please keep this page open."):
        try:
            build_index_main()
            st.success("✅ Index built successfully! You can now switch your main app to `app_llm.py`.")
        except Exception as e:
            st.exception(e)
