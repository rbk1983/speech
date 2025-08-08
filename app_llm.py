import streamlit as st
from rag_utils import retrieve, llm_complete

st.title("Kristalina Georgieva Speech Intelligence")

query = st.text_input("Search speeches")
if query:
    results = retrieve(query, k=20)
    st.subheader("Most Relevant Speeches")
    for r in results:
        st.markdown(f"### [{r['title']}]({r['link']}) — {r['date']}")
        snippet = llm_complete(f"Summarize in 3 sentences: {r['chunk']}")
        st.write(snippet)

    st.subheader("Quick Compare Narrative")
    year_a = st.selectbox("Year A", sorted({r['date'][:4] for r in results}))
    year_b = st.selectbox("Year B", sorted({r['date'][:4] for r in results}))
    if year_a and year_b:
        text_a = " ".join([r['chunk'] for r in results if r['date'].startswith(year_a)])
        text_b = " ".join([r['chunk'] for r in results if r['date'].startswith(year_b)])
        compare_prompt = f"Compare Kristalina Georgieva's speeches on '{query}' in {year_a} vs {year_b}. Focus on key issues, shifts in emphasis, and new or dropped themes."
        st.write(llm_complete(compare_prompt))
