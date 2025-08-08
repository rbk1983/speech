# Kristalina Georgieva Speech Intelligence App

This app lets you search, compare, and analyze Kristalina Georgieva's speeches using LLM-powered summaries, thematic analysis, and quote extraction.

## Features
- Search speeches by keyword or theme
- View speeches sorted by most recent first
- Quick Compare narrative: issue-focused summaries for two years
- Messaging evolution analysis between years
- Top 5 quotes with date and source link
- Downloadable briefing pack

## Usage (Local)
1. Clone or download this repo.
2. Place your dataset file (e.g., speeches_data.pkl) in the folder.
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Set your OpenAI API key:
```bash
export OPENAI_API_KEY="sk-..."
```
5. Build the index:
```bash
python build_index.py
```
6. Run the app:
```bash
streamlit run app_llm.py
```

## Deployment
You can deploy this to Streamlit Cloud by uploading these files to a GitHub repo, setting your OpenAI API key in Streamlit Secrets, and deploying.
