@echo off
echo Starting AI Public Health Chatbot...
echo Setting up environment variables...
set TF_ENABLE_ONEDNN_OPTS=0
set TF_USE_LEGACY_KERAS=1

echo Using virtual environment...
if exist .venv\Scripts\activate (
    echo Activating .venv...
    call .venv\Scripts\activate
    streamlit run app.py
) else (
    echo Error: .venv not found. Please create it or run: python -m venv .venv
    pause
)
