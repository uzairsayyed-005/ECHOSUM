# ECHOSUM

Instant, speaker-aware meeting summaries, action extraction, and traceability — demo-ready with on‑prem Whisper support and LLM analysis.

## Key features
- Real-time 2–4 sentence summaries and topic extraction  
- Speaker-aware diarization and action attribution (owner, due date, priority)  
- Traceability: every summary/action links back to transcript snippets & timestamps  
- On‑prem Whisper + cloud STT support for privacy-flexible deployments  
- Lightweight Streamlit demo, CLI hooks, CSV/JSON exports, and integration stubs (Jira/Slack)  
- Testable pipeline with sample test scripts and batch export for evaluation

## Quickstart (Windows)
1. Create & activate virtual env
   .venv\Scripts\activate
2. Install dependencies
   pip install -r requirements.txt
3. Run a quick test (uses sample media path in test_run.py)
   python test_run.py
4. Run demo UI (if Streamlit is available)
   streamlit run streamlit_app.py

(If using a fresh PowerShell session: python -m venv .venv && .\.venv\Scripts\Activate.ps1)

## Typical workflow
1. Record meeting audio (live or upload recording)  
2. Transcribe & diarize (Whisper or Cloud STT)  
3. Preprocess text and normalize speaker IDs  
4. Run LLM analysis (topics, sentiment, summaries)  
5. Extract action items (owners, due-dates, priorities) and attach transcript snippets  
6. Export results to UI, CSV/JSON, or integrations

## Files of interest (core logic)
- main.py — pipeline entry points (transcription, diarization helpers)  
- llm_analysis.py — LLM prompts and parsing for summaries, topics, actions  
- analysis_proj.py — orchestration & post-processing utilities  
- streamlit_app.py — demo UI  
- test_run.py / test_whisper.py — test scripts and example runs  
- requirements.txt / LLM_SETUP.md — environment & LLM setup notes

## Running tests
- Unit / smoke test:
  python test_run.py
- Add more test cases in testing/ and run correspondingly.

## Deployment & privacy notes
- For private meetings, prefer Whisper (on‑prem) path to keep audio local.  
- Cloud STT path available for lower infrastructure overhead.


