#!/usr/bin/env bash
set -e

# 启动 FastAPI（后台）
uvicorn app.server:app --host 0.0.0.0 --port 8000 &

# 启动 Streamlit（前台）
streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port 7860
