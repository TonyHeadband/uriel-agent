Project: Local AI Household Assistant — Software Layer (hardware-independent, dev on RTX 3070)

Stack: Python, FastAPI, LangGraph, FastMCP, Postgres (+ pgvector), Ollama (OpenAI-compatible endpoint), Docker/k3s

Components to build:

FastAPI gateway — reads Remote-User/Remote-Groups headers (Authelia forward-auth pattern), routes to LangGraph, per-user thread_id
LangGraph agent — StateGraph with messages, user_id, groups state; agent node binds tools, conditional edge to ToolNode; Postgres checkpointer for per-user persistent history
FastMCP tool server — separate service, tools gated by ctx.request_context.meta["groups"], raises ToolException on permission denial. Start with stub tools: homelab_status, search_documents (RAG), door_camera_last_event (mock for now)
Model routing — task-type based model selection (interactive vs. background), config-driven, not hardcoded per deployment
RAG pipeline — embedding + rerank + pgvector ingest/search, as a FastMCP tool
Dev model config — Qwen3-8B (dense, fits 3070) for fast iteration; swap to Qwen3.6-35B-A3B via config once on target hardware

Explicitly out of scope for now: voice (STT/TTS), vision/camera integration, image gen — these are separate services to bolt on later via the same FastMCP pattern, don't build them yet.

Deliverable shape: a runnable docker-compose (gateway + FastMCP + Postgres + Ollama) that takes a chat message with fake Authelia headers and completes a full agent loop, including at least one gated tool call, end to end.

Want me to turn this into an actual starter repo scaffold (files, docker-compose, first passing test) right now, or is this meant purely as the brief to hand off elsewhere?