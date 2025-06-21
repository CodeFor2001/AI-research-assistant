# src/api.py
"""
FastAPI application for the AI-powered Research Assistant Agent.
Provides endpoints to execute the research pipeline and health checks.
"""
from dotenv import load_dotenv
load_dotenv()  # loads variables from .env into os.environ

import os
import logging
import traceback
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import mlflow

from src.agents.orchestrator import OrchestratorAgent
from src.agents.mcp import MCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure OpenAI API key is set
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    logger.error("OPENAI_API_KEY not set in environment variables")
    raise RuntimeError("OPENAI_API_KEY not configured")

# Request model for topic input
class TopicRequest(BaseModel):
    topic: str

# Initialize FastAPI app with metadata
app = FastAPI(
    title="AI Research Assistant Agent API",
    description="Endpoints to run an end-to-end research pipeline: search, summarize, and more.",
    version="0.1.0",
)

# Enable CORS (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/run", response_model=Dict[str, Any])
async def run_agent(req: TopicRequest) -> Dict[str, Any]:
    """
    Execute the full AI research pipeline for a given topic.
    Returns the entire MCP context as JSON.
    """
    topic = req.topic
    try:
        # Start an MLflow run for tracking the pipeline
        with mlflow.start_run(run_name=f"pipeline_{topic}"):
            logger.info("Starting pipeline for topic: %s", topic)
            orchestrator = OrchestratorAgent()
            mcp: MCP = orchestrator.run(topic)
            logger.info("Pipeline completed: %d summaries generated", len(mcp.summaries))
        return mcp.to_dict()
    except Exception:
        # Log full traceback for debugging
        logger.error("Exception in /run endpoint:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/health", response_model=Dict[str, str])
async def health_check() -> Dict[str, str]:
    """
    Simple health check endpoint.
    """
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("src.api:app", host="0.0.0.0", port=port, reload=True)
