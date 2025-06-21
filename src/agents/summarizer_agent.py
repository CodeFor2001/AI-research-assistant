# src/agents/summarizer_agent.py
"""
SummarizerAgent module: generates summaries for papers in mcp.search_results
using the OpenAI v1 client, integrates MLOps via MLflow logging and
DVC-based data versioning, and handles rate‐limit errors gracefully.
"""
import os
import json
import hashlib
import time
import logging
from pathlib import Path

import mlflow
import openai
from openai import OpenAI

from src.agents.mcp import MCP

logger = logging.getLogger(__name__)

class SummarizerAgent:
    """
    Agent responsible for summarizing paper abstracts via OpenAI LLM.
    Logs prompt, model, and outputs to MLflow; writes summaries for DVC.
    Catches RateLimitError to avoid pipeline failure.
    """
    def __init__(
        self,
        model_name: str = "gpt-4",
        prompt_template: str = "summarize_prompt.txt",
        summary_dir: str = "data/summaries",
        max_tokens: int = 256,
        retry_delay: float = 2.0
    ):
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.summary_dir = summary_dir
        self.max_tokens = max_tokens
        self.retry_delay = retry_delay

        # v1 client
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI()

        # load prompt template
        project_root = Path(__file__).resolve().parents[2]
        template_path = project_root / "prompts" / prompt_template
        if not template_path.exists():
            raise FileNotFoundError(f"Prompt template not found at {template_path}")
        self.template = template_path.read_text(encoding="utf-8")

    def run(self, mcp: MCP) -> MCP:
        os.makedirs(self.summary_dir, exist_ok=True)
        mlflow.log_param("summarizer_model", self.model_name)
        mlflow.log_param("prompt_template", self.prompt_template)

        summaries = []
        for paper in mcp.search_results:
            title    = paper.get("title", "")
            abstract = paper.get("abstract", "")
            url      = paper.get("url", "")

            # Build prompt
            prompt = self.template.format(
                title=title,
                abstract=abstract,
                url=url
            )

            # Log prompt hash
            prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
            safe_key    = hashlib.md5(title.encode()).hexdigest()
            mlflow.log_param(f"prompt_hash_{safe_key}", prompt_hash)

            # Invoke API, handle rate limits
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens
                )
                summary_text = response.choices[0].message.content.strip()
            except openai.RateLimitError as e:
                logger.warning("Rate limit hit for '%s': %s", title, e)
                mlflow.log_param("rate_limit_error", True)
                summary_text = ""
                time.sleep(self.retry_delay)

            # Build and store record
            record = {
                "title": title,
                "summary": summary_text,
                "model": self.model_name,
                "prompt_hash": prompt_hash
            }
            summaries.append(record)

            # Write JSON for DVC
            filename = f"summary_{safe_key}.json"
            filepath = os.path.join(self.summary_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)

            # Log artifact
            mlflow.log_artifact(filepath, artifact_path="summaries")

        mcp.summaries = summaries
        return mcp
