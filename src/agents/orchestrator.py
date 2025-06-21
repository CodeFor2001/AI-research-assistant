# src/agents/orchestrator.py
"""
OrchestratorAgent ties together all agents into a single pipeline.
Integrates MLOps via MLflow run context and parameter logging.
"""
import mlflow
from src.agents.mcp import MCP
from src.agents.search_agent import SearchAgent
from src.agents.summarizer_agent import SummarizerAgent
# from src.agents.trend_agent import TrendAgent
# from src.agents.graph_agent import GraphAgent

class OrchestratorAgent:
    """
    Main runner that orchestrates the end-to-end research assistant pipeline.
    """

    def __init__(
        self,
        max_search_results: int = 5,
        summary_model: str = "gpt-3.5-turbo",
        summary_prompt: str = "summarize_prompt.txt",
    ):
        self.search_agent = SearchAgent(max_results=max_search_results)
        self.summary_agent = SummarizerAgent(
            model_name=summary_model,
            prompt_template=summary_prompt,
        )
        # self.trend_agent  = TrendAgent()
        # self.graph_agent  = GraphAgent()

    def run(self, topic: str) -> MCP:
        """
        Run the full pipeline: search, summarize, (optional: trend, graph).
        Uses nested MLflow runs to ensure idempotency across multiple calls.
        """
        # If a run is already active, we’ll start a nested run
        run_kwargs = {"run_name": f"pipeline_{topic}", "nested": True}

        # Optionally, ensure no runaway active run:
        if mlflow.active_run() is not None and not run_kwargs.get("nested", False):
            mlflow.end_run()

        with mlflow.start_run(**run_kwargs):
            # Log the high-level pipeline param
            mlflow.log_param("pipeline_topic", topic)

            # Initialize and execute context
            mcp = MCP(topic)
            mcp = self.search_agent.run(mcp)
            mcp = self.summary_agent.run(mcp)
            # mcp = self.trend_agent.run(mcp)
            # mcp = self.graph_agent.run(mcp)

            # Final metrics
            mlflow.log_param("num_search_results", len(mcp.search_results))
            mlflow.log_param("num_summaries", len(mcp.summaries))

        return mcp
