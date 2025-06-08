import mlflow
from src.agents.mcp import MCP
from src.agents.search_agent import SearchAgent
from src.agents.summarizer_agent import SummarizerAgent

class OrchestratorAgent:

  def __init__(self,
                 max_search_results: int = 5,
                 summary_model: str = "gpt-4",
                 summary_prompt: str = "summarize_prompt.txt"):
        self.search_agent = SearchAgent(max_results=max_search_results)
        self.summary_agent = SummarizerAgent(
            model_name=summary_model,
            prompt_template=summary_prompt
        )

  def run(self, topic: str) -> MCP:

    with mlflow.start_run(run_name=f"pipeline_{topic}"):
        mlflow.log_param("pipeline_topic", topic)

        mcp = MCP(topic)

        mcp = self.search_agent.run(mcp)
        mcp = self.summary_agent.run(mcp)

        mlflow.log_param("num_search_results", len(mcp.search_results))
        mlflow.log_param("num_summaries", len(mcp.summaries))

    return mcp
