import os
import json
import hashlib
import mlflow
import arxiv
from src.agents.mcp import MCP

class SearchAgent:

  def __init__(self,max_results: int = 5, raw_data_dir: str = "data/raw"):
    self.max_results: int = max_results
    self.raw_data_dir: str = raw_data_dir


  def run(self, mcp: MCP) -> MCP:

    #start MLflow
    mlflow.log_param("search_topic", mcp.topic)
    mlflow.log_param("max_results", self.max_results)

    #perform search
    search = arxiv.Search(query=mcp.topic, 
                          max_results=self.max_results,
                          sort_by=arxiv.SortCriterion.Relevance)
    
    results = []

    for paper in search.results():
      results.append({
        "title": paper.title,
        "authors": [str(a) for a in paper.authors],
        "abstract": paper.summary,
        "url": paper.entry_id,
        "published": paper.published.isoformat()
      })

    mcp.search_results = results #update the mcp object

    os.makedirs(self.raw_data_dir, exist_ok=True)

    #save the search results to a file
    topic_hash = hashlib.md5(mcp.topic.encode()).hexdigest()
    filename = f"search_{topic_hash}.json"
    filepath = os.path.join(self.raw_data_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
      json.dump(results, f, ensure_ascii=False, indent=2)

    #log the search results to MLflow
    mlflow.log_artifact(filepath, artifact_path="search_results")

    return mcp


  
