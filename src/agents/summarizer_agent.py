import os
import json
import hashlib
import mlflow
import openai
from src.agents.mcp import MCP

class SummarizerAgent:

  def __init__(self, model_name: str = "gpt-3.5-turbo",
               prompt_template: str = "summarize_prompt.txt", summary_dir: str = "data/summaries",
               max_tokens: int = 256):
    
    self.model_name = model_name
    self.prompt_template = prompt_template
    self.summary_dir = summary_dir
    self.max_tokens = max_tokens

    template_path = os.path.join("prompts", self.prompt_template)
    with open(template_path, "r", encoding="utf-8") as f:
      self.template = f.read()
    
  def run(self, mcp: MCP) -> MCP:
    
    #start MLflow
    os.makedirs(self.summary_dir, exist_ok=True)
    mlflow.log_param("summariser_model", self.model_name)
    mlflow.log_param("promt_template", self.prompt_template)

    summaries = []

    for paper in mcp.search_results:
      prompt = self.template.format(titel = paper["title"], abstract=paper["abstract"], url = paper["url"])

      prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
      mlflow.log_param("prompt_hash", prompt_hash)

      #call openai api llm
      response = openai.ChatCompletion.create(
        model=self.model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=self.max_tokens,
      )

      summary_text = response.choices[0].message.content.strip()

      #build the summary record
      record = {
        "title": paper["title"],
        "summary": summary_text,
        "model": self.model_name,
        "prompt_hash": prompt_hash
      }
      summaries.append(record)

      #write the summary to a file for dvc tracking
      filename = f"summary_{hashlib.md5(prompt.encode()).hexdigest()}.json"
      filepath = os.path.join(self.summary_dir, filename)
      with open(filepath, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
      mlflow.log_artifact(filepath, artifact_path="summaries")

    mcp.summaries = summaries #update the mcp object

    return mcp
  
        
