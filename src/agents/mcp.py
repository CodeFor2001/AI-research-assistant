# src/agents/mcp.py

class MCP:

    def __init__(self, topic: str):
        self.topic: str = topic # “Your input topic”
        self.query_refined: bool = False #if the query is refined
        self.search_results: list = [] #raw output from Arxiv api
        self.selected_papers: list = [] #subset of "search_results" that are relevant to the topic
        self.summaries: list = [] #agent generated summaries of the selected papers {"title": title, "summary": summary}
        self.temporal_trends: list = [] #patterns from trend analysis
        self.knowledge_graph_nodes: list = [] #nodes for the knowledge graph
        self.citation_links: list = [] #relation between nodes/papers
        self.cache_hash: str = None #unique key to store or retrieve this context in a cache

    def to_dict(self) -> dict:
        return self.__dict__.copy() 

    @classmethod
    def from_dict(cls, data: dict): #new mcp object , updates the object with the data in the dict
        obj = cls(data.get("topic", ""))
        obj.__dict__.update(data)
        return obj
