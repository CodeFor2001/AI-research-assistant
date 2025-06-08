from src.agents.mcp import MCP


def test_mcp_init():
  topic = "diffusion models"
  mcp = MCP(topic)
  assert mcp.topic == topic
  assert mcp.search_results == []
  assert mcp.cache_hash == None
