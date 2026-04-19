from mcp.server import FastMCP

mcp = FastMCP("My Server")

mcp.run(transport="streamable-http")