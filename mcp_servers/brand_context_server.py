"""
Brand Context MCP Server
========================
Exposes the brand RAG (company_data.json + FAISS index) as MCP tools.
Any MCP-compatible client can query brand guidelines, tone rules,
and product information without importing this project's Python code.

Run standalone:
    python3 mcp_servers/brand_context_server.py

Connected via agent in mcp_client.py:
    "brand_context": {
        "command": "python3",
        "args": ["mcp_servers/brand_context_server.py"],
        "transport": "stdio",
    }
"""

import sys
import os

# Add project root to path so services/ imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP
from services.logger    import get_logger

log = get_logger(__name__)
mcp = FastMCP("Brand Context Server")


@mcp.tool()
def retrieve_brand_context(query: str) -> str:
    """
    Retrieve brand guidelines, tone rules, approved claims, and
    product information relevant to a query.
    Use this when you need brand voice, restrictions, or product details.
    """
    from services.rag import retrieve_brand_context as _retrieve
    result = _retrieve(query)
    log.debug(f"brand context retrieved: {len(result)} chars")
    return result


@mcp.tool()
def get_brand_summary() -> str:
    """
    Get a high-level overview of the brand — name, tagline,
    target audience, and tone. Use for a quick brand snapshot.
    """
    from services.rag import load_company_data
    data  = load_company_data()
    brand = data.get("brand", {})

    lines = [
        f"Brand:    {brand.get('name',            'Unknown')}",
        f"Tagline:  {brand.get('tagline',         '')}",
        f"Audience: {brand.get('target_audience', '')}",
        f"Tone:     {brand.get('tone',            '')}",
    ]
    return "\n".join(lines)


@mcp.tool()
def check_brand_compliance(content: str) -> str:
    """
    Check whether a piece of content follows brand guidelines.
    Returns approved claims, forbidden phrases, and a compliance note.
    """
    from services.rag import retrieve_brand_context as _retrieve

    # Retrieve guidelines relevant to the content
    guidelines = _retrieve(content)

    return (
        f"Brand guidelines relevant to this content:\n\n"
        f"{guidelines}\n\n"
        f"Review the above and check: forbidden phrases, tone match, "
        f"approved claims only."
    )


if __name__ == "__main__":
    mcp.run()