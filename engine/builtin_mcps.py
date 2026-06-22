"""Pre-set MCP server configurations.
Users configure credentials; everything else is pre-filled.
"""

PRESET_MCPS: list[dict] = [
    {
        "id":          "preset_slack",
        "name":        "Slack",
        "icon":        "💬",
        "description": "Post messages, read channels, manage Slack workspaces",
        "command":     "npx",
        "args":        ["-y", "@modelcontextprotocol/server-slack"],
        "env_vars": [
            {"key": "SLACK_BOT_TOKEN",  "label": "Bot Token (xoxb-…)",   "required": True},
            {"key": "SLACK_TEAM_ID",    "label": "Team ID (optional)",    "required": False},
            {"key": "SLACK_CHANNEL_ID", "label": "Default Channel ID",   "required": False},
        ],
    },
    {
        "id":          "preset_gmail",
        "name":        "Gmail",
        "icon":        "📧",
        "description": "Send and read emails via Gmail OAuth",
        "command":     "npx",
        "args":        ["-y", "@gongrzhe/server-gmail-autoauth-mcp"],
        "env_vars": [
            {"key": "GMAIL_CLIENT_ID",     "label": "OAuth Client ID",     "required": True},
            {"key": "GMAIL_CLIENT_SECRET", "label": "OAuth Client Secret", "required": True},
        ],
    },
    {
        "id":          "preset_github",
        "name":        "GitHub",
        "icon":        "🐱",
        "description": "Manage repositories, issues, pull requests and code",
        "command":     "npx",
        "args":        ["-y", "@modelcontextprotocol/server-github"],
        "env_vars": [
            {"key": "GITHUB_PERSONAL_ACCESS_TOKEN", "label": "Personal Access Token", "required": True},
        ],
    },
    {
        "id":          "preset_notion",
        "name":        "Notion",
        "icon":        "📝",
        "description": "Read and write Notion pages and databases",
        "command":     "npx",
        "args":        ["-y", "@notionhq/notion-mcp-server"],
        "env_vars": [
            {"key": "NOTION_API_TOKEN", "label": "Integration Token (secret_…)", "required": True},
        ],
    },
    {
        "id":          "preset_fetch",
        "name":        "Web Fetch",
        "icon":        "🌐",
        "description": "Fetch and process any web page — no credentials needed",
        "command":     "npx",
        "args":        ["-y", "@modelcontextprotocol/server-fetch"],
        "env_vars":    [],
    },
    {
        "id":          "preset_filesystem",
        "name":        "Filesystem",
        "icon":        "📁",
        "description": "Read and write local files within allowed directories",
        "command":     "npx",
        "args":        ["-y", "@modelcontextprotocol/server-filesystem", "./"],
        "env_vars": [
            {"key": "FILESYSTEM_ROOT", "label": "Allowed root path (default: ./)", "required": False},
        ],
    },
    {
        "id":          "preset_discord",
        "name":        "Discord",
        "icon":        "🎮",
        "description": "Post messages and interact with Discord channels",
        "command":     "npx",
        "args":        ["-y", "discord-mcp-server"],
        "env_vars": [
            {"key": "DISCORD_TOKEN",      "label": "Bot Token",    "required": True},
            {"key": "DISCORD_CHANNEL_ID", "label": "Channel ID",   "required": False},
        ],
    },
    {
        "id":          "preset_postgres",
        "name":        "PostgreSQL",
        "icon":        "🗄️",
        "description": "Query and manage PostgreSQL databases",
        "command":     "npx",
        "args":        ["-y", "@modelcontextprotocol/server-postgres"],
        "env_vars": [
            {"key": "POSTGRES_CONNECTION_STRING", "label": "Connection string (postgresql://…)", "required": True},
        ],
    },
]
