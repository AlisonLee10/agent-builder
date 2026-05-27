import json
from pathlib import Path
from services.database import init_db, insert_campaign

init_db()
campaigns_dir = Path("campaigns")

if not campaigns_dir.exists():
    print("No campaigns/ folder found — nothing to migrate.")
else:
    migrated = 0
    for path in campaigns_dir.glob("*.json"):
        try:
            with open(path) as f:
                data = json.load(f)
            data.setdefault("platform", "")
            data.setdefault("sources",  "")
            data.setdefault("articles", [])
            insert_campaign(data)
            migrated += 1
            print(f"  ✅ {path.name}")
        except Exception as e:
            print(f"  ⚠️  {path.name}: {e}")
    print(f"\nMigrated {migrated} campaigns → campaigns.db")