import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

CAMPAIGNS_DIR = Path("campaigns")
INDEX_DIR     = Path("memory/campaign_index")

_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


# ── Helpers ────────────────────────────────────────────────────

def _load_all_campaigns() -> list[dict]:
    """Read every campaign JSON from campaigns/ folder."""
    campaigns = []
    if not CAMPAIGNS_DIR.exists():
        return campaigns
    for path in sorted(CAMPAIGNS_DIR.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data             = json.load(f)
                data["_filename"] = str(path)
                campaigns.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    return campaigns


def _to_document(campaign: dict) -> Document:
    """Convert a campaign dict into an embeddable LangChain Document."""
    user_prompt = campaign.get("user_prompt", "")
    content     = campaign.get("content", campaign.get("full_post", ""))[:300]
    status      = campaign.get("status", "unknown")
    hashtags    = " ".join(campaign.get("hashtags", []))

    text = (
        f"User wanted: {user_prompt}\n"
        f"Content: {content}\n"
        f"Hashtags: {hashtags}\n"
        f"Status: {status}"
    )

    return Document(
        page_content = text,
        metadata     = {
            "status":      status,
            "timestamp":   campaign.get("timestamp", ""),
            "filename":    campaign.get("_filename", ""),
            "user_prompt": user_prompt,
            "hashtags":    ", ".join(campaign.get("hashtags", [])),
        },
    )


# ── Index management ───────────────────────────────────────────

def build_campaign_index() -> FAISS | None:
    """Embed all campaigns and save index to disk."""
    campaigns = _load_all_campaigns()
    if not campaigns:
        print("  [Memory] No campaigns yet — index will be created after first run")
        return None

    docs  = [_to_document(c) for c in campaigns]
    index = FAISS.from_documents(docs, _embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index.save_local(str(INDEX_DIR))
    print(f"  [Memory] Index built — {len(docs)} campaigns embedded")
    return index


def load_campaign_index() -> FAISS | None:
    """Load the existing index from disk."""
    if not (INDEX_DIR / "index.faiss").exists():
        return None
    return FAISS.load_local(
        str(INDEX_DIR),
        _embeddings,
        allow_dangerous_deserialization=True,
    )


def load_or_build_index() -> FAISS | None:
    """Load existing index, or build one from scratch if it doesn't exist."""
    index = load_campaign_index()
    if index is not None:
        # Count stored vectors
        n = index.index.ntotal
        print(f"  [Memory] Campaign index loaded — {n} campaigns")
        return index
    return build_campaign_index()


def add_campaign_to_index(campaign_path: str) -> None:
    """Add one new campaign to the index after a run completes."""
    try:
        with open(campaign_path, "r", encoding="utf-8") as f:
            campaign              = json.load(f)
            campaign["_filename"] = campaign_path
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [Memory] Could not read campaign: {e}")
        return

    doc   = _to_document(campaign)
    index = load_campaign_index()

    if index is None:
        index = FAISS.from_documents([doc], _embeddings)
    else:
        index.add_documents([doc])

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index.save_local(str(INDEX_DIR))
    status = campaign.get("status", "unknown")
    print(f"  [Memory] Added to index — status: {status} "
          f"· total: {index.index.ntotal}")


def rebuild_index() -> None:
    """Force a full rebuild — useful after importing old campaigns."""
    import shutil
    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)
        print("  [Memory] Old index cleared")
    build_campaign_index()


# ── Search ─────────────────────────────────────────────────────

def search_campaigns(
    query:         str,
    k:             int       = 3,
    filter_status: str | None = None,
) -> list[Document]:
    """
    Semantic search over past campaigns.

    Args:
        query:         what to search for (free text)
        k:             number of results to return
        filter_status: 'posted', 'denied', or None for all

    Returns:
        list of matching Documents with metadata attached
    """
    index = load_campaign_index()
    if index is None:
        return []

    fetch_k = k * 3 if filter_status else k
    results = index.similarity_search(query, k=fetch_k)

    if filter_status:
        results = [
            r for r in results
            if r.metadata.get("status") == filter_status
        ]

    return results[:k]


def get_few_shot_examples(query: str, k: int = 2) -> str:
    """
    Find k approved campaigns similar to this query and format
    them as few-shot examples for the copywriter.
    Returns empty string if nothing relevant found.
    """
    index = load_campaign_index()
    if index is None:
        return ""

    # Use score to filter out poor matches
    results_with_scores = index.similarity_search_with_score(query, k=k * 3)

    # Lower L2 distance = more similar. Threshold: 1.2
    filtered = [
        doc for doc, score in results_with_scores
        if score < 1.2 and doc.metadata.get("status") == "posted"
    ][:k]

    if not filtered:
        return ""

    lines = ["=== Similar approved campaigns (use as style reference) ==="]
    for i, doc in enumerate(filtered, 1):
        meta    = doc.metadata
        content = ""
        for line in doc.page_content.splitlines():
            if line.startswith("Content:"):
                content = line[len("Content:"):].strip()
                break

        lines.append(f"\n[Example {i}]")
        lines.append(f"Topic    : {meta.get('user_prompt', '')[:120]}")
        lines.append(f"Content  : {content[:250]}")
        lines.append(f"Hashtags : {meta.get('hashtags', '')}")

    lines.append("\n=== End of examples ===")
    return "\n".join(lines)