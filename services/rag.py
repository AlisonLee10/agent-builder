import json
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from services.logger import get_logger

load_dotenv()

log = get_logger(__name__)

COMPANY_DATA_PATH = "company_data.json"

def load_company_data() -> dict:
    with open(COMPANY_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def chunk_company_data(data: dict) -> list[Document]:
    """Convert each section of company data into a LangChain Document."""
    docs = []

    brand = data.get("brand", {})
    docs.append(Document(
        page_content=(
            f"Brand: {brand.get('name')}. "
            f"Tagline: {brand.get('tagline')}. "
            f"Mission: {brand.get('mission')}. "
            f"Category: {brand.get('category')}."
        ),
        metadata={"section": "brand"},
    ))

    for product in data.get("products", []):
        benefits = ", ".join(product.get("key_benefits", []))
        docs.append(Document(
            page_content=(
                f"Product: {product.get('name')}. "
                f"{product.get('description')} "
                f"Key benefits: {benefits}. "
                f"Pricing: {product.get('pricing')}."
            ),
            metadata={"section": "product"},
        ))

    audience = data.get("target_audience", {})
    docs.append(Document(
        page_content=(
            f"Target audience: {audience.get('primary')}. "
            f"Industries: {', '.join(audience.get('industries', []))}. "
            f"Pain points: {', '.join(audience.get('pain_points', []))}. "
        ),
        metadata={"section": "audience"}
    ))

    tone = data.get("tone_guidelines", {})
    docs.append(Document(
        page_content=(
            f"Brand voice: {tone.get('voice')}. "
            f"Do: {', '.join(tone.get('do', []))}. "
            f"Avoid: {', '.join(tone.get('avoid', []))}. "
        ),
        metadata={"section": "tone"}
    ))

    claims = data.get("approved_claims", [])
    docs.append(Document(
        page_content=f"Approved claims you can use: {'. '.join(claims)}.",
        metadata={"section": "claims"}
    ))

    forbidden = data.get("forbidden_phrases", [])
    docs.append(Document(
        page_content=f"Forbidden phrases - never use these: {'. '.join(forbidden)}.",
        metadata={"section": "forbidden"}
    ))

    for campaign in data.get("past_campaigns", []):
        docs.append(Document(
            page_content=(
                f"Past campaign: '{campaign.get('title')}': "
                f"{campaign.get('message')}. "
                f"Result: {campaign.get('result')}. "
            ),
            metadata={"section": "past_campaigns"}
        ))

    return docs


def build_vector_store() -> FAISS:
    log.debug("Building brand RAG vector store from company_data.json")
    data = load_company_data()
    docs = chunk_company_data(data)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    index = FAISS.from_documents(docs, embeddings)
    log.debug(f"Brand RAG index ready — {len(docs)} documents")
    return index


# Build once when the module is first imported
_vector_store = build_vector_store()
_retriever = _vector_store.as_retriever(search_kwargs={"k": 3})


def retrieve_brand_context(query: str) -> str:
    log.debug(f"retrieve_brand_context — query='{query[:80]}'")
    docs = _retriever.invoke(query)
    if not docs:
        return ""
    return "\n".join(doc.page_content for doc in docs)

