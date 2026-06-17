from __future__ import annotations

import json
import pickle
import shutil
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from services.logger import get_logger

if TYPE_CHECKING:
    pass

log = get_logger(__name__)

# =============================================================================
# embedder.py
#
# Builds and queries two FAISS indexes for the domain Training Data:
#
#   approved_index  — approved workflows and email examples used for
#                     Top-K few-shot retrieval in the Generator system prompt
#   rejected_index  — rejected examples with rejection_reason labels used
#                     for negative-example awareness and the self-learning loop
#
# WHAT THIS REPLACES
#   The FAISSRetriever stub in domain_pack.py currently delegates directly to
#   the existing campaign_memory.py functions (get_few_shot_examples,
#   get_denial_lessons_for_agent). Those functions are useful but have two
#   limitations for the Agent Builder:
#     1. They only index campaigns/ — not the domain Training Data folder
#     2. They have no task_type filter — search returns results across all
#        task types regardless of what the current workflow needs
#
#   embedder.py adds:
#     - Indexing of domains/{domain}/training_data/ in addition to campaigns/
#     - task_type metadata on every indexed document for filtered retrieval
#     - PyTorch sentence-transformers as the embedding model (offline, no API cost)
#     - Automatic re-indexing after 5 new rejections (self-learning loop KPI)
#
# TECHNOLOGY
#   sentence-transformers  — PyTorch model paraphrase-multilingual-MiniLM-L12-v2
#                            produces 384-dim embeddings. Runs fully offline.
#                            No OpenAI API call required for indexing.
#   FAISS (faiss-cpu)      — IndexFlatIP for cosine similarity search.
#                            Already installed as a LangChain dependency.
#   numpy                  — normalises vectors for cosine similarity.
#   pickle                 — stores document metadata alongside the FAISS index.
#
# WHY sentence-transformers OVER OpenAIEmbeddings
#   The existing campaign_memory.py uses OpenAIEmbeddings (text-embedding-3-small).
#   That works well but costs money per embedding call and requires internet.
#   paraphrase-multilingual-MiniLM-L12-v2 is free, offline, and produces
#   384-dim embeddings that are fast to search with FAISS IndexFlatIP.
#   campaign_memory.py is left unchanged — it continues to use OpenAIEmbeddings
#   for the main campaign index. embedder.py is additive, not a replacement.
# =============================================================================

# ── Constants ─────────────────────────────────────────────────────────────────

# Number of new rejections that trigger an automatic FAISS re-index.
# Matches the self-learning KPI from the Domain Selection Brief.
REINDEX_THRESHOLD = 5

# ── Document dataclass ────────────────────────────────────────────────────────

@dataclass
class TrainingDoc:
    """
    A single document in the Training Data index.
    Stores both the text content (for embedding) and metadata (for filtering).
    """
    text:            str
    task_type:       str            # e.g. "email_generation"
    status:          str            # "approved" or "rejected"
    rejection_reason: str = ""     # only set for rejected examples
    source_file:     str = ""      # original filename for traceability
    metadata:        dict = field(default_factory=dict)


# ── FAISSRetriever ────────────────────────────────────────────────────────────

class FAISSRetriever:
    """
    Replaces the FAISSRetriever stub in domain_pack.py.

    Builds and queries two FAISS indexes from the domain's training_data/
    folder plus the existing campaigns/ folder.

    Usage (called by DomainPack.load() in domain_pack.py):
        retriever = FAISSRetriever(cfg["training_data"], domain_folder)
        examples  = retriever.get_top_k(nl_input, k=3, task_type="email_generation")
        lessons   = retriever.get_denial_lessons(nl_input, k=2)
    """

    def __init__(self, training_data_cfg: dict, domain_folder: Path):
        self._approved_path  = domain_folder / training_data_cfg["approved"]
        self._rejected_path  = domain_folder / training_data_cfg["rejected"]
        self._embed_model_id = training_data_cfg.get(
            "embed_model", "paraphrase-multilingual-MiniLM-L12-v2"
        )

        # Index storage: memory/domain_index/{approved,rejected}/
        self._index_root = Path("memory") / "domain_index"
        self._approved_index_dir = self._index_root / "approved"
        self._rejected_index_dir = self._index_root / "rejected"

        # New rejection counter for self-learning re-index trigger
        self._new_rejections_count = 0

        # Lazy-loaded — built on first query, not at import time
        self._model        = None
        self._approved_idx = None   # FAISS index object
        self._approved_docs: list[TrainingDoc] = []
        self._rejected_idx = None
        self._rejected_docs: list[TrainingDoc] = []

        log.debug(
            f"FAISSRetriever initialised — model: {self._embed_model_id} | "
            f"approved: {self._approved_path} | rejected: {self._rejected_path}"
        )

    # ── Public API ─────────────────────────────────────────────────────────

    def get_top_k(
        self,
        nl_input:  str,
        k:         int = 3,
        task_type: str | None = None,
    ) -> str:
        """
        Return the k most similar approved examples as a formatted string
        for few-shot injection into the Generator system prompt.

        Parameters
        ----------
        nl_input  : the user's NL prompt
        k         : number of examples to return
        task_type : if set, only returns examples of this task type.
                    Falls back to campaign_memory if domain index is empty.
        """
        self._ensure_loaded()

        if not self._approved_docs:
            # Domain Training Data not populated yet — fall back to
            # existing campaign_memory for zero-disruption behaviour
            log.debug("Domain approved index empty — falling back to campaign_memory")
            from services.campaign_memory import get_few_shot_examples
            return get_few_shot_examples(nl_input, k=k)

        results = self._search(
            query     = nl_input,
            index     = self._approved_idx,
            docs      = self._approved_docs,
            k         = k,
            task_type = task_type,
        )

        if not results:
            from services.campaign_memory import get_few_shot_examples
            return get_few_shot_examples(nl_input, k=k)

        return self._format_approved(results)

    def get_denial_lessons(
        self,
        nl_input:  str,
        k:         int = 2,
        task_type: str | None = None,
    ) -> str:
        """
        Return k rejected examples and their rejection reasons as a
        formatted string for the Generator and agent system prompts.
        """
        self._ensure_loaded()

        if not self._rejected_docs:
            log.debug("Domain rejected index empty — falling back to campaign_memory")
            from services.campaign_memory import get_denial_lessons_for_agent
            return get_denial_lessons_for_agent(nl_input, k=k)

        results = self._search(
            query     = nl_input,
            index     = self._rejected_idx,
            docs      = self._rejected_docs,
            k         = k,
            task_type = task_type,
        )

        if not results:
            from services.campaign_memory import get_denial_lessons_for_agent
            return get_denial_lessons_for_agent(nl_input, k=k)

        return self._format_rejected(results)

    def add_rejection(
        self,
        text:             str,
        task_type:        str,
        rejection_reason: str,
        source_file:      str = "",
    ) -> None:
        """
        Add one new rejected example to the rejected index.
        Called by the HITL rejection handler (Phase 4b) when a human
        reviewer rejects an agent output.

        Automatically triggers a full re-index when REINDEX_THRESHOLD
        new rejections have accumulated — the self-learning KPI.
        """
        self._ensure_loaded()

        doc = TrainingDoc(
            text             = text,
            task_type        = task_type,
            status           = "rejected",
            rejection_reason = rejection_reason,
            source_file      = source_file,
        )

        vector = self._embed([doc.text])

        if self._rejected_idx is None:
            import faiss
            import numpy as np
            dim = vector.shape[1]
            self._rejected_idx = faiss.IndexFlatIP(dim)
            self._rejected_idx.add(vector)
        else:
            self._rejected_idx.add(vector)

        self._rejected_docs.append(doc)
        self._save_index(
            self._rejected_idx, self._rejected_docs, self._rejected_index_dir
        )

        self._new_rejections_count += 1
        log.debug(
            f"Added rejection to index — task_type: {task_type} | "
            f"new rejections since last reindex: {self._new_rejections_count}"
        )

        # Self-learning trigger: re-index when threshold is reached
        if self._new_rejections_count >= REINDEX_THRESHOLD:
            log.debug(
                f"Self-learning threshold reached ({REINDEX_THRESHOLD} rejections) "
                f"— triggering full re-index"
            )
            self.rebuild()
            self._new_rejections_count = 0

    def rebuild(self) -> None:
        """
        Force a full re-index of both approved and rejected Training Data.
        Called automatically by add_rejection() at the threshold, or
        manually after bulk-loading new training examples.
        """
        log.debug("FAISSRetriever: starting full re-index")

        # Clear cached state
        self._approved_idx  = None
        self._approved_docs = []
        self._rejected_idx  = None
        self._rejected_docs = []

        # Clear saved indexes from disk
        for d in (self._approved_index_dir, self._rejected_index_dir):
            if d.exists():
                shutil.rmtree(d)

        # Reload from source files
        self._load_training_data()
        self._load_campaigns()
        self._build_indexes()
        log.debug("FAISSRetriever: re-index complete")

    # ── Internal: loading ──────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Load indexes from disk (or build from scratch) on first query."""
        if self._approved_idx is not None:
            return  # already loaded

        loaded = self._try_load_from_disk()
        if not loaded:
            log.debug("No saved domain index found — building from training data")
            self._load_training_data()
            self._load_campaigns()
            self._build_indexes()

    def _try_load_from_disk(self) -> bool:
        """
        Attempt to load pre-built FAISS indexes from disk.
        Returns True if both indexes loaded successfully.
        """
        try:
            import faiss
            a_idx_path  = self._approved_index_dir / "index.faiss"
            a_docs_path = self._approved_index_dir / "docs.pkl"
            r_idx_path  = self._rejected_index_dir / "index.faiss"
            r_docs_path = self._rejected_index_dir / "docs.pkl"

            if not (a_idx_path.exists() and a_docs_path.exists()):
                return False

            self._approved_idx = faiss.read_index(str(a_idx_path))
            with open(a_docs_path, "rb") as f:
                self._approved_docs = pickle.load(f)

            if r_idx_path.exists() and r_docs_path.exists():
                self._rejected_idx = faiss.read_index(str(r_idx_path))
                with open(r_docs_path, "rb") as f:
                    self._rejected_docs = pickle.load(f)

            log.debug(
                f"Domain index loaded from disk — "
                f"{len(self._approved_docs)} approved, "
                f"{len(self._rejected_docs)} rejected"
            )
            return True

        except Exception as e:
            log.warning(f"Could not load domain index from disk: {e}")
            return False

    def _load_training_data(self) -> None:
        """
        Load documents from domains/{domain}/training_data/approved/ and
        rejected/ folders. Supports .txt, .md, .json, and .yaml files.
        """
        self._approved_docs += self._read_folder(
            self._approved_path, status="approved"
        )
        self._rejected_docs += self._read_folder(
            self._rejected_path, status="rejected"
        )
        log.debug(
            f"Training data loaded — "
            f"{len(self._approved_docs)} approved, "
            f"{len(self._rejected_docs)} rejected"
        )

    def _load_campaigns(self) -> None:
        """
        Also index the existing campaigns/ folder so the domain retriever
        has access to real past run history from the start.
        Campaigns are tagged with task_type='email_generation' as default
        since all existing campaigns are marketing posts.
        """
        campaigns_dir = Path("campaigns")
        if not campaigns_dir.exists():
            return

        for path in sorted(campaigns_dir.glob("*.json")):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            status  = data.get("status", "unknown")
            content = data.get("content", data.get("full_post", ""))
            prompt  = data.get("user_prompt", "")

            if not content and not prompt:
                continue

            text = f"User wanted: {prompt}\nContent: {content[:300]}"

            doc = TrainingDoc(
                text             = text,
                task_type        = "email_generation",  # existing campaigns default
                status           = status,
                rejection_reason = data.get("denial_reason", ""),
                source_file      = str(path),
            )

            if status in ("posted", "approved"):
                self._approved_docs.append(doc)
            elif status == "denied":
                self._rejected_docs.append(doc)

        log.debug(
            f"Campaigns indexed — "
            f"{sum(1 for d in self._approved_docs if 'campaigns' in d.source_file)} "
            f"approved, "
            f"{sum(1 for d in self._rejected_docs if 'campaigns' in d.source_file)} "
            f"rejected from campaigns/"
        )

    def _read_folder(self, folder: Path, status: str) -> list[TrainingDoc]:
        """
        Read all supported files from a training data folder.
        Each file becomes one TrainingDoc. task_type is inferred from the
        file name if it contains a known task type keyword, otherwise
        defaults to 'email_generation'.
        """
        docs = []
        if not folder.exists():
            return docs

        for path in sorted(folder.iterdir()):
            if path.suffix not in {".txt", ".md", ".json", ".yaml", ".yml"}:
                continue
            try:
                text, rejection_reason, task_type = self._parse_file(path)
                if not text.strip():
                    continue
                docs.append(TrainingDoc(
                    text             = text,
                    task_type        = task_type,
                    status           = status,
                    rejection_reason = rejection_reason,
                    source_file      = str(path),
                ))
            except Exception as e:
                log.warning(f"Could not read training file {path}: {e}")

        return docs

    def _parse_file(self, path: Path) -> tuple[str, str, str]:
        """
        Parse a single training data file.
        Returns (text, rejection_reason, task_type).

        JSON files may have a rejection_reason field and a task_type field.
        Text/Markdown files are read as plain text.
        task_type is inferred from filename keywords if not set in the file.
        """
        rejection_reason = ""
        task_type        = self._infer_task_type(path.stem)

        if path.suffix == ".json":
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            # Support both flat strings and structured objects
            if isinstance(data, str):
                text = data
            else:
                text = (
                    data.get("text")
                    or data.get("content")
                    or data.get("body")
                    or json.dumps(data, indent=2)
                )
                rejection_reason = data.get("rejection_reason", "")
                task_type = data.get("task_type", task_type)

        elif path.suffix in {".yaml", ".yml"}:
            import yaml
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                text = (
                    data.get("text")
                    or data.get("content")
                    or yaml.dump(data)
                )
                rejection_reason = data.get("rejection_reason", "")
                task_type = data.get("task_type", task_type)
            else:
                text = str(data)

        else:
            text = path.read_text(encoding="utf-8")

        return text, rejection_reason, task_type

    @staticmethod
    def _infer_task_type(filename_stem: str) -> str:
        """
        Infer task_type from filename keywords.
        e.g. 'cold_email_example_01' → 'email_generation'
             'competitor_analysis_saas' → 'competitor_analysis'
        Falls back to 'email_generation' as the domain default.
        """
        stem = filename_stem.lower()
        if any(k in stem for k in ("email", "cold", "nurture", "outreach", "sequence")):
            return "email_generation"
        if any(k in stem for k in ("research", "summary", "news")):
            return "research_summary"
        if any(k in stem for k in ("competitor", "competitive", "analysis")):
            return "competitor_analysis"
        if any(k in stem for k in ("brief", "campaign", "plan")):
            return "campaign_brief"
        return "email_generation"

    # ── Internal: indexing ─────────────────────────────────────────────────

    def _build_indexes(self) -> None:
        """
        Embed all loaded documents and build FAISS IndexFlatIP indexes.
        Saves both indexes to disk for fast reload on next startup.
        """
        import faiss
        import numpy as np

        if self._approved_docs:
            texts = [d.text for d in self._approved_docs]
            vecs  = self._embed(texts)
            self._approved_idx = faiss.IndexFlatIP(vecs.shape[1])
            self._approved_idx.add(vecs)
            self._save_index(
                self._approved_idx, self._approved_docs, self._approved_index_dir
            )
            log.debug(f"Approved index built — {len(self._approved_docs)} docs")

        if self._rejected_docs:
            texts = [d.text for d in self._rejected_docs]
            vecs  = self._embed(texts)
            self._rejected_idx = faiss.IndexFlatIP(vecs.shape[1])
            self._rejected_idx.add(vecs)
            self._save_index(
                self._rejected_idx, self._rejected_docs, self._rejected_index_dir
            )
            log.debug(f"Rejected index built — {len(self._rejected_docs)} docs")

    def _embed(self, texts: list[str]):
        """
        Embed a list of texts using sentence-transformers.
        Returns a float32 numpy array normalised for cosine similarity.

        Lazy-loads the model on first call — model download happens once
        and is cached by sentence-transformers in ~/.cache/huggingface/.
        """
        import numpy as np
        from sentence_transformers import SentenceTransformer

        if self._model is None:
            log.debug(f"Loading embedding model: {self._embed_model_id}")
            self._model = SentenceTransformer(self._embed_model_id)
            log.debug("Embedding model loaded")

        vecs = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        # Normalise to unit vectors so IndexFlatIP computes cosine similarity
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # avoid divide-by-zero
        return (vecs / norms).astype("float32")

    # ── Internal: search ──────────────────────────────────────────────────

    def _search(
        self,
        query:     str,
        index,
        docs:      list[TrainingDoc],
        k:         int,
        task_type: str | None,
    ) -> list[TrainingDoc]:
        """
        Run a FAISS similarity search and return the top-k matching docs,
        optionally filtered by task_type.
        """
        if index is None or not docs:
            return []

        import numpy as np

        query_vec = self._embed([query])

        # Fetch more than k to leave room for task_type filtering
        fetch_k = k * 4 if task_type else k
        fetch_k = min(fetch_k, len(docs))

        scores, indices = index.search(query_vec, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(docs):
                continue
            doc = docs[idx]
            if task_type and doc.task_type != task_type:
                continue
            results.append(doc)
            if len(results) >= k:
                break

        return results

    # ── Internal: formatting ──────────────────────────────────────────────

    @staticmethod
    def _format_approved(docs: list[TrainingDoc]) -> str:
        """Format approved examples for injection into the Generator prompt."""
        lines = ["=== Approved examples (match tone and structure) ==="]
        for i, doc in enumerate(docs, 1):
            lines.append(f"\n[Example {i}] (task_type: {doc.task_type})")
            lines.append(doc.text[:400])
        lines.append("\n=== End of examples ===")
        return "\n".join(lines)

    @staticmethod
    def _format_rejected(docs: list[TrainingDoc]) -> str:
        """Format rejected examples for injection into the Generator prompt."""
        lines = ["=== Rejected examples — avoid these patterns ==="]
        for i, doc in enumerate(docs, 1):
            lines.append(f"\n[Rejected {i}] (task_type: {doc.task_type})")
            if doc.rejection_reason:
                lines.append(f"Why rejected: {doc.rejection_reason}")
            lines.append(doc.text[:300])
        lines.append("\n=== End of rejection lessons ===")
        return "\n".join(lines)

    # ── Internal: persistence ─────────────────────────────────────────────

    @staticmethod
    def _save_index(index, docs: list[TrainingDoc], folder: Path) -> None:
        """Save FAISS index and doc metadata to disk."""
        import faiss
        folder.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(folder / "index.faiss"))
        with open(folder / "docs.pkl", "wb") as f:
            pickle.dump(docs, f)