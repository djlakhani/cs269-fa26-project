from __future__ import annotations

from typing import List, Optional

from .retrieval.embedding import BaseRetriever, build_retriever
from .llm_layer.client import LLMClient
from .memory.models import (
    BucketTrackRef,
    SessionChunk,
    SessionDelta,
)
from .memory.schema import schema_prompt_text
from .core.config import PipelineThresholds, TraceConfig
from .core.engine_utils import EngineUtilityMixin
from .core.sample_runner import SampleRunnerMixin
from .store_layer import ProfileStore
from .write.chunker import ChunkerMixin
from .write.delta_extractor import DeltaExtractorMixin
from .write.update_resolver import UpdateResolverMixin
from .write.invalidation_resolver import InvalidationResolverMixin
from .write.stale_linker import StaleLinkerMixin
from .write.writer import SessionWriterMixin
from .query.action_grounding import ActionGroundingMixin
from .query.engine import QueryEngineMixin
from .query.retrieval import QueryRetrievalMixin
from .query.premise_verifier import VerifierBasisMixin


class CupMemEngine(EngineUtilityMixin, ChunkerMixin, DeltaExtractorMixin, UpdateResolverMixin, InvalidationResolverMixin, StaleLinkerMixin, SessionWriterMixin, VerifierBasisMixin, ActionGroundingMixin, QueryRetrievalMixin, QueryEngineMixin, SampleRunnerMixin):
    def __init__(
        self,
        *,
        llm: LLMClient,
        embedding_model_path: Optional[str] = None,
        embedding_device: str = "cpu",
        thresholds: Optional[PipelineThresholds] = None,
        trace_config: Optional[TraceConfig] = None,
    ):
        self.llm = llm
        self.thresholds = thresholds or PipelineThresholds()
        self.trace_config = trace_config or TraceConfig()
        if not embedding_model_path:
            raise ValueError(
                "embedding_model_path is required. Please pass a local sentence-embedding model directory "
                "(for example, a local all-MiniLM-L6-v2 checkpoint)."
            )
        self.store = ProfileStore()
        self.embedding: BaseRetriever = build_retriever(
            embedding_model_path,
            device=embedding_device,
        )
        self.chunk_bank: List[SessionChunk] = []
        self.delta_store: List[SessionDelta] = []
        self._chunk_counter = 0
        self._delta_counter = 0
        self._proposal_counter = 0
        self.schema_text = schema_prompt_text()

    def reset(self) -> None:
        self.store.reset()
        self.chunk_bank = []
        self.delta_store = []
        self._chunk_counter = 0
        self._delta_counter = 0
        self._proposal_counter = 0

    def export_state(self) -> dict:
        """Export all mutable engine state needed for deterministic resume."""
        return {
            "schema_version": 1,
            "store": self.store.export_state(),
            "chunk_bank": [chunk.to_dict() for chunk in self.chunk_bank],
            "delta_store": [delta.to_dict() for delta in self.delta_store],
            "counters": {
                "chunk": self._chunk_counter,
                "delta": self._delta_counter,
                "proposal": self._proposal_counter,
            },
        }

    def restore_state(self, state: dict) -> None:
        """Restore an engine checkpoint created by :meth:`export_state`."""
        if int(state.get("schema_version", 0)) != 1:
            raise ValueError("Unsupported CupMemEngine state schema")
        self.store.restore_state(state.get("store", {}))
        self.chunk_bank = []
        for raw in state.get("chunk_bank", []):
            payload = dict(raw)
            payload["candidate_bucket_tracks"] = [
                BucketTrackRef(**ref)
                for ref in payload.get("candidate_bucket_tracks", [])
            ]
            self.chunk_bank.append(SessionChunk(**payload))
        self.delta_store = [
            SessionDelta(**raw) for raw in state.get("delta_store", [])
        ]
        counters = state.get("counters", {})
        self._chunk_counter = int(counters.get("chunk", len(self.chunk_bank)))
        self._delta_counter = int(counters.get("delta", len(self.delta_store)))
        self._proposal_counter = int(counters.get("proposal", 0))

    def _next_chunk_id(self) -> str:
        self._chunk_counter += 1
        return f"c_{self._chunk_counter:05d}"

    def _next_delta_id(self) -> str:
        self._delta_counter += 1
        return f"d_{self._delta_counter:05d}"

    def _next_proposal_id(self) -> str:
        self._proposal_counter += 1
        return f"lip_{self._proposal_counter:05d}"
