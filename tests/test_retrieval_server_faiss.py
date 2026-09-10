import pytest
from fastapi import HTTPException
from types import SimpleNamespace

from examples.search.retriever import retrieval_server


def test_require_faiss_gpu_support_rejects_cpu_build(monkeypatch):
    monkeypatch.delattr(retrieval_server.faiss, "GpuMultipleClonerOptions", raising=False)
    monkeypatch.delattr(retrieval_server.faiss, "StandardGpuResources", raising=False)
    monkeypatch.setattr(retrieval_server.faiss, "get_num_gpus", lambda: 0)

    with pytest.raises(RuntimeError, match="GPU-enabled FAISS"):
        retrieval_server.require_faiss_gpu_support(expected_gpus=8)


def test_require_faiss_gpu_support_accepts_requested_gpu_count(monkeypatch):
    monkeypatch.setattr(retrieval_server.faiss, "GpuMultipleClonerOptions", object, raising=False)
    monkeypatch.setattr(retrieval_server.faiss, "StandardGpuResources", object, raising=False)
    monkeypatch.setattr(retrieval_server.faiss, "index_cpu_to_all_gpus", object, raising=False)
    monkeypatch.setattr(retrieval_server.faiss, "get_num_gpus", lambda: 8)

    assert retrieval_server.require_faiss_gpu_support(expected_gpus=8) == 8


@pytest.mark.parametrize(
    "query_request",
    [
        retrieval_server.QueryRequest(query="q", queries=["q"]),
        retrieval_server.QueryRequest(query=""),
        retrieval_server.QueryRequest(queries=[]),
        retrieval_server.QueryRequest(query="q", topk=0),
        retrieval_server.QueryRequest(query="q", topk=101),
    ],
)
def test_retriever_request_validation_rejects_ambiguous_or_unbounded_inputs(monkeypatch, query_request):
    monkeypatch.setattr(retrieval_server, "config", SimpleNamespace(retrieval_topk=3), raising=False)
    with pytest.raises(HTTPException):
        retrieval_server._validated_request(query_request)


def test_retriever_request_validation_normalizes_single_query(monkeypatch):
    monkeypatch.setattr(retrieval_server, "config", SimpleNamespace(retrieval_topk=3), raising=False)
    queries, topk, is_batch = retrieval_server._validated_request(
        retrieval_server.QueryRequest(query="  Hamlet author  ", return_scores=True)
    )
    assert queries == ["Hamlet author"]
    assert topk == 3
    assert is_batch is False


def test_load_docs_rejects_faiss_negative_sentinel():
    with pytest.raises(RuntimeError, match="out-of-range"):
        retrieval_server.load_docs([{"id": 0}], [-1])
