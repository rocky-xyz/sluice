from sluice.agents.cache import CallCache


def test_stored_answer_survives_reopening(tmp_path):
    path = tmp_path / "nested" / "llm.db"
    with CallCache(path) as cache:
        assert cache.get("k") is None
        cache.put("k", "request", "answer")
    with CallCache(path) as cache:
        assert cache.get("k") == "answer"
        assert len(cache) == 1


def test_first_answer_is_never_replaced(tmp_path):
    with CallCache(tmp_path / "llm.db") as cache:
        cache.put("k", "request", "first")
        cache.put("k", "request", "second")
        assert cache.get("k") == "first"
        assert len(cache) == 1
