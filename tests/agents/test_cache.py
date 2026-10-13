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


def test_threads_can_share_one_cache(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    with CallCache(tmp_path / "llm.db") as cache:

        def work(n: int) -> str | None:
            cache.put(f"k{n}", "request", f"answer {n}")
            return cache.get(f"k{n}")

        with ThreadPoolExecutor(max_workers=8) as pool:
            assert list(pool.map(work, range(200))) == [f"answer {n}" for n in range(200)]
        assert len(cache) == 200
