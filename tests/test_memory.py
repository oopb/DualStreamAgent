from dualstream_agent.memory.store import MemoryItem, MemoryStore


def test_memory_persists_and_retrieves(tmp_path):
    path = tmp_path / "memory.sqlite3"
    store = MemoryStore(str(path))
    store.add(MemoryItem(content="The red cup moved to the table", tags=["cup"]))
    matches = store.search("red cup", top_k=2)
    assert len(matches) == 1
    assert "red cup" in matches[0].content.lower()
    store.close()
