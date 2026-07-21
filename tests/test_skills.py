from dualstream_agent.skills.manager import SkillManager


def test_skill_roundtrip_and_retrieval(tmp_path):
    manager = SkillManager(str(tmp_path))
    manager.write_candidate(
        name="verify temporal order",
        description="Verify event ordering in video",
        body="Check timestamps before answering.",
        tags=["temporal"],
    )
    candidate = manager.read_skill("verify-temporal-order")
    assert candidate.status == "candidate"
    manager.promote("verify-temporal-order", {"score": 1.0})
    matches = manager.retrieve("temporal video", top_k=1)
    assert matches[0].name == "verify-temporal-order"
    assert matches[0].status == "active"
