import pytest

from dualstream_agent.skills.manager import SkillManager, SkillStatus


def test_candidate_is_isolated_until_validated_and_promoted(tmp_path):
    manager = SkillManager(tmp_path)
    candidate_path = manager.write_candidate_skill(
        name="verify temporal order",
        description="Verify event ordering in video",
        body="Check timestamps before answering.",
        tags=["temporal"],
        metadata={"status": "active", "version": 99},
    )

    candidate = manager.read_skill(
        "verify-temporal-order",
        status=SkillStatus.CANDIDATE,
    )
    assert candidate_path == tmp_path / ".candidates/verify-temporal-order/SKILL.md"
    assert candidate.status == SkillStatus.CANDIDATE
    assert candidate.version == 1
    assert manager.list_skills() == []
    assert manager.retrieve_skills("temporal video") == []

    with pytest.raises(ValueError, match="successful validation"):
        manager.promote_skill("verify-temporal-order", {"promoted": False})

    active_path = manager.promote_skill(
        "verify-temporal-order",
        {"promoted": True, "utility": 0.2},
    )
    assert active_path == tmp_path / "verify-temporal-order/SKILL.md"
    assert not candidate_path.exists()
    matches = manager.retrieve_skills("temporal video", top_k=1)
    assert [skill.name for skill in matches] == ["verify-temporal-order"]
    assert matches[0].status == SkillStatus.ACTIVE
    assert manager.recent_retrievals(1)[0].skill_names == ["verify-temporal-order"]


def test_new_candidate_does_not_overwrite_active_skill(tmp_path):
    manager = SkillManager(tmp_path)
    manager.write_candidate_skill(
        name="verify",
        description="First version",
        body="Original procedure.",
    )
    manager.promote_skill("verify", {"promoted": True})

    manager.write_candidate_skill(
        name="verify",
        description="Second version",
        body="Candidate procedure.",
    )

    active = manager.read_skill("verify", status=SkillStatus.ACTIVE)
    candidate = manager.read_skill("verify", status=SkillStatus.CANDIDATE)
    assert active.version == 1
    assert active.body == "Original procedure."
    assert candidate.version == 2
    assert candidate.body == "Candidate procedure."


def test_disabled_skill_is_not_retrieved(tmp_path):
    manager = SkillManager(tmp_path)
    manager.write_candidate_skill(
        name="observe scene",
        description="Observe visual changes",
        body="Compare causal frames.",
        tags=["visual"],
    )
    manager.promote_skill("observe-scene", {"promoted": True})

    manager.disable_skill("observe-scene", reason="regression detected")

    disabled = manager.read_skill("observe-scene", status=SkillStatus.DISABLED)
    assert disabled.metadata["disabled_reason"] == "regression detected"
    assert manager.list_skills() == []
    assert manager.retrieve_skills("visual") == []


def test_retrieval_log_is_bounded(tmp_path):
    manager = SkillManager(tmp_path, retrieval_log_size=2)
    for query in ("one", "two", "three"):
        manager.retrieve_skills(query)

    assert [record.query for record in manager.recent_retrievals()] == ["two", "three"]
