import pytest

from dualstream_agent.evolution.validator import ReplayValidator
from dualstream_agent.skills.manager import SkillManager, SkillStatus


def _candidate(manager: SkillManager, name: str) -> None:
    manager.write_candidate_skill(
        name=name,
        description="Test candidate",
        body="Follow a testable procedure.",
    )


@pytest.mark.asyncio
async def test_validator_promotes_only_when_utility_passes(tmp_path):
    manager = SkillManager(tmp_path)
    _candidate(manager, "useful")

    async def evaluate(candidate):
        if candidate is None:
            return {"score": 0.5, "latency_s": 1.0, "tokens": 100}
        return {
            "score": 0.8,
            "latency_s": 1.5,
            "tokens": 110,
            "regression_rate": 0.0,
        }

    result = await ReplayValidator(manager).validate("useful", evaluate)

    assert result.promoted is True
    assert result.token_delta == 10
    active = manager.read_skill("useful", status=SkillStatus.ACTIVE)
    assert active.metadata["validation"]["utility"] == pytest.approx(result.utility)


@pytest.mark.asyncio
async def test_validator_keeps_rejected_candidate_isolated(tmp_path):
    manager = SkillManager(tmp_path)
    _candidate(manager, "regression")

    async def evaluate(candidate):
        if candidate is None:
            return {"score": 0.8, "latency_s": 1.0, "tokens": 100}
        return {
            "score": 0.7,
            "latency_s": 2.0,
            "tokens": 200,
            "regression_rate": 0.2,
        }

    result = await ReplayValidator(manager).validate("regression", evaluate)

    assert result.promoted is False
    candidate = manager.read_skill("regression", status=SkillStatus.CANDIDATE)
    assert candidate.metadata["validation"]["promoted"] is False
    assert manager.list_skills() == []
