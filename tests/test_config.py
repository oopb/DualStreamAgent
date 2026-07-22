import pytest
from pydantic import ValidationError

from dualstream_agent.config import ControllerConfig, PerceptionConfig, SkillConfig


@pytest.mark.parametrize(
    ("config_type", "values"),
    [
        (PerceptionConfig, {"buffer_size": 0}),
        (PerceptionConfig, {"change_threshold": 1.1}),
        (ControllerConfig, {"confidence_threshold": -0.1}),
        (ControllerConfig, {"cooldown_s": -1.0}),
        (SkillConfig, {"top_k": 0}),
    ],
)
def test_runtime_config_rejects_invalid_bounds(config_type, values):
    with pytest.raises(ValidationError):
        config_type(**values)
