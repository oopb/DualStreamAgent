from dualstream_agent.harness.visualclaw.schemas import VisualClawRound


def test_official_round_meta_is_normalized() -> None:
    item = VisualClawRound.from_dict(
        {
            "id": "q2",
            "type": "exec_check",
            "question": "Write a JSON file.",
            "update_ids": [],
            "eval": {
                "command": "python ${eval_dir}/${agent_id}/scripts/check.py ${workspace}",
                "expect_exit": 0,
                "timeout": 30,
            },
            "feedback": {"correct": "ok", "incorrect": "bad"},
            "meta": {
                "round": 2,
                "required_modalities": ["video", "text"],
                "required_skills": ["source-evaluation"],
                "anti_skills": ["guessing"],
                "expected_sources": ["clip.mp4", "workspace/input.pdf"],
                "tags": ["cross-modal"],
                "video_required": True,
                "evidence_type": "visual_required",
            },
        }
    )
    assert item.round_number == 2
    assert item.required_modalities == ["video", "text"]
    assert item.required_skills == ["source-evaluation"]
    assert item.anti_skills == ["guessing"]
    assert item.expected_sources == ["clip.mp4", "workspace/input.pdf"]
    assert item.tags == ["cross-modal"]
    assert item.video_required is True
    assert item.evidence_type == "visual_required"
