from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dualstream_agent.backends.base import ModelBackend
from dualstream_agent.schemas import GenerationRequest
from dualstream_agent.skills.manager import SkillManager
from dualstream_agent.utils import extract_json_object


@dataclass(slots=True)
class SkillProposal:
    name: str
    description: str
    body: str
    tags: list[str]
    source_trace_ids: list[str]


class SkillEvolver:
    """Generate candidate procedural knowledge from failed traces.

    Candidates are written with status=candidate and must pass a validator before
    promotion. The online runtime never promotes a skill automatically.
    """

    def __init__(self, backend: ModelBackend, skills: SkillManager):
        self.backend = backend
        self.skills = skills

    async def propose(self, failures: list[dict[str, Any]]) -> SkillProposal:
        compact = []
        for failure in failures[:12]:
            compact.append(
                {
                    "trace_id": failure.get("trace_id", ""),
                    "input": failure.get("text") or failure.get("user_goal") or "",
                    "signal": failure.get("signal", {}),
                    "decision": failure.get("decision", {}),
                    "response": failure.get("response", ""),
                    "failure": failure.get("failure", failure.get("error", "")),
                }
            )
        prompt = f"""
Analyze the failed agent traces below and propose one reusable procedural skill.
The skill must generalize beyond a single example and must not contain private or
scenario-specific identifiers. Return JSON with name, description, body, tags.
The body should contain applicability, procedure, verification, and anti-patterns.

Failures:
{compact}
""".strip()
        result = await self.backend.generate(
            GenerationRequest(
                messages=[
                    {
                        "role": "system",
                        "content": "You improve an agent by writing compact, testable procedural skills.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_new_tokens=900,
                temperature=0.2,
            )
        )
        data = extract_json_object(result.text)
        proposal = SkillProposal(
            name=str(data.get("name", "candidate-skill")),
            description=str(data.get("description", "Candidate generated from failures.")),
            body=str(data.get("body", result.text)),
            tags=[str(tag) for tag in data.get("tags", [])],
            source_trace_ids=[str(item.get("trace_id", "")) for item in compact],
        )
        self.skills.write_candidate_skill(
            name=proposal.name,
            description=proposal.description,
            body=proposal.body,
            tags=proposal.tags,
            metadata={"source_trace_ids": proposal.source_trace_ids},
        )
        return proposal
