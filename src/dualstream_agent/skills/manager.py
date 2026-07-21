from pathlib import Path


class SkillManager:
    def __init__(self, root="skills"):
        self.root = Path(root)

    def list_skills(self):
        if not self.root.exists():
            return []
        return [p.name for p in self.root.iterdir() if p.is_dir()]

    def read_skill(self, name):
        path = self.root / name / "SKILL.md"
        return path.read_text(encoding="utf-8")

    def retrieve(self, query, top_k=3):
        # Initial implementation: keyword retrieval.
        # Embedding retrieval will be added later.
        skills = self.list_skills()
        q = query.lower()
        return [s for s in skills if q in s.lower()][:top_k]
