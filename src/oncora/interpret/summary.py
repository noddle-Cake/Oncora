"""Evidence summary: claims grouped by section, each citing bundle paths (spec 07, Req 3–4)."""

from typing import Literal

from pydantic import BaseModel

Section = Literal["prediction", "annotation", "cross"]


class Claim(BaseModel):
    section: Section
    text: str
    cites: list[str]


class EvidenceSummary(BaseModel):
    renderer: Literal["template", "claude"]
    claims: list[Claim]
    disclaimer: str
    dropped_claims: int = 0
    fallback_reason: str | None = None

    def section(self, name: Section) -> list[Claim]:
        return [c for c in self.claims if c.section == name]

    def to_markdown(self) -> str:
        titles = {"prediction": "Model predictions", "annotation": "Database annotations",
                  "cross": "How the evidence relates"}
        out = []
        for key, title in titles.items():
            claims = self.section(key)
            if claims:
                out += [f"**{title}**", *[f"- {c.text}" for c in claims], ""]
        out.append(f"_{self.disclaimer}_")
        return "\n".join(out)
