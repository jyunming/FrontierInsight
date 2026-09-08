"""Skills in the web dashboard — the third surface of the promotion gate.

Reads the same serialiser the CLI's ``--json`` and the VSCode chat panel read
(``SkillState.to_dict``). Discovery, self-test execution, content hashing, the
approval ledger and the static scan all stay in ``core.skills``; nothing about
the gate is re-decided here, because three copies of that logic is how the
three interfaces drifted apart before.

Two things this module *does* own, both about the approval ceremony:

**A named approver is required, and it comes from the form.** There is no
anonymous approver anywhere in FI, and a browser session is not a person. The
name is typed for each approval rather than remembered, for the same reason
the CLI has no default and the VSCode box is never auto-submitted.

**A high-severity finding needs a second, explicit submit.** Same rule as
``--despite-findings``: approving past a flagged network call is often
correct, but it has to be a decision rather than a click-through.

The dashboard binds to 127.0.0.1 by default. That is the only thing standing
between this route and anyone who can reach the port — there is no auth layer
— so a deployment that changes the bind address is choosing to expose skill
approval along with everything else.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

_log = logging.getLogger("fi.web.skills")


def _states(run_tests: bool) -> list[dict[str, Any]]:
    from core.skills import discover, evaluate

    return [evaluate(s, run_test=run_tests).to_dict() for s in discover()]


def register_skills_routes(app: FastAPI) -> None:
    """Attach the /skills page and /api/skills/* routes."""
    static_dir = Path(__file__).resolve().parent / "static"

    @app.get("/skills", response_class=HTMLResponse)
    async def skills_page() -> HTMLResponse:
        page = static_dir / "skills.html"
        if not page.exists():
            return HTMLResponse("<h1>Skills UI not installed</h1>", status_code=500)
        return HTMLResponse(page.read_text(encoding="utf-8"))

    @app.get("/api/skills")
    async def list_skills(run_selftests: bool = False) -> JSONResponse:
        """The library.

        Self-tests are off by default: each may take up to 120 seconds, so a
        real library would hang the page. The response says which it was, so
        the UI can show "as last recorded" rather than implying it verified
        anything.
        """
        try:
            rows = await asyncio.to_thread(_states, run_selftests)
        except Exception as e:  # noqa: BLE001
            _log.warning("skills listing failed: %s", e)
            raise HTTPException(500, f"could not read the skill library: {e}")
        return JSONResponse({"skills": rows, "selftests_run": run_selftests})

    @app.get("/api/skills/{name}/scan")
    async def scan_skill(name: str) -> JSONResponse:
        from core.skills import discover, scan

        skill = next((s for s in discover() if s.name == name), None)
        if skill is None:
            raise HTTPException(404, f"no skill named {name}")
        found = await asyncio.to_thread(scan.scan, skill)
        return JSONResponse({
            "name": skill.name,
            "path": str(skill.path),
            "files_reviewed": len(skill.hashable_files()),
            "rule_count": scan.RULE_COUNT,
            "worst": scan.worst(found),
            "summary": scan.summarise(found),
            "findings": [
                {"rule": f.rule, "severity": f.severity, "path": f.path,
                 "line": f.line, "detail": f.detail}
                for f in found
            ],
        })

    @app.post("/api/skills/{name}/approve")
    async def approve_skill(name: str, request: Request) -> JSONResponse:
        body = await request.json()
        who = str(body.get("approved_by") or "").strip()
        despite = bool(body.get("despite_findings"))

        if not who:
            raise HTTPException(
                400,
                "An approver's name is required — the gate exists so that a "
                "person decides, and there is no anonymous approver.",
            )

        from core.skills import approval, discover, evaluate, scan
        from core.skills.base import Status

        skill = next((s for s in discover() if s.name == name), None)
        if skill is None:
            raise HTTPException(404, f"no skill named {name}")

        state = await asyncio.to_thread(evaluate, skill)
        if state.status is Status.QUARANTINED:
            raise HTTPException(
                409,
                f"{name} fails its own self-test, so it cannot be approved. "
                "A person signing off on something that provably does not "
                "work is the one case where the human gate is not the last "
                "word.",
            )
        if state.status is Status.UNTESTED:
            raise HTTPException(
                409,
                f"{name} carries no selftest.py and can never be promoted.",
            )

        found = await asyncio.to_thread(scan.scan, skill)
        highs = [f for f in found if f.severity == scan.HIGH]
        if highs and not despite:
            # Not an error the UI should swallow — it is the review the user
            # has to see before the second submit.
            return JSONResponse(
                status_code=409,
                content={
                    "needs_confirmation": True,
                    "high_findings": [
                        {"rule": f.rule, "severity": f.severity, "path": f.path,
                         "line": f.line, "detail": f.detail}
                        for f in highs
                    ],
                    "message": (
                        f"{len(highs)} high-severity finding(s). These are "
                        "heuristics, not verdicts — a real tool may "
                        "legitimately need what they flag — but approving "
                        "past them has to be a decision. Resubmit with "
                        "despite_findings to record that you made it."
                    ),
                },
            )

        note = "approved via web dashboard"
        if highs:
            note += f" DESPITE {len(highs)} high-severity scan finding(s)"
        try:
            from core.skills.scaffold import selftest_is_generated

            if selftest_is_generated(skill.path):
                note += " [generated selftest]"
        except Exception:  # noqa: BLE001
            pass

        h = skill.content_hash()
        await asyncio.to_thread(
            approval.approve, name, h, approved_by=who, note=note,
        )
        return JSONResponse({
            "approved": True, "name": name, "content_hash": h,
            "approved_by": who, "despite_findings": bool(highs),
            "note": "Editing the skill lapses this approval.",
        })

    @app.post("/api/skills/{name}/revoke")
    async def revoke_skill(name: str) -> JSONResponse:
        from core.skills import approval

        removed = await asyncio.to_thread(approval.revoke, name)
        return JSONResponse({"revoked": bool(removed), "name": name})
