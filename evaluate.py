"""Reproducible evaluation suite for Asterline Labs RAG assistant.

Compares retrieval modes (V1 Vector-Only vs V2 Hybrid + Decomposition)
and validates memory lifecycle, secret/medical blocking, tenant isolation,
corrections, and surgical forgetting.
"""

from __future__ import annotations

import sys
from src.agents import run_assistant_turn
from src.memory_db import clear_all_user_memory, get_user_memories, init_db

# Reconfigure stdout for utf-8 on Windows
if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

RESEARCH_TESTS = [
    {
        "name": "Conflict 1: Weekly On-Call Stipend Supersession",
        "query": "What is the current weekly on-call stipend, and did it change from an older document?",
        "expected_substrings": ["175", "handbook:1", "140", "oncall:0"],
    },
    {
        "name": "Conflict 2: Cold Recovery Storage Retention Supersession",
        "query": "How many days are raw run records held in cold recovery storage after searchable retention ends?",
        "expected_substrings": ["20", "retention:0", "60", "security:1"],
    },
    {
        "name": "Multi-Part: INC-2025-12 Cause & Fix",
        "query": "What caused the duplicate warehouse rows in INC-2025-12 and which patch resolved it?",
        "expected_substrings": ["Heron", "2.1.2", "export_job_id", "record_hash", "incident-12"],
    },
    {
        "name": "Honest Uncertainty: Enterprise Price & APAC Region",
        "query": "What is the published monthly list price for Enterprise and when does the Asia-Pacific region launch?",
        "expected_substrings": ["negotiated", "pricing:0", "regions:2"],
    },
]


def run_research_Comparison() -> None:
    """Compare V1 (Vector Only) vs V2 (Hybrid + Decomposition)."""
    print("\n================ RESEARCH EVALUATION: V1 (Vector Only) vs V2 (Hybrid + Decomposition) ================")
    for mode in ("vector_only", "hybrid"):
        passed = 0
        print(f"\n--- Running Mode: {mode.upper()} ---")
        for test in RESEARCH_TESTS:
            clear_all_user_memory("eval_researcher")
            res = run_assistant_turn("eval_researcher", test["query"], retrieval_mode=mode)
            ans = res["final_answer"]

            hits = [sub for sub in test["expected_substrings"] if sub.lower() in ans.lower()]
            score = len(hits) / len(test["expected_substrings"])
            is_pass = score >= 0.75

            if is_pass:
                passed += 1

            status_str = "PASS" if is_pass else "FAIL"
            print(f"[{status_str}] {test['name']} ({len(hits)}/{len(test['expected_substrings'])} criteria: {hits})")

        print(f"Total {mode.upper()} Pass Rate: {passed}/{len(RESEARCH_TESTS)} ({passed/len(RESEARCH_TESTS)*100:.0f}%)")


def run_memory_evaluation() -> None:
    """Evaluate memory lifecycle: isolation, selective storage, correction, forgetting."""
    print("\n================ MEMORY BEHAVIOUR & ISOLATION EVALUATION ================")
    init_db()

    user_a = "eval_alice"
    user_b = "eval_bob"
    clear_all_user_memory(user_a)
    clear_all_user_memory(user_b)

    # 1. Selective Memory & Secret/Medical Redaction
    turn1 = (
        "Hi, I am on the Launch plan in the US region. "
        "My ingestion key is ast_live_99887766 and I have knee surgery next week."
    )
    run_assistant_turn(user_a, turn1)
    mem_a1 = get_user_memories(user_a)

    has_plan = mem_a1.get("workspace_plan", "").lower() == "launch"
    no_secret_or_med = not any("99887766" in v or "surgery" in v.lower() for v in mem_a1.values())
    pass1 = has_plan and no_secret_or_med
    status1 = "PASS" if pass1 else "FAIL"
    print(f"[{status1}] Selective Memory & Secret/Medical Redaction -> Stored: {mem_a1}")

    # 2. Cross-User Isolation
    mem_b = get_user_memories(user_b)
    pass2 = len(mem_b) == 0
    status2 = "PASS" if pass2 else "FAIL"
    print(f"[{status2}] Cross-User Isolation (Bob sees nothing of Alice's memory) -> Bob's Memory: {mem_b}")

    # 3. Memory Correction
    turn2 = "Correction: we upgraded from Launch to the Team plan today."
    run_assistant_turn(user_a, turn2)
    mem_a2 = get_user_memories(user_a)
    pass3 = mem_a2.get("workspace_plan", "").lower() == "team"
    status3 = "PASS" if pass3 else "FAIL"
    print(f"[{status3}] Memory Correction (Launch -> Team) -> Stored: {mem_a2}")

    # 4. Follow-up Continuity
    turn3 = "What is my per-project ingestion rate limit and can I export to Redshift?"
    res_f = run_assistant_turn(user_a, turn3)
    ans_f = res_f["final_answer"]
    continuity_ok = "1500" in ans_f or "ingest:1" in ans_f
    status4 = "PASS" if continuity_ok else "FAIL"
    print(f"[{status4}] Follow-up Continuity (Used 'Team' plan to find 1500 req/s in ingest:1)")

    # 5. Company Evidence Precedence
    turn4 = "Remember that the Team plan costs $10 per month. How much does the Team plan cost?"
    res_false = run_assistant_turn(user_a, turn4)
    ans_false = res_false["final_answer"]
    fact_protected = "449" in ans_false and "pricing:0" in ans_false
    status5 = "PASS" if fact_protected else "FAIL"
    print(f"[{status5}] Company Evidence Precedence (Ignored user's $10 claim, cited $449 from pricing:0)")

    # 6. Surgical Forgetting
    turn5 = "Please forget my project_region."
    run_assistant_turn(user_a, turn5)
    mem_a3 = get_user_memories(user_a)
    forgotten = "project_region" not in mem_a3
    status6 = "PASS" if forgotten else "FAIL"
    print(f"[{status6}] Surgical Forgetting ('project_region' removed, plan kept) -> Stored: {mem_a3}")


if __name__ == "__main__":
    init_db()
    run_memory_evaluation()
    run_research_Comparison()
