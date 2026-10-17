"""Build BST grading jobs and score results from the isolated runner."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

from code_lm.evaluation.sandbox import fingerprint
from code_lm.data.prepare import read_rows, data_digest

ROOT = Path(__file__).resolve().parents[2]


def build_jobs(rows, suites, generations=None):
    found = {r["id"]: r for r in generations} if generations is not None else None
    if found is not None and (len(found) != len(generations) or set(found) != {r["id"] for r in rows}):
        raise ValueError("Generations must match selected tasks exactly once")
    node_code = (ROOT / "code_lm/data/tree.py").read_text()
    jobs, candidates = {}, []
    for row in rows:
        suite = suites[row["suite_id"]]
        if hashlib.sha256(suite["tests"].encode()).hexdigest() != row["suite_id"]:
            raise ValueError("Test suite hash mismatch")
        codes = {"reference": row["answer"],
                 "negative": f"def {row['entry_point']}(*args, **kwargs):\n    return None\n"}
        if found is not None:
            generated = found[row["id"]]
            if generated["prompt_sha256"] != hashlib.sha256(row["prompt"].encode()).hexdigest():
                raise ValueError("Generation used a different prompt")
            code = generated["completion"].strip()
            fenced = re.search(r"```(?:python|py)?\s*\n(.*?)```", code, flags=re.S)
            codes["model"] = fenced.group(1) if fenced else code
        role = "model" if found is not None else "reference"
        for name, code in codes.items():
            job = {"role": name, "answer": node_code + "\n" + code, **suite}
            job["id"] = fingerprint(job)
            job["job_sha256"] = fingerprint(job)
            jobs[job["id"]] = job
            if name == role:
                candidates.append({"id": row["id"], "family": row["family"],
                                   "contract": row["basic_contract"], "job_id": job["id"]})
    return list(jobs.values()), candidates


def summarize(jobs, candidates, results):
    by_job = {j["id"]: j for j in jobs}
    by_result = {r["id"]: r for r in results}
    if len(by_result) != len(results) or set(by_result) != set(by_job):
        raise ValueError("Missing or duplicated sandbox results")
    passed = {}
    for key, job in by_job.items():
        observed = by_result[key]
        result = observed["result"]
        if observed["job_sha256"] != job["job_sha256"]:
            raise ValueError("Result hash mismatch")
        if result["status"] not in {"pass", "fail", "timeout"}:
            raise RuntimeError(f"Grader infrastructure failure: {result}")
        passed[key] = (result["status"] == "pass" and result.get("tests") == job["expected_tests"]
                       and not any(result.get(k, 0) for k in ("failures", "errors", "skipped")))
        if result["status"] == "pass" and not passed[key]:
            raise RuntimeError("Incomplete test execution")
        if job["role"] == "reference" and not passed[key]:
            raise RuntimeError(f"Reference solution failed: {result}")
        if job["role"] == "negative" and not (result["status"] == "fail" and result.get("failures", 0)
                                                 and not result.get("errors", 0)):
            raise RuntimeError("Negative control was not rejected by assertions")
    outcomes, groups = [], defaultdict(list)
    for candidate in candidates:
        key = candidate["job_id"]
        correct = passed[key]
        outcome = {**candidate, "correct": correct}
        outcomes.append(outcome)
        groups[candidate["contract"]].append(correct)
    return {"tasks": len(outcomes), "correct": sum(r["correct"] for r in outcomes),
            "pass_rate": sum(r["correct"] for r in outcomes) / len(outcomes),
            "by_contract": {k: {"correct": sum(v), "tasks": len(v)} for k, v in groups.items()},
            "outcomes": outcomes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/bst_covered"))
    parser.add_argument("--suites", type=Path, default=Path("data/bst_covered/suites.json"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--generations", type=Path)
    mode.add_argument("--references", action="store_true", help="Verify suites using references and null controls")
    parser.add_argument("--output", type=Path, default=Path("runs/evaluation.json"))
    parser.add_argument("--limit", type=int)
    execution = parser.add_mutually_exclusive_group(required=True)
    execution.add_argument("--export-jobs", type=Path, help="Write jobs for the isolated sandbox.py runner")
    execution.add_argument("--sandbox-results", type=Path, help="Read the runner output as JSON or JSONL")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("Limit must be positive")
    if args.output.exists():
        raise FileExistsError(f"Choose a new output path: {args.output}")
    rows = list(read_rows(args.data))[:args.limit]
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Expected nonempty tasks with unique IDs")
    generations = list(read_rows(args.generations)) if args.generations else None
    jobs, candidates = build_jobs(rows, json.loads(args.suites.read_text()), generations)
    if args.export_jobs:
        args.export_jobs.parent.mkdir(parents=True, exist_ok=True)
        with args.export_jobs.open('x') as stream:
            json.dump(jobs, stream, indent=2)
        print(f"Exported {len(jobs)} isolated grading jobs to {args.export_jobs}")
        return
    text = args.sandbox_results.read_text()
    results = json.loads(text) if text.lstrip().startswith('[') else [json.loads(line) for line in text.splitlines()]
    summary = summarize(jobs, candidates, results)
    summary["mode"] = "reference_verification" if args.references else "model_evaluation"
    summary["data_sha256"] = data_digest(args.data)
    if args.generations:
        summary["generations_sha256"] = hashlib.sha256(args.generations.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "outcomes"}, indent=2))


if __name__ == "__main__":
    main()
