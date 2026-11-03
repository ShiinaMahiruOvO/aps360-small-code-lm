"""Build BST grading jobs and score results from the isolated runner."""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re

from code_lm.data.bst_tasks import NODE_CODE
from code_lm.evaluation.sandbox import fingerprint
from code_lm.data.prepare import read_rows, data_digest



def build_jobs(rows, suites, generations=None):
    found = {r["id"]: r for r in generations} if generations is not None else None
    if found is not None and (len(found) != len(generations) or set(found) != {r["id"] for r in rows}):
        raise ValueError("Generations must match selected tasks exactly once")
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
            job = {"role": name, "answer": NODE_CODE + "\n" + code, **suite}
            job["id"] = fingerprint(job)
            job["job_sha256"] = fingerprint(job)
            jobs[job["id"]] = job
            if name == role:
                candidates.append({"id": row["id"], "source_id": row.get("source_id", row["id"]),
                                   "split": row.get("split"), "family": row["family"],
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


def pass_at_k(n, c, k):
    """Unbiased pass@k from n samples with c correct (Chen et al., 2021)."""
    return 1.0 if n - c < k else 1 - math.prod((n - c - i) / (n - i) for i in range(k))


def report(outcomes, ks=(1, 8), resamples=2000, seed=0):
    """Per-split pass@k, per-family pass@1 and a task-level bootstrap 95% CI for pass@1."""
    tasks = defaultdict(list)
    for o in outcomes:
        tasks[(o["split"], o["source_id"], o["contract"])].append(o["correct"])
    result = {}
    for split in sorted({s for s, _, _ in tasks}):
        rows = [(c, v) for (s, _, c), v in tasks.items() if s == split]
        p1 = [sum(v) / len(v) for _, v in rows]
        rng = random.Random(seed)
        boot = sorted(sum(rng.choices(p1, k=len(p1))) / len(p1) for _ in range(resamples))
        families = defaultdict(list)
        for (contract, v), p in zip(rows, p1):
            families[contract].append(p)
        result[split] = {"tasks": len(rows), "samples_per_task": len(rows[0][1]),
                         **{f"pass@{k}": sum(pass_at_k(len(v), sum(v), k) for _, v in rows) / len(rows)
                            for k in ks if k <= min(len(v) for _, v in rows)},
                         "pass@1_ci95": [boot[int(.025 * resamples)], boot[int(.975 * resamples) - 1]],
                         "by_contract": {c: sum(v) / len(v) for c, v in sorted(families.items())}}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/bst"))
    parser.add_argument("--suites", type=Path, help="Defaults to suites.json inside --data")
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
    if generations and "source_id" in generations[0]:
        source = {r["id"]: r for r in rows}  # one grading row per sample
        rows = [{**source[g["source_id"]], "id": g["id"], "source_id": g["source_id"]} for g in generations]
    suites = args.suites or (args.data if args.data.is_dir() else args.data.parent) / "suites.json"
    jobs, candidates = build_jobs(rows, json.loads(suites.read_text()), generations)
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
    summary["report"] = report(summary["outcomes"])
    summary["data_sha256"] = data_digest(args.data)
    if args.generations:
        summary["generations_sha256"] = hashlib.sha256(args.generations.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary["report"], indent=2))


if __name__ == "__main__":
    main()
