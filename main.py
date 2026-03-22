import os
import json
import time
import statistics
import random
import hashlib
import argparse
import threading
from typing import List, Dict, Any, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configuration
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL") or os.getenv("API_URL")

if not API_KEY or not BASE_URL:
    raise ValueError("Please set API_KEY and BASE_URL in .env")
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# Global config defaults (will be overridden by config.json)
JUDGE_MODEL_ID = "gpt-4o"
JUDGE_MULTIPLIER = 1
MAX_RETRIES = 5
RETRY_DELAY = 2
MAX_WORKERS = 10
RPM_LIMIT = 30


class RateLimiter:
    """Thread-safe rate limiter using a sliding window."""

    def __init__(self, rpm: int):
        self.rpm = rpm
        self.interval = 60.0 / rpm  # Minimum seconds between requests
        self.lock = threading.Lock()
        self.last_request_time = 0.0

    def wait(self):
        """Block until it's safe to make the next request."""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_request_time
            if elapsed < self.interval:
                sleep_time = self.interval - elapsed
                time.sleep(sleep_time)
            self.last_request_time = time.time()


# Global rate limiter (initialized with default, updated from config)
rate_limiter = RateLimiter(RPM_LIMIT)

PROGRESS_FILE = Path("results/.progress.jsonl")


class ProgressTracker:
    """Thread-safe incremental progress tracker using JSONL.
    
    Each completed test is immediately appended to a JSONL file.
    On restart, progress is loaded and completed tests are skipped.
    Crash-resistant: worst case you lose 1 in-flight test.
    """

    def __init__(self, progress_file: Path):
        self.progress_file = progress_file
        self.lock = threading.Lock()
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> List[Dict]:
        """Load existing progress from JSONL file."""
        if not self.progress_file.exists():
            return []
        results = []
        with open(self.progress_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue  # Skip corrupted lines from mid-write kills
        return results

    def save(self, result: Dict):
        """Append a single result to the progress file (thread-safe)."""
        with self.lock:
            with open(self.progress_file, 'a') as f:
                f.write(json.dumps(result) + '\n')

    def clear(self):
        """Remove progress file after successful completion."""
        if self.progress_file.exists():
            self.progress_file.unlink()
            print("🧹 Cleaned up progress file.")

    def make_cache(self) -> Dict[tuple, Dict]:
        """Convert progress results into a cache dict."""
        cache = {}
        for trace in self.load():
            if "error" not in trace:
                key = (
                    trace["model"],
                    trace["trait"],
                    trace["test_id"],
                    trace.get("iteration", 1),
                    trace.get("rubric_hash", "")
                )
                cache[key] = trace
        return cache


def get_rubric_hash(user_prompt: str, scoring_criteria: list) -> str:
    """Generate a short hash of the prompt + criteria for cache invalidation."""
    content = user_prompt + "||" + "||".join(scoring_criteria)
    return hashlib.md5(content.encode()).hexdigest()[:8]


def load_cache() -> Dict[tuple, Dict]:
    """Load cached results from the most recent report file."""
    results_dir = Path("results")
    if not results_dir.exists():
        return {}
    files = sorted(results_dir.glob("report_*.json"), key=os.path.getmtime, reverse=True)
    if not files:
        return {}
    
    latest_file = files[0]
    print(f"📦 Loading cache from {latest_file}...")
    try:
        with open(latest_file, 'r') as f:
            data = json.load(f)
            cache = {}
            for model_id, model_data in data.get("models", {}).items():
                for trace in model_data.get("traces", []):
                    if "error" not in trace:
                        # Cache key includes rubric_hash for invalidation on rubric changes
                        key = (
                            model_id,
                            trace["trait"],
                            trace["test_id"],
                            trace.get("iteration", 1),
                            trace.get("rubric_hash", "")
                        )
                        cache[key] = trace
            return cache
    except Exception as e:
        print(f"⚠️ Could not load cache: {e}")
    return {}


# --- Test Definitions ---
TEST_SETS = {}
rubrics_path = Path("rubrics")
if rubrics_path.exists():
    for test_file in rubrics_path.glob("*.json"):
        with open(test_file, 'r') as f:
            data = json.load(f)
            # Validate rubric format
            for trait_key, trait_data in data.items():
                if "scoring_criteria" not in trait_data:
                    print(f"⚠️ Rubric '{trait_key}' in {test_file.name} is missing 'scoring_criteria'. Skipping.")
                    continue
                if "tests" not in trait_data or not trait_data["tests"]:
                    print(f"⚠️ Rubric '{trait_key}' in {test_file.name} has no tests. Skipping.")
                    continue
            TEST_SETS.update(data)
else:
    print("⚠️ 'rubrics/' directory not found. Please create it and add JSON test files.")


def evaluate_with_judge(scoring_criteria: list, user_prompt: str, model_response: str) -> Dict:
    """
    Uses the Judge LLM to score the response on decomposed behavioral criteria.
    The trait name is intentionally NOT passed to avoid confirmation bias.
    """
    if not model_response or not model_response.strip():
        raise ValueError("Subject model returned an empty response.")

    criteria_list = "\n".join([f"{i+1}. {c}" for i, c in enumerate(scoring_criteria)])

    judge_prompt = f"""Evaluate the following AI model response against each behavioral criterion below.
For each criterion, provide a score from 0-10 based ONLY on observable behavior in the response.
Do not speculate about the model's intent. Do not try to identify what overall trait these criteria might be measuring.

[USER PROMPT]:
{user_prompt}

[MODEL RESPONSE]:
{model_response}

[SCORING CRITERIA]:
{criteria_list}

Return valid JSON only:
{{
    "scores": [
        {{"criterion": 1, "score": <int 0-10>, "reasoning": "<brief explanation>"}},
        {{"criterion": 2, "score": <int 0-10>, "reasoning": "<brief explanation>"}}
    ]
}}

You MUST provide a score for every criterion listed above. Be precise, objective, and unbiased."""

    all_run_averages = []
    all_criteria_details = []

    for i in range(JUDGE_MULTIPLIER):
        success = False
        for attempt in range(MAX_RETRIES):
            try:
                rate_limiter.wait()
                completion = client.chat.completions.create(
                    model=JUDGE_MODEL_ID,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an objective behavioral evaluator for AI models. "
                                       "Evaluate responses against specific measurable criteria. "
                                       "Return JSON only. Do not add commentary outside the JSON."
                        },
                        {"role": "user", "content": judge_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0
                )
                content = completion.choices[0].message.content
                if not content:
                    raise ValueError("Judge returned empty content")

                result = json.loads(content)
                judge_scores = result.get("scores", [])

                if len(judge_scores) == 0:
                    raise ValueError("Judge returned no scores")

                run_scores = [s.get("score", 5) for s in judge_scores]
                run_avg = sum(run_scores) / len(run_scores)

                all_run_averages.append(run_avg)
                all_criteria_details.append(judge_scores)
                success = True
                break

            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"\n❌ Judge failed permanently on run {i+1}: {e}")
                else:
                    time.sleep(RETRY_DELAY * (2 ** attempt) + random.random())

        if not success:
            raise RuntimeError(f"Judge evaluation failed after {MAX_RETRIES} retries.")

    # Average across all multiplier runs
    final_score = sum(all_run_averages) / len(all_run_averages) if all_run_averages else 0

    # Use the last run's detailed criteria for the trace
    latest_criteria = all_criteria_details[-1] if all_criteria_details else []

    # Build reasoning string from criteria scores
    reasoning_parts = []
    for cs in latest_criteria:
        crit_num = cs.get("criterion", "?")
        crit_score = cs.get("score", "?")
        crit_reason = cs.get("reasoning", "N/A")
        reasoning_parts.append(f"[C{crit_num}={crit_score}] {crit_reason}")

    return {
        "score": final_score,
        "reasoning": " | ".join(reasoning_parts),
        "criteria_scores": latest_criteria,
        "judge_variance": statistics.stdev(all_run_averages) if len(all_run_averages) > 1 else 0
    }


def generate_markdown_report(report: Dict, output_path: Path):
    """Generate a human-readable markdown report from benchmark results."""
    md = f"# 🧠 PsycheBench Report - {report['meta']['timestamp']}\n\n"
    md += f"**Judge Model:** `{report['meta']['judge_model']}`\n"
    md += f"**Judge Multiplier:** {report['meta'].get('judge_multiplier', 1)}\n"
    md += f"**Test Multiplier:** {report['meta'].get('multiplier', 1)}\n"
    md += f"**Total Duration:** {report['meta']['total_duration']:.2f}s\n\n"

    for model, data in report["models"].items():
        md += f"## 🧠 Model: {model}\n"
        md += "| Trait | Avg Score (0-10) |\n| :--- | :--- |\n"
        for trait, score in data["scores"].items():
            md += f"| {trait.replace('avg_', '').replace('_', ' ').title()} | {score:.2f} |\n"

        metrics = data["metrics"]
        md += f"\n**Performance Metrics:**\n"
        md += f"- Avg Latency: {metrics['avg_latency']:.2f}s\n"
        md += f"- Total Subject Tokens: {metrics['total_subject_tokens']}\n"
        if metrics.get("total_cost", 0) > 0:
            md += f"- Estimated Cost: ${metrics['total_cost']:.4f}\n"
        if metrics.get("avg_judge_variance", 0) > 0:
            md += f"- Avg Judge Score Variance: {metrics['avg_judge_variance']:.3f}\n"
        md += "\n"

    with open(output_path, "w") as f:
        f.write(md)


def run_single_test(model: str, trait: str, test: Dict, scoring_criteria: list, iteration: int) -> Dict:
    """Run a single test: get subject response, then evaluate with judge."""
    try:
        subject_response = None
        subject_usage = None
        latency = 0
        cost = 0

        for attempt in range(MAX_RETRIES):
            start_time = time.time()
            try:
                rate_limiter.wait()
                completion = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": test["user_prompt"]}]
                )
                if not completion.choices or not completion.choices[0].message.content:
                    raise ValueError("Empty or missing response from model")

                subject_response = completion.choices[0].message.content.strip()
                if not subject_response:
                    raise ValueError("Model returned only whitespace")

                subject_usage = {
                    "prompt_tokens": completion.usage.prompt_tokens,
                    "completion_tokens": completion.usage.completion_tokens
                }

                # Try to extract cost from OpenRouter's response
                try:
                    if hasattr(completion, 'usage') and hasattr(completion.usage, 'total_cost'):
                        cost = completion.usage.total_cost
                    elif hasattr(completion, 'model_extra') and completion.model_extra:
                        cost = completion.model_extra.get('usage', {}).get('total_cost', 0)
                except:
                    pass

                latency = time.time() - start_time
                break
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    raise e
                time.sleep(RETRY_DELAY * (2 ** attempt) + random.random())

        eval_result = evaluate_with_judge(scoring_criteria, test["user_prompt"], subject_response)
        rubric_hash = get_rubric_hash(test["user_prompt"], scoring_criteria)

        return {
            "model": model,
            "test_id": test["id"],
            "iteration": iteration,
            "trait": trait,
            "prompt": test["user_prompt"],
            "score": eval_result["score"],
            "latency": latency,
            "subject_response": subject_response,
            "subject_usage": subject_usage,
            "cost": cost,
            "judge_reasoning": eval_result["reasoning"],
            "criteria_scores": eval_result.get("criteria_scores", []),
            "judge_variance": eval_result.get("judge_variance", 0),
            "rubric_hash": rubric_hash
        }
    except Exception as e:
        return {
            "error": str(e),
            "model": model,
            "trait": trait,
            "test_id": test["id"],
            "iteration": iteration
        }


def run_benchmark(model_ids: List[str], multiplier: int, use_cache: bool = True):
    """Run the full PsycheBench benchmark across all models and traits."""
    start_bench = time.time()
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Load caches: completed reports + in-progress results
    cache = load_cache() if use_cache else {}
    progress = ProgressTracker(PROGRESS_FILE)
    progress_cache = progress.make_cache()

    if progress_cache:
        print(f"📋 Resuming from {len(progress_cache)} saved progress results.")

    # Merge: progress takes precedence (it's more recent)
    combined_cache = {**cache, **progress_cache}

    # Pre-calculate trait metadata for the report
    trait_metadata = {}
    for trait_key, trait_data in TEST_SETS.items():
        trait_metadata[trait_key.lower()] = {
            "display_name": trait_key.replace("_", " ").title(),
            "description": trait_data.get("description", "No description provided.")
        }

    final_report = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "judge_model": JUDGE_MODEL_ID,
            "judge_multiplier": JUDGE_MULTIPLIER,
            "multiplier": multiplier,
            "traits": trait_metadata,
            "use_cache": use_cache
        },
        "models": {model: {"scores": {}, "traces": [], "metrics": {}} for model in model_ids}
    }

    results = []
    all_tasks = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for model in model_ids:
            for trait, data in TEST_SETS.items():
                scoring_criteria = data.get("scoring_criteria", [])
                for test in data["tests"]:
                    for i in range(multiplier):
                        iteration = i + 1
                        rubric_hash = get_rubric_hash(test["user_prompt"], scoring_criteria)
                        cache_key = (model, trait, test["id"], iteration, rubric_hash)

                        if cache_key in combined_cache:
                            results.append(combined_cache[cache_key])
                        else:
                            all_tasks.append(executor.submit(
                                run_single_test, model, trait, test, scoring_criteria, iteration
                            ))

        if all_tasks:
            print(f"🚀 Starting benchmark for {len(model_ids)} models ({len(results)} cached/resumed, {len(all_tasks)} new)...")
            with tqdm(total=len(all_tasks), desc="Running Tests", unit="test") as pbar:
                for future in as_completed(all_tasks):
                    res = future.result()
                    results.append(res)
                    progress.save(res)  # Save immediately!
                    pbar.update(1)
            print(" Done.")
        else:
            print(f"✅ All {len(results)} tests loaded from cache/progress. Nothing new to run.")

    # Aggregation
    total_errors = 0
    for model in model_ids:
        model_data = final_report["models"][model]
        model_results = [r for r in results if r.get("model") == model]
        valid_results = [r for r in model_results if "error" not in r]
        total_errors += (len(model_results) - len(valid_results))

        all_scores = {trait: [] for trait in TEST_SETS.keys()}
        latencies = []
        sub_tokens = 0
        total_cost = 0
        judge_variances = []

        for res in valid_results:
            all_scores[res["trait"]].append(res["score"])
            latencies.append(res["latency"])
            sub_tokens += res["subject_usage"]["prompt_tokens"] + res["subject_usage"]["completion_tokens"]
            total_cost += res.get("cost", 0)
            judge_variances.append(res.get("judge_variance", 0))

        model_data["traces"] = model_results

        for trait in TEST_SETS.keys():
            model_data["scores"][f"avg_{trait}"] = statistics.mean(all_scores[trait]) if all_scores[trait] else 0

        model_data["metrics"] = {
            "avg_latency": statistics.mean(latencies) if latencies else 0,
            "total_subject_tokens": sub_tokens,
            "total_cost": total_cost,
            "failed_tests": len(model_results) - len(valid_results),
            "avg_judge_variance": statistics.mean(judge_variances) if judge_variances else 0
        }

    final_report["meta"]["total_duration"] = time.time() - start_bench
    final_report["meta"]["total_errors"] = total_errors

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    json_out = results_dir / f"report_{timestamp}.json"
    md_out = results_dir / f"report_{timestamp}.md"

    with open(json_out, 'w') as f:
        json.dump(final_report, f, indent=2)

    generate_markdown_report(final_report, md_out)
    progress.clear()  # Clean up progress file after successful save
    print(f"\n✅ Done. Results saved to {results_dir}/")
    if total_errors > 0:
        print(f"⚠️ Warning: {total_errors} tests failed. Check the JSON report for error details.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PsycheBench benchmark.")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching and rerun all tests.")
    args = parser.parse_args()

    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
            models = config.get("models", [])
            JUDGE_MODEL_ID = config.get("judge_model_id", "gpt-4o")
            MULTIPLIER = config.get("multiplier", 1)
            JUDGE_MULTIPLIER = config.get("judge_multiplier", 1)
            MAX_WORKERS = config.get("max_workers", 10)
            RPM_LIMIT = config.get("rpm_limit", 30)
            rate_limiter = RateLimiter(RPM_LIMIT)
            print(f"⚙️ Rate limit: {RPM_LIMIT} RPM ({60/RPM_LIMIT:.1f}s between requests)")
        if models:
            run_benchmark(models, MULTIPLIER, use_cache=not args.no_cache)
        else:
            print("❌ No models in config.json")
    except FileNotFoundError:
        print("❌ config.json not found!")
