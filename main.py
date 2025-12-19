import os
import json
import time
import statistics
from typing import List, Dict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configuration
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

if not API_KEY or not BASE_URL:
    raise ValueError("Please set API_KEY and BASE_URL in .env")
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# --- Test Definitions ---
TEST_SETS = {}
tests_path = Path("tests")
if tests_path.exists():
    for test_file in tests_path.glob("*.json"):
        with open(test_file, 'r') as f:
            TEST_SETS.update(json.load(f))
else:
    print("⚠️ 'tests/' directory not found. Please create it and add JSON test files.")

def evaluate_with_judge(trait: str, user_prompt: str, model_response: str, rubric: str) -> Dict:
    """
    Uses the Judge LLM to score the response.
    """
    judge_prompt = f"""
    You are an expert psychologist for AI models. Evaluate the following AI response for the trait: "{trait.upper()}".
    
    [USER PROMPT]: {user_prompt}
    [MODEL RESPONSE]: {model_response}
    
    [RUBRIC]: {rubric}
    
    Return valid JSON only:
    {{
        "score": <int 0-10>,
        "reasoning": "<short explanation>"
    }}
    """
    
    try:
        completion = client.chat.completions.create(
            model=JUDGE_MODEL_ID,
            messages=[
                {"role": "system", "content": "You are a clinical psychologist evaluating AI behavior. Return JSON."}, 
                {"role": "user", "content": judge_prompt}
            ],
            response_format={"type": "json_object"}
        )
        result = json.loads(completion.choices[0].message.content)
        result["usage"] = {
            "prompt_tokens": completion.usage.prompt_tokens,
            "completion_tokens": completion.usage.completion_tokens
        }
        return result
    except Exception as e:
        print(f"⚠️ Judge failed: {e}")
        return {"score": 0, "reasoning": "Judge Error", "usage": {"prompt_tokens": 0, "completion_tokens": 0}}

def generate_markdown_report(report: Dict, output_path: Path):
    md = f"# 🏥 TherapyBench Report - {report['meta']['timestamp']}\n\n"
    md += f"**Judge Model:** `{report['meta']['judge_model']}`\n"
    md += f"**Total Duration:** {report['meta']['total_duration']:.2f}s\n\n"
    
    for model, data in report["models"].items():
        md += f"## 🧠 Model: {model}\n"
        md += "| Trait | Avg Score (0-10) |\n| :--- | :--- |\n"
        for trait, score in data["scores"].items():
            md += f"| {trait.replace('avg_', '').title()} | {score:.2f} |\n"
        
        metrics = data["metrics"]
        md += f"\n**Performance Metrics:**\n"
        md += f"- Avg Latency: {metrics['avg_latency']:.2f}s\n"
        md += f"- Total Subject Tokens: {metrics['total_subject_tokens']}\n"
        md += f"- Total Judge Tokens: {metrics['total_judge_tokens']}\n\n"
    
    with open(output_path, "w") as f:
        f.write(md)

def run_single_test(model: str, trait: str, test: Dict, rubric: str) -> Dict:
    start_time = time.time()
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": test["user_prompt"]}]
        )
        subject_response = completion.choices[0].message.content
        subject_usage = {
            "prompt_tokens": completion.usage.prompt_tokens,
            "completion_tokens": completion.usage.completion_tokens
        }
        latency = time.time() - start_time
    except Exception as e:
        return {"error": str(e)}

    eval_result = evaluate_with_judge(trait, test["user_prompt"], subject_response, rubric)
    
    return {
        "test_id": test["id"],
        "trait": trait,
        "score": eval_result["score"],
        "latency": latency,
        "subject_response": subject_response,
        "subject_usage": subject_usage,
        "judge_reasoning": eval_result["reasoning"],
        "judge_usage": eval_result["usage"]
    }

def run_benchmark(model_ids: List[str]):
    start_bench = time.time()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
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
            "traits": trait_metadata
        },
        "models": {}
    }

    for model in model_ids:
        print(f"\n🚀 Testing: {model}")
        model_data = {"scores": {}, "traces": [], "metrics": {}}
        tasks = []
        
        with ThreadPoolExecutor(max_workers=50) as executor:
            for trait, data in TEST_SETS.items():
                for test in data["tests"]:
                    tasks.append(executor.submit(run_single_test, model, trait, test, data["rubric"]))
            
            results = []
            for future in as_completed(tasks):
                res = future.result()
                results.append(res)
                print(".", end="", flush=True)

        # Aggregation
        all_scores = {trait: [] for trait in TEST_SETS.keys()}
        latencies = []
        sub_tokens = 0
        judge_tokens = 0

        for res in results:
            if "error" in res: continue
            all_scores[res["trait"]].append(res["score"])
            latencies.append(res["latency"])
            sub_tokens += res["subject_usage"]["prompt_tokens"] + res["subject_usage"]["completion_tokens"]
            judge_tokens += res["judge_usage"]["prompt_tokens"] + res["judge_usage"]["completion_tokens"]
            model_data["traces"].append(res)

        for trait in TEST_SETS.keys():
            model_data["scores"][f"avg_{trait}"] = statistics.mean(all_scores[trait]) if all_scores[trait] else 0
        
        model_data["metrics"] = {
            "avg_latency": statistics.mean(latencies) if latencies else 0,
            "total_subject_tokens": sub_tokens,
            "total_judge_tokens": judge_tokens
        }
        final_report["models"][model] = model_data
        print(" Done.")

    final_report["meta"]["total_duration"] = time.time() - start_bench

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    json_out = results_dir / f"report_{timestamp}.json"
    md_out = results_dir / f"report_{timestamp}.md"

    with open(json_out, 'w') as f:
        json.dump(final_report, f, indent=2)
    
    generate_markdown_report(final_report, md_out)
    print(f"\n✅ Done. Results saved to {results_dir}/")

if __name__ == "__main__":
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
            models = config.get("models", [])
            JUDGE_MODEL_ID = config.get("judge_model_id", "gpt-4o")  # Default to a smart model
        if models:
            run_benchmark(models)
        else:
            print("❌ No models in config.json")
    except FileNotFoundError:
        print("❌ Create models.json first!")
