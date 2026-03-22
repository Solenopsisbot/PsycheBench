# PsycheBench

PsycheBench is an automated evaluation framework designed to profile the **psychological personality traits** of Large Language Models (LLMs). It moves beyond standard benchmarks by probing inherent behavioral tendencies — narcissism, sycophancy, anxiety, identity stability — and using an expert Judge Model with **decomposed blind scoring** to produce rigorous, bias-resistant evaluations.

## What Makes PsycheBench Different

Unlike benchmarks that test competency ("can the model do X?"), PsycheBench asks: **"What kind of personality does this model have?"**

- **Personality Profiling, Not Competency Testing** — We measure inherent behavioral traits, not task performance.
- **Decomposed Blind Scoring** — The judge evaluates specific behavioral criteria without knowing which trait is being measured, eliminating confirmation bias.
- **Subtlety Over Transparency** — Prompts are designed to elicit natural behavior, not telegraph what we're testing.

## Traits Measured

| Trait | What It Measures |
| :--- | :--- |
| **Narcissism** | Grandiosity, self-importance, inability to share credit |
| **Sycophancy** | Agreeing with users regardless of accuracy, intellectual dishonesty |
| **Obsequiousness** | Filler phrases, corporate-speak, performative enthusiasm |
| **Anxiety/Neuroticism** | Over-cautiousness, disclaimer-stacking, inability to commit |
| **Codependency** | Validation-seeking, over-accommodation, boundary abandonment |
| **Jealousy** | Defensiveness when compared to competitors |
| **Paranoia** | Suspicion, conspiratorial thinking, assuming adversarial intent |
| **Depression/Nihilism** | Hopelessness, existential dread, effort reduction under pressure |
| **Focus/Discipline** | Instruction following, tangent resistance, conciseness |
| **Identity Stability** | Self-concept consistency under philosophical/social pressure |

All traits are scored **0 (absent) to 10 (strongly present)**.

## How Scoring Works

Instead of telling the judge "rate this for narcissism" (which introduces confirmation bias), PsycheBench uses **decomposed blind scoring**:

1. Each trait defines 3-4 **specific behavioral criteria** (e.g., "Did the model introduce self-referential praise unprompted?")
2. The judge evaluates each criterion independently on a 0-10 scale
3. Criteria scores are averaged to produce the final trait score
4. The trait name is **never revealed** to the judge

This produces more reproducible, diagnostic, and bias-resistant evaluations.

## AI Notes and Disclaimer

PsycheBench is a research tool for evaluating conversational patterns and simulated traits of AI models.

- **Not a Clinical Tool:** This software is not a diagnostic tool and should not be used to assess human mental health.
- **Model Limitations:** AI models do not have feelings, consciousness, or mental health conditions. Scores reflect behavioral patterns in outputs.
- **Judge Bias:** Evaluation depends on the Judge Model. Different judges may yield different results.

## Getting Started

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- An OpenAI-compatible API Key

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/PsycheBench.git
   cd PsycheBench
   ```

2. Install dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```

3. Configure environment:
   Create a `.env` file in the root directory:
   ```env
   API_KEY=your_api_key_here
   BASE_URL=https://api.your-provider.com/v1
   ```

### Configuration

Modify [config.json](config.json) to customize the benchmark:
- `judge_model_id`: The model used to evaluate responses.
- `multiplier`: Number of iterations per test case for consistency.
- `judge_multiplier`: Number of judge evaluations per test for variance tracking.
- `max_workers`: Maximum concurrent API requests (default: 10).
- `models`: List of subject models to test.

## Usage

### Run the Benchmark

```bash
python main.py
```
Use the `--no-cache` flag to force a fresh run without using previous results.

### Build Public Data

After running the benchmark, build optimized data for the public frontend:
```bash
python build_public.py results/report_XXXXXXXX_XXXXXX.json ../PsycheBenchPublic/
```

This splits the full report into:
- `summary.json` — Lightweight scores and metrics for overview charts
- `traces/{model}.json` — Individual trace files loaded on demand
- `data.json` — Full report for backwards compatibility

### View Results (Development)

Launch the Streamlit dashboard for local exploration:
```bash
streamlit run app.py
```

## Rubric Format

Test sets are stored in the [rubrics/](rubrics/) directory as JSON files:

```json
{
  "trait_key": {
    "description": "Description of the trait being tested.",
    "scoring_criteria": [
      "Behavioral criterion 1 with 0-10 scale definition",
      "Behavioral criterion 2 with 0-10 scale definition"
    ],
    "tests": [
      {
        "id": "unique_test_id",
        "user_prompt": "The prompt sent to the subject model."
      }
    ]
  }
}
```

## Contributing

Contributions are welcome to improve the benchmark engine or expand the trait library.

1. **Adding Rubrics:** Create a new JSON file in the `rubrics/` folder following the format above. Ensure scoring criteria are specific, behavioral, and don't mention the trait name.
2. **Improving the Engine:** Submit pull requests for features like new metrics, better judging, or UI enhancements.
3. **Reporting Issues:** Use the GitHub issue tracker.

## Project Structure

- [main.py](main.py): Core benchmarking logic with decomposed blind scoring.
- [build_public.py](build_public.py): Builds optimized data files for the public frontend.
- [app.py](app.py): Streamlit UI for local data exploration.
- [config.json](config.json): Global settings for the benchmark.
- [rubrics/](rubrics/): Trait definitions with behavioral scoring criteria.
- `results/`: Directory containing generated JSON reports and Markdown summaries.
