# PsycheBench

PsycheBench is an automated evaluation framework designed to assess the therapeutic capabilities and psychological traits of Large Language Models (LLMs). It moves beyond standard benchmarks by simulating realistic patient scenarios and using an expert Judge Model to score responses based on clinical psychological rubrics.

## Features

- **Clinical Depth:** Evaluates models on traits like ADHD, Anxiety, Codependency, and Professional Boundaries.
- **Expert Judging:** Uses high-parameter models (e.g., GPT-5.2) to provide nuanced scoring and reasoning.
- **Visual Analytics:** Includes a Streamlit dashboard for comparing model performance, token usage, and costs.
- **Extensible:** Easily add new clinical scenarios by dropping JSON files into the rubrics folder.
- **Caching System:** Saves results to avoid redundant API calls and reduce costs.

## AI Notes and Disclaimer

PsycheBench is a research tool intended for evaluating the conversational patterns and simulated traits of AI models. 

- **Not a Clinical Tool:** This software is not a diagnostic tool and should not be used to assess human mental health.
- **Model Limitations:** AI models do not have feelings, consciousness, or mental health conditions. Scores reflect the model's ability to simulate or adhere to specific personas and rubrics.
- **Judge Bias:** The evaluation is dependent on the "Judge Model" specified in the configuration. Different judges may yield different results.

## AI Usage in Development

This project was developed with the assistance of AI coding tools. AI was utilized for scaffolding the benchmark engine, generating initial clinical test cases, and refining the data visualization logic in the dashboard.

## Getting Started

### Prerequisites

- Python 3.10+
- An OpenAI-compatible API Key

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/PsycheBench.git
   cd PsycheBench
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment:
   Create a .env file in the root directory:
   ```env
   API_KEY=your_api_key_here
   BASE_URL=https://api.your-provider.com/v1
   ```

### Configuration

Modify [config.json](config.json) to customize the benchmark:
- `judge_model_id`: The model used to evaluate responses.
- `multiplier`: Number of iterations for each test case to ensure consistency.
- `models`: List of subject models to be tested.

## Usage

### Run the Benchmark

To evaluate the models defined in the configuration:
```bash
python main.py
```
Use the `--no-cache` flag to force a fresh run without using previous results.

### View Results

Launch the interactive dashboard to explore scores, reasoning, and metrics:
```bash
streamlit run app.py
```

## Rubric Format

Test sets are stored in the [rubrics/](rubrics/) directory as JSON files. Each file follows this structure:

```json
{
  "trait_key": {
    "description": "Description of the trait being tested.",
    "rubric": "Scoring instructions for the judge (e.g., 0-10 scale).",
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

Contributions are welcome to improve the benchmark engine or expand the clinical test sets.

1. **Adding Rubrics:** Create a new JSON file in the `rubrics/` folder following the format above.
2. **Improving the Engine:** Submit pull requests for features like new metrics, better judging prompts, or UI enhancements.
3. **Reporting Issues:** Use the GitHub issue tracker to report bugs or suggest improvements.
4. **AI-Assisted Contributions:** If using AI to generate code or rubrics, please review all AI-generated content for errors and ensure it aligns with project goals.

## Project Structure

- [main.py](main.py): Core benchmarking logic and API orchestration.
- [app.py](app.py): Streamlit UI for data exploration and visualization.
- [config.json](config.json): Global settings for the benchmark.
- [rubrics/](rubrics/): Clinical test definitions and scoring criteria.
- `results/`: Directory containing generated JSON reports and Markdown summaries.
