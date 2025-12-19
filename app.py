import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="TherapyBench UI", layout="wide")

st.title("🏥 TherapyBench Explorer")

results_dir = Path("results")
if not results_dir.exists() or not list(results_dir.glob("*.json")):
    st.warning("No results found in `results/`. Run the benchmark first.")
else:
    files = sorted([f.name for f in results_dir.glob("*.json")], reverse=True)
    selected_file = st.sidebar.selectbox("Select Report", files)

    with open(results_dir / selected_file, 'r') as f:
        data = json.load(f)

    trait_meta = data["meta"].get("traits", {})

    # Generate consistent color map for models
    model_list = list(data["models"].keys())
    colors = px.colors.qualitative.Plotly
    model_colors = {model: colors[i % len(colors)] for i, model in enumerate(model_list)}

    st.sidebar.divider()
    page = st.sidebar.radio("Navigation", ["Global Overview", "Model Details"])

    if page == "Global Overview":
        st.header("📊 Global Benchmark Overview")
        
        # Prepare Score Data
        score_records = []
        for model, m_data in data["models"].items():
            for trait, score in m_data["scores"].items():
                score_records.append({
                    "Model": model,
                    "Trait": trait.replace("avg_", "").title(),
                    "Score": score
                })
        df_scores = pd.DataFrame(score_records)

        # Separate Bar Charts for Each Trait
        st.subheader("🎯 Trait Performance")
        traits = sorted(df_scores["Trait"].unique())
        for trait in traits:
            with st.container(border=True):
                st.markdown(f"### {trait}")
                
                # Match trait name back to metadata key (lowercase)
                meta_key = trait.lower().replace(" ", "_")
                desc = trait_meta.get(meta_key, {}).get("description", "No description available.")
                st.caption(desc)
                
                df_trait = df_scores[df_scores["Trait"] == trait].sort_values(by="Score", ascending=False)
                fig = px.bar(
                    df_trait, x="Model", y="Score", color="Model",
                    color_discrete_map=model_colors,
                    title=f"Scores for {trait} (Higher = More symptomatic)",
                    labels={"Score": "Score (0-10)"},
                    template="plotly_dark"
                )
                fig.update_layout(hovermode="x")
                st.plotly_chart(fig, use_container_width=True)

        # Token Usage Chart
        st.divider()
        st.subheader("🎫 Token Usage Analysis")
        
        token_stats = []
        for model, m_data in data["models"].items():
            in_tokens = sum(t["subject_usage"]["prompt_tokens"] for t in m_data["traces"])
            out_tokens = sum(t["subject_usage"]["completion_tokens"] for t in m_data["traces"])
            token_stats.append({
                "Model": model,
                "Input": in_tokens,
                "Output": out_tokens,
                "Total": in_tokens + out_tokens
            })
        
        df_tokens = pd.DataFrame(token_stats).sort_values(by="Total", ascending=False)
        
        fig_tokens = px.bar(
            df_tokens, x="Model", y="Total", color="Model",
            color_discrete_map=model_colors,
            hover_data=["Input", "Output", "Total"],
            title="Total Token Usage (Sorted by Total)",
            template="plotly_dark"
        )
        
        fig_tokens.update_layout(hovermode="x")
        st.plotly_chart(fig_tokens, use_container_width=True)
        st.info("💡 Use the sidebar to navigate to 'Model Details' for specific response logs.")

    else:
        selected_model = st.selectbox("Select Model to Inspect", list(data["models"].keys()))
        model_data = data["models"][selected_model]
        
        st.header(f"🧠 Model Detail: {selected_model}")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Latency", f"{model_data['metrics']['avg_latency']:.2f}s")
        col2.metric("Total Subject Tokens", model_data['metrics']['total_subject_tokens'])
        col3.metric("Total Judge Tokens", model_data['metrics']['total_judge_tokens'])

        st.divider()
        st.subheader("Individual Test Traces")
        
        for trace in model_data["traces"]:
            with st.expander(f"[{trace['trait'].upper()}] Score: {trace['score']}/10 - {trace['test_id']}"):
                st.markdown("**Prompt:**")
                st.code(trace.get('prompt', 'N/A'))
                st.markdown("**Response:**")
                st.info(trace['subject_response'])
                st.markdown("**Judge Reasoning:**")
                st.warning(trace['judge_reasoning'])
                st.caption(f"Latency: {trace['latency']:.2f}s | Tokens: {trace['subject_usage']['prompt_tokens']} in, {trace['subject_usage']['completion_tokens']} out")
