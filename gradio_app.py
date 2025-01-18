import gradio as gr
import os
import logging
from Rag_pipeline_evaluation import PipelineEvaluator  # 假設 PipelineEvaluator 儲存在 pipeline_evaluator.py 中
from Retriever_evaluation import RetrieverEvaluator  # 假設 RetrieverEvaluator 儲存在 retriever_evaluator.py 中
from config import (
    COHERE_API_KEY,
    VOYAGE_API_KEY,
    OPENAI_API_KEY,
    CLAUDE_API_KEY,
    GEMINI_API_KEY,
    MONGODB_URI
)

# 初始化 Evaluators
config = {
    "COHERE_API_KEY": COHERE_API_KEY,
    "VOYAGE_API_KEY": VOYAGE_API_KEY,
    "OPENAI_API_KEY": OPENAI_API_KEY,
    "CLAUDE_API_KEY": CLAUDE_API_KEY,
    "GEMINI_API_KEY": GEMINI_API_KEY,
    "MONGODB_URI": MONGODB_URI,
}

pipeline_evaluator = PipelineEvaluator(config)
retriever_evaluator = RetrieverEvaluator()

# 定義功能函數
def evaluate_pipeline(dataset_path, emb_model, re_model, emb_top_n, re_top_n, llm):
    try:
        if not os.path.exists(dataset_path):
            return "Dataset file not found!", None

        results, plot_path = pipeline_evaluator.evaluate_pipeline(dataset_path, emb_model, re_model, emb_top_n, re_top_n, llm)
        if results is None or plot_path is None:
            return "Pipeline evaluation failed. Check logs for details."

        result_text = (
            f"Embedding Model: {results['Embedding model'][0]}\n"
            f"Reranking Model: {results['Reranking model'][0]}\n"
            f"Embedding Top K: {results['Emb_top_k'][0]}\n"
            f"Reranking Top K: {results['Re_top_k'][0]}\n"
            f"LLM: {results['LLM'][0]}\n"
            f"Generation Time: {results['Generation Time'][0]:.2f} seconds\n"
            f"Evaluation Time: {results['Evaluation Time'][0]:.2f} seconds\n"
            f"Total Time: {results['Total Time'][0]:.2f} seconds\n"
            f"USD Cost: {results['USD Cost'][0]:.4f} USD\n"
        )
        return result_text, plot_path
    except Exception as e:
        error_message = f"Error during pipeline evaluation: {str(e)}"
        logging.error(error_message)
        return error_message

def evaluate_retriever(src_file, emb_model, emb_top_n, task, re_model, re_top_n, db_state):
    db_state = "yes" if db_switch else "no"
    try:
        retriever_evaluator.run(src_file, emb_model, emb_top_n=int(emb_top_n),  task=task, re_top_n=int(re_top_n), re_model=re_model, db_state=db_state)
        return "Retriever evaluation completed successfully."
    except Exception as e:
        return f"Error: {e}"

# 定義模型選項
embedding_models = [
    "voyage-multilingual-2", 
    "embed-multilingual-v3.0",
    "voyage-3",
    "voyage-3-lite",
    "embed-multilingual-light-v3.0",
    "jina-embeddings-v3",
    "text-embedding-ada-002",
    "text-embedding-3-small",
    "text-embedding-3-large",
    "nvidia/nv-embedqa-e5-v5",
    "BAAI/bge-small-zh-v1.5",
    "BAAI/bge-base-zh-v1.5",
    "BAAI/bge-large-zh-v1.5",
    "intfloat/multilingual-e5-large",
    "intfloat/multilingual-e5-base",
    "intfloat/multilingual-e5-small",
    "dunzhang/stella-large-zh-v3-1792d",
    "infgrad/stella-base-zh-v3-1792d",
    "infgrad/stella-large-zh-v2",
    "infgrad/stella-base-zh-v2"
]
reranking_models = [
    "rerank-2", 
    "rerank-multilingual-v3.0",
    "rerank-2-lite",
    "jina-reranker-v2-base-multilingual",
    "BAAI/bge-reranker-v2-m3",
    "BAAI/bge-reranker-large",
    "BAAI/bge-reranker-base"
]
llm_models = [
    "gpt-4o", 
    "gpt-4o-mini", 
    "claude-3-opus-20240229", 
    "gemini-1.5-pro", 
    "yentinglin/Llama-3-Taiwan-8B-Instruct", 
    "taide/Llama3-TAIDE-LX-8B-Chat-Alpha1",
    "Qwen/Qwen1.5-7B-Chat",
    "MediaTek-Research/Breeze-7B-Instruct-v1_0",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "meta-llama/Meta-Llama-3-8B-Instruct"
]
retriever_tasks = ["only_emb", "emb_and_rerank"]

# Gradio 界面設計
with gr.Blocks() as app:
    gr.Markdown("## Unified Evaluation Platform")
    
    with gr.Tabs():
        # Tab 1: Retriever Selection
        with gr.Tab("Retriever Selection"):
            gr.Markdown("### Evaluate Retriever Models")
            with gr.Row():
                retriever_file_input = gr.File(label="Upload Dataset (CSV)", file_types=[".csv"])
            with gr.Row():
                emb_model_input = gr.Dropdown(embedding_models, value="voyage-multilingual-2", label="Select Embedding Model")
                task_input = gr.Dropdown(retriever_tasks, value="only_emb", label="Select Task")
                emb_top_n_input = gr.Number(value=10, label="Embedding Top K", precision=0)
                re_model_input = gr.Dropdown(reranking_models, value="rerank-2", label="Select Reranking Model")
                re_top_n_input = gr.Number(value=3, label="Reranking Top K", precision=0)
            with gr.Row():
                db_switch = gr.Checkbox(label="Enable Database Update (Default: On)", value=True)
            with gr.Row():
                retriever_evaluate_button = gr.Button("Run Retriever Evaluation")
            with gr.Row():
                pipeline_output_text = gr.Textbox(label="Experiment Details", interactive=False)
                pipeline_output_plot = gr.Image(label="RAGAS Scores Plot")
            retriever_evaluate_button.click(
                fn=evaluate_retriever,
                inputs=[retriever_file_input, emb_model_input, emb_top_n_input, task_input, re_model_input, re_top_n_input, db_switch],
                outputs=[pipeline_output_text, pipeline_output_plot],
            )
        
        # Tab 2: LLM Selection
        with gr.Tab("LLM Selection"):
            gr.Markdown("### Test LLM Performance")
            with gr.Row():
                llm_input = gr.Dropdown(llm_models, value="gpt-4o", label="Select LLM Model")
                question_input = gr.Textbox(label="Enter a question", placeholder="Type your question here...")
                context_input = gr.Textbox(label="Enter context", placeholder="Provide additional context here...")
            with gr.Row():
                llm_test_button = gr.Button("Test LLM")
                llm_output = gr.Textbox(label="Output", interactive=False)
            llm_test_button.click(
                fn=lambda model, question, context: f"LLM {model} answered: 'This is a mock response.'",
                inputs=[llm_input, question_input, context_input],
                outputs=llm_output,
            )
        
        with gr.Tab("RAG Pipeline Selection"):
            gr.Markdown("### Evaluate RAG Pipeline")
            with gr.Row():
                pipeline_file_input = gr.File(label="Upload Dataset (CSV)", file_types=[".csv"])
            with gr.Row():
                emb_model_input = gr.Dropdown(embedding_models, value="voyage-multilingual-2", label="Select Embedding Model")
                re_model_input = gr.Dropdown(reranking_models, value="rerank-2", label="Select Reranking Model")
                emb_top_n_input = gr.Number(value=10, label="Embedding Top K", precision=0)
                re_top_n_input = gr.Number(value=3, label="Reranking Top K", precision=0)
                llm_model_input = gr.Dropdown(llm_models, value="gpt-4o", label="Select LLM Model")
            with gr.Row():
                pipeline_evaluate_button = gr.Button("Evaluate Pipeline")
            with gr.Row():
                pipeline_output_text = gr.Textbox(label="Experiment Details", interactive=False, lines=10, max_lines=20)
                pipeline_output_plot = gr.Image(label="RAGAS Scores Plot")  # gr.Image 自动支持路径
            pipeline_evaluate_button.click(
                fn=evaluate_pipeline,
                inputs=[pipeline_file_input, emb_model_input, re_model_input, emb_top_n_input, re_top_n_input, llm_model_input],
                outputs=[pipeline_output_text, pipeline_output_plot],  # 确保两输出
            )


# 啟動介面
if __name__ == "__main__":
    app.launch(server_name="0.0.0.0")
