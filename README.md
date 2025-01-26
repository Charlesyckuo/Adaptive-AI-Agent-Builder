# Adaptive AI Agent Builder

---

## Overview
The **Adaptive AI Agent Builder** is a powerful and modular framework designed to help users efficiently build and evaluate AI agents by selecting optimal components. It allows for testing, analyzing, and comparing various retrievers, language models (LLMs), and Retrieval-Augmented Generation (RAG) pipelines. With a focus on adaptability and extensibility, this tool supports both text-based applications and lays the groundwork for future multimodal capabilities.

---

## File Structure
```
.env                     # Environment variables including API keys
config.py                # Configuration file for API keys and system settings
csv_id_processor.py      # Script for processing and assigning unique IDs to dataset rows
db_vector_writing.py     # Handles vector writing to databases
email_notification.py    # Sends notifications with evaluation results via email
emb_plot.py              # Generates embedding-related plots
embedding.py             # Embedding-related services and utilities
evaluation.py            # Evaluates retriever and RAG performance
gradio_app.py            # Launches the user interface with Gradio
llm.py                   # Language model services and utilities
processing_time.py       # Measures and logs processing time for evaluations
Rag_pipeline_evaluation.py  # Core logic for evaluating RAG pipelines
Ragas.py                 # Evaluation script for RAGAS metrics
re_plot.py               # Generates reranking-related plots
Retriever_evaluation.py  # Evaluates retrievers and supports pipeline selection
```

---

## How to Use

### Installation
1. Clone the repository:
   ```bash
   git clone <https://github.com/Charlesyckuo/Adaptive-AI-Agent-Builder>
   ```
2. Navigate to the project directory:
   ```bash
   cd adaptive-ai-agent-builder
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### API Keys Setting
1. Create a `.env` file in the root directory.
2. Fill in your API keys for the services you plan to use (e.g., OpenAI, Cohere, Jina, etc.):
   ```env
   OPENAI_API_KEY=your_openai_api_key
   COHERE_API_KEY=your_cohere_api_key
   VOYAGE_API_KEY=your_voyage_api_key
   ```

### Start User Interface
Run the following command to start the Gradio-based user interface:
```bash
python gradio_app.py
```
This will launch a local web interface for interacting with the tool.

### Input Dataset Constraints
To ensure proper functionality, your dataset must adhere to the following format:

#### Retriever Selection
If you are only using the **retriever selection function**, each row in your dataset must include:
- **Summary:** A concise summary of the context.
- **Question:** A question associated with the summary.
- **Id:** A unique identifier for each context (if two summaries are identical, they should share the same ID).

#### Full Pipeline Evaluation
If you are using the **pipeline evaluation function** (including LLMs and enhancement methods), your dataset must include the above fields and an additional field:
- **Answer:** The ground truth answer corresponding to each question.

#### Missing IDs?
If your dataset lacks the `Id` field, you can generate IDs automatically by running:
```bash
python csv_id_processor.py
```

### Usage
1. **Select Components:** Choose the components to test (e.g., Retriever, LLM, RAG).
2. **Upload Dataset:** Input your test file via the user interface.
3. **Experiment with Combinations:** Create different combinations of models and pipelines.
4. **View Results:** Receive detailed results including:
   - Final **scores** (e.g., RAGAS metrics).
   - **USD costs** incurred.
   - **Time spent** on evaluation.
5. **Make Decisions:** Use the results to balance cost, efficiency, and performance, helping you make the best choice for your organization.

---

## Features

### Email Notification
Don’t wait in front of your device for test results. Configure your email in the interface, and the system will send results to you automatically upon completion.

### Model Extension
Want to experiment with new models? You can easily add more models by extending the following scripts:
- **`llm.py`**: For new LLMs.
- **`embedding.py`**: For new embedding models.

---

## Future Developments
1. **Model Expansion:**
   - Support for additional models across all use cases.
2. **Enhanced Techniques:**
   - Integration of advanced enhancement methods for RAG and AI agent optimization.
3. **Multimodal Capabilities:**
   - Extend support beyond text-based processing to include multimodal models (e.g., images, audio, and video).
4. **Automated Agent Construction:**
   - Use LLMs to dynamically build AI agents tailored to specific applications, guided by the selection tool’s scores and recommendations.

---

## Acknowledgments
This project leverages state-of-the-art APIs and frameworks to provide robust AI agent construction capabilities. Special thanks to the open-source community for their contributions and ongoing innovation.

---

## Contact
For questions, feature requests, or collaboration opportunities, please contact:
- **Email:** [charles.kuo.yc@gmail.com]
- **GitHub:** [https://github.com/Charlesyckuo/Adaptive-AI-Agent-Builder]

Feel free to contribute and share your feedback to help us improve the tool!

