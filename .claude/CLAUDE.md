# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a multi-modal medical diagnosis system for Diabetic Retinopathy (DR) that combines computer vision with large language models and retrieval-augmented generation (RAG) to provide intelligent diagnostic reports.

## Common Development Commands

### Setup and Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Download models (ensure these paths exist in ./models/):
# - ResNet50 DR grading model
# - Qwen-VL model for visual language understanding
# - R1-7B fine-tuned model for medical reasoning
```

### Running the Application
```bash
# Start the FastAPI server
python main.py
# Server runs on http://0.0.0.0:8000

# Test with a single image
python run_diagnosis.py path/to/image.jpg

# Open web interface
# Open frontend.html in a browser
```

### Knowledge Base Management
```bash
# Aggregate codebase for AI analysis (utility script)
python aggregate_for_gemini.py
# Generates multimodal_dr_diagnosis_for_gemini.txt
```

## Architecture Overview

The system follows a modular, multi-stage pipeline architecture:

### Core Components
1. **Configuration Layer** (`settings.py`): Pydantic-based settings with environment variable support
2. **Vision Processing** (`vision_processors.py`):
   - `DRGradingModule`: ResNet50-based DR severity classification
   - `QwenVLModule`: Visual-language model for lesion description
3. **LLM Integration** (`llm_loader.py`): Loads R1-7B model with LoRA adapters as LangChain component
4. **RAG Pipeline** (`rag_chain_builder.py`): LangChain LCEL-based retrieval and reasoning chain
5. **API Layer** (`main.py`): FastAPI service orchestrating the full pipeline
6. **Web Interface** (`frontend.html`): Complete frontend with health checks and real-time diagnostics

### Data Flow
1. Image upload → FastAPI endpoint `/diagnose`
2. **Vision Stage**: ResNet50 grading + Qwen-VL lesion description (parallel processing)
3. **RAG Stage**: Retrieve relevant medical knowledge using FAISS vector store
4. **Reasoning Stage**: R1-7B LLM performs Chain-of-Thought reasoning with retrieved context
5. **Output**: Structured JSON report with CoT reasoning, recommendations, and traceability

### Key Design Patterns
- **Global Resource Loading**: Models loaded once at startup in `startup_event()`
- **Modular Vision Processing**: Separate classes for grading and description tasks
- **LangChain Integration**: Uses LCEL (LangChain Expression Language) for composable chains
- **Configuration Management**: Centralized settings with `.env` file support
- **Error Handling**: Comprehensive logging and HTTP exception handling

### Model Paths and Dependencies
- DR Grading: `./models/resnet50_dr_grading.pth` (ResNet50 fine-tuned for 5-class DR grading)
- Vision-Language: `./models/Qwen-VL/` (Qwen-VL model for lesion description)
- Reasoning LLM: `./models/R1-7B-finetuned/` (R1-7B with LoRA for medical CoT reasoning)
- Knowledge Base: `./knowledge_base/` (Medical guidelines in .txt format)
- Vector DB: `./vector_db/` (FAISS index for efficient retrieval)

### RAG System Details
- Uses sentence-transformers embeddings (`all-MiniLM-L6-v2`)
- FAISS vector store with persistent storage
- Recursive character text splitter (500 chars, 50 overlap)
- Top-K retrieval (default: 3 documents)
- Sophisticated prompt template enforcing structured JSON output with CoT reasoning

## Development Notes

### Model Loading
- All models support automatic device detection (CPU/GPU)
- R1-7B model includes LoRA adapter loading with graceful fallback
- Vision models use appropriate transforms and preprocessing

### API Features
- Single `/diagnose` endpoint for complete pipeline
- Health check endpoint for model status monitoring
- Knowledge base reload capability
- Real-time logging accessible via web interface

### Frontend Integration
- Complete HTML interface with Tailwind CSS
- Real-time health monitoring and status indicators
- Image preview and diagnostic result display
- Toggle options for RAG and CoT reasoning
- Log viewing and API configuration persistence