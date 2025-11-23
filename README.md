
# Multimodal DR Diagnosis System

This is a multimodal system for diagnosing diabetic retinopathy (DR) using fundus images and RAG-enhanced reasoning.

## Project Structure

- `app/`: Main application code.
  - `core/`: Core components (LLM, RAG, Vision models).
  - `api/`: API endpoints and schemas.
  - `services/`: Business logic services.
  - `config/`: Configuration settings.
  - `utils/`: Utility functions.
- `data/`: Data storage (Vector DB, Knowledge Base).
- `models/`: Model weights (ResNet, Qwen-VL, R1-7B).
- `scripts/`: Utility scripts.
- `web/`: Frontend interface.
- `run_service.py`: Entry point to start the service.

## Prerequisites

- Python 3.8+
- CUDA-enabled GPU (recommended for models)

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure model weights are placed in `models/` directory as configured in `app/config/settings.py`.

3. Initialize Vector Database (if needed):
   ```bash
   python scripts/init_vector_db.py
   ```

## Running the Service

Start the complete service (Gateway + Models) with a single command:

```bash
python run_service.py
```

The API will be available at `http://localhost:8000`.
API Documentation: `http://localhost:8000/docs`.

## Configuration

Configuration is managed in `app/config/settings.py`. You can override settings using environment variables or a `.env` file.

## Frontend

Open `web/frontend.html` in your browser to interact with the system.