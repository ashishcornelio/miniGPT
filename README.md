# MiniGPT Python Training Project

A compact end-to-end Python code generation pipeline using a MiniGPT-style transformer.

## What this project does

- downloads and extracts Python code datasets from Hugging Face
- combines and prepares text datasets for tokenizer and model training
- trains a small transformer-based GPT model on Python code
- generates Python code from a trained checkpoint
- provides a Flask web UI for dataset prepare/train/generate control

## Key files

- `download_datasets.py`: downloads multiple open-source Hugging Face code datasets and saves them to `python_training_data.txt`
- `prepare_dataset.py`: merges local and downloaded text sources into `combined_training_data.txt` and creates `training_corpus.txt`
- `train_complete.py`: full tokenizer + model training pipeline, checkpointing to `checkpoints/`
- `generate_with_model.py`: loads a trained model and tokenizer to generate Python code from a text prompt
- `app.py`: Flask web interface for training, generation, and dataset management
- `delete_trained_data.py`: removes `checkpoints/`, `logs/`, and `tokenizer/`
- `requirements.txt`: Python dependencies used by the project

## Prerequisites

- Python 3.12 (project includes `venv312/` for local environment)
- CUDA-capable GPU recommended for training, but CPU fallback is available
- `pip` for installing dependencies

## Setup

1. Activate the project virtual environment if available:

```powershell
.\venv312\Scripts\activate
```

2. Install required Python packages:

```powershell
pip install -r requirements.txt
```

## Usage

### 1. Download datasets

```powershell
python download_datasets.py
```

This creates `python_training_data.txt` by extracting code snippets from selected Hugging Face datasets.

### 2. Combine and prepare data

```powershell
python prepare_dataset.py
```

This merges available sources into `combined_training_data.txt` and writes a cleaned corpus to `training_corpus.txt`.

### 3. Train the model

```powershell
python train_complete.py --data combined_training_data.txt --epochs 3 --batch-size 4 --lr 1e-4 --max-length 256 --chunk-size 500000
```

- Checkpoints are saved under `checkpoints/`
- Training logs are saved under `logs/`
- The final model is saved as `checkpoints/minigpt_final.pt`

### 4. Generate code

```powershell
python generate_with_model.py --model checkpoints/minigpt_final.pt --prompt "def " --length 100 --temp 0.7
```

### 5. Run the web UI

```powershell
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

The web UI lets you:
- prepare the dataset
- start/monitor training
- generate code
- delete datasets and trained data safely

### 6. Delete trained artifacts

```powershell
python delete_trained_data.py
```

## Notes

- If `python_training_data.txt` is not present, `prepare_dataset.py` will combine whatever local sources exist.
- The project uses a `ByteLevelBPETokenizer` with `vocab_size=30000`.
- Training on CPU may be slow, so GPU is strongly recommended if available.

## Directory overview

- `checkpoints/`: saved model checkpoints
- `logs/`: training log files
- `tokenizer/`: tokenizer vocabulary and merge files
- `tokenizer/merges.txt`, `tokenizer/vocab.json`: tokenizer assets generated during training

## License

This repository does not include a license file. Add one if you intend to share or publish the code.
