# Vietnamese Scam Detection - Two-Stage Pipeline (BKAI Bi-Encoder)

A deep learning system for detecting and classifying Vietnamese scam conversations using **BKAI Vietnamese Bi-Encoder** (`bkai-foundation-models/vietnamese-bi-encoder`) with a two-stage architecture:

1. **Stage 1 (Binary)**: Detect if a conversation is a scam or harmless (streaming per-turn detection)
2. **Stage 2 (Multiclass)**: Classify scam conversations into scenarios (A/B/C/D)

---

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Data Format](#data-format)
- [Quick Start](#quick-start)
- [Training](#training)
- [Evaluation](#evaluation)
- [Model Architecture](#model-architecture)
- [Differences from Baseline1](#differences-from-baseline1)
- [Troubleshooting](#troubleshooting)

---

## ✨ Features

- **BKAI Vietnamese Bi-Encoder**: State-of-the-art Vietnamese language model from BKAI
- **Two-stage detection pipeline**: Binary → Multiclass
- **Streaming detection**: Per-turn scam probability for early detection
- **Causal attention**: Turn t only sees turns 0..t-1 (realistic streaming scenario)
- **Hybrid loss**: Soft-label + hard-label BCE with patience penalty
- **Contrastive learning**: Supervised contrastive loss for multiclass
- **Data augmentation**: Truncate sub-window augmentation
- **Full fine-tuning**: Gradual encoder unfreezing with gradient checkpointing

---

## 🔧 Installation

### 1. Requirements

- Python 3.8+
- CUDA-capable GPU (recommended, but CPU works)

### 2. Install Dependencies

```bash
# Clone or navigate to the project directory
cd baseline2/

# Install PyTorch (choose your CUDA version)
# Example for CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

**Manual installation (if needed):**
```bash
pip install torch>=2.0.0 transformers>=4.30.0
pip install numpy>=1.23.0 scikit-learn>=1.2.0 tqdm>=4.65.0
```

### 3. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from transformers import AutoModel, AutoTokenizer; print('Transformers OK')"
```

---

## 📊 Data Format

Your training and test data should be JSON files with the following structure:

```json
[
  {
    "label": "scam",
    "scenario": "A",
    "turns": [
      "Xin chào, bạn có muốn vay tiền không?",
      "Có, tôi cần vay.",
      "Bạn cần chuyển 500k phí trước nhé."
    ]
  },
  {
    "label": "harmless",
    "scenario": null,
    "turns": [
      "Hôm nay thời tiết thế nào?",
      "Trời nắng đẹp!"
    ]
  }
]
```

**Required fields:**
- `label`: `"scam"` or `"harmless"`
- `turns`: List of conversation turns (strings)
- `scenario`: Scam scenario letter (`"A"`, `"B"`, `"C"`, `"D"`) or `null` for harmless

---

## 🚀 Quick Start

### Basic Training

Train on your data with default settings:

```bash
python train.py --train data/train.json --test data/test.json
```

This will:
- Train Stage 1 (binary) for 50 epochs
- Train Stage 2 (multiclass) for 50 epochs
- Save models to `outputs/experiment/`
- Use `bkai-foundation-models/vietnamese-bi-encoder` encoder

### Custom Output Directory

```bash
python train.py \
  --train data/train.json \
  --test data/test.json \
  --output-dir outputs/my_experiment
```

### Adjust Training Epochs

```bash
python train.py \
  --train data/train.json \
  --test data/test.json \
  --bin-epochs 30 \
  --mc-epochs 30
```

### Change Learning Rates

```bash
python train.py \
  --train data/train.json \
  --test data/test.json \
  --bin-lr 1e-5 \
  --mc-lr 2e-4
```

---

## 🎯 Training

### Command-Line Arguments

Run `python train.py --help` to see all options:

**Data:**
- `--train PATH` (required): Training JSON file
- `--test PATH` (required): Test JSON file

**Model:**
- `--model-name`: Transformer model (default: `bkai-foundation-models/vietnamese-bi-encoder`)
- `--hidden-dim`: Hidden dimension (default: 256)
- `--max-turn-len`: Max tokens per turn (default: 144)
- `--max-turns`: Max turns per dialogue (default: 20)

**Stage 1 (Binary):**
- `--bin-epochs`: Training epochs (default: 50)
- `--bin-lr`: Learning rate (default: 2e-5)
- `--bin-batch-size`: Batch size (default: 8)
- `--bin-grad-accum`: Gradient accumulation (default: 8)
- `--bin-unfreeze-epoch`: Epoch to unfreeze encoder (default: 3)
- `--bin-patience`: Early stopping patience (default: 5)

**Stage 2 (Multiclass):**
- `--mc-epochs`: Training epochs (default: 50)
- `--mc-lr`: Learning rate (default: 3e-4)
- `--mc-batch-size`: Batch size (default: 8)
- `--mc-grad-accum`: Gradient accumulation (default: 8)
- `--mc-unfreeze-epoch`: Epoch to unfreeze encoder (default: 5)
- `--mc-patience`: Early stopping patience (default: 8)

**Augmentation:**
- `--no-truncate-aug`: Disable truncate augmentation
- `--aug-k`: Number of sub-windows per dialogue (default: 3)

**Other:**
- `--seed`: Random seed (default: 42)
- `--val-ratio`: Validation split ratio (default: 0.20)
- `--binary-threshold`: Binary threshold (default: 0.5)
- `--no-grad-ckpt`: Disable gradient checkpointing

### Training Output

After training, you'll find:

```
outputs/experiment/
├── binary_model/
│   ├── model.pt              # Binary model checkpoint
│   ├── config.json           # Model configuration
│   ├── tokenizer.json        # Tokenizer files
│   └── test_metrics.json     # Test set metrics
└── mc_model/
    ├── model.pt              # Multiclass model checkpoint
    ├── scenario_map.json     # Scenario label mapping
    └── test_metrics.json     # Test set metrics
```

---

## 📈 Evaluation

### Using eval.py

Evaluate a trained model on new data:

```bash
python eval.py \
  --ckpt-dir outputs/experiment \
  --data data/new_test.json \
  --batch-size 16
```

**Arguments:**
- `--ckpt-dir`: Path to checkpoint directory (contains `binary_model/` and `mc_model/`)
- `--data`: JSON file to evaluate
- `--threshold`: Binary threshold (default: 0.5)
- `--batch-size`: Inference batch size (default: 8)
- `--device`: `cuda` or `cpu` (default: auto-detect)
- `--save-json`: Save metrics to JSON file

**Output:**
- Binary metrics: Accuracy, F1, AUROC, Detection rate, False alarm rate
- Streaming metrics: Avg detection delay, Early detection stats
- Multiclass metrics: Per-scenario accuracy, Confusion matrix

---

## 🏗️ Model Architecture

### Overview

```
Input Text (per turn)
    ↓
BKAI Vietnamese Bi-Encoder Tokenizer
    ↓
BKAI Vietnamese Bi-Encoder (768-dim)
    ↓ [B, T, 768]
Linear Projection → [B, T, 256]
    ↓
CrossTurnAttention (causal)
    ↓ [B, T, 256]
Task-specific Head
```

### BKAI Vietnamese Bi-Encoder

The **BKAI Vietnamese Bi-Encoder** (`bkai-foundation-models/vietnamese-bi-encoder`) is a state-of-the-art Vietnamese language model developed by BKAI Foundation. Key features:

- **Pre-trained on Vietnamese corpus**: Optimized for Vietnamese language understanding
- **Bi-encoder architecture**: Efficient for similarity tasks and classification
- **768-dimensional embeddings**: Rich semantic representations
- **Subword tokenization**: Handles Vietnamese-specific morphology

**Usage in this model:**
```python
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bkai-foundation-models/vietnamese-bi-encoder')
model = AutoModel.from_pretrained('bkai-foundation-models/vietnamese-bi-encoder')

# Tokenize
encoded = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Get embeddings (CLS token)
outputs = model(**encoded)
embeddings = outputs.last_hidden_state[:, 0]  # [batch_size, 768]
```

### Stage 1: Binary Classification

- **Input**: Dialogue with T turns
- **Output**: Per-turn scam probability p(t) for t=0..T-1
- **Loss**: Hybrid detection loss
  - First half: Soft-label BCE × 0.1 (evidence gathering)
  - Second half: Hard BCE × weight(t) (commitment phase, weight 1→3)
  - Patience penalty: Penalize p(t) > p(t+1) for scam dialogues

### Stage 2: Multiclass Classification

- **Input**: Full dialogue (scam conversations only)
- **Output**: Scenario prediction (A/B/C/D)
- **Loss**: 0.5 × SupCon + 0.5 × CrossEntropy
  - SupCon: Supervised contrastive learning (pull same-class together)
  - CrossEntropy: Standard classification loss

### CrossTurnAttention

Causal attention mechanism where turn t attends to historical turns 0..t-1:

```python
for t in range(1, T):
    context = MultiheadAttention(query=h[t], keys=h[:t])
    h[t] = LayerNorm(Fusion(h[t], context) + h[t])
```

---

## 🔄 Differences from Baseline1

| Feature | Baseline1 | Baseline2 (This) |
|---------|-----------|------------------|
| **Encoder** | `dangvantuan/vietnamese-embedding` (SentenceTransformer) | `bkai-foundation-models/vietnamese-bi-encoder` (Transformers) |
| **Tokenization** | PyVi + SentenceTransformer | BKAI subword tokenizer |
| **Embedding extraction** | `model.encode()` | `AutoModel` + CLS token |
| **Input format** | Raw text strings | Token IDs + attention masks |
| **Embedding dim** | ~768 (varies) | 768 (fixed) |
| **Fine-tuning** | Full model | Full model |

**When to use Baseline1 vs Baseline2:**

- **Use Baseline1** if:
  - You want sentence-level embeddings optimized for similarity tasks
  - You prefer simple `.encode()` API
  - You need word segmentation with PyVi

- **Use Baseline2** if:
  - You want state-of-the-art Vietnamese language understanding from BKAI
  - You prefer standard Transformers API
  - You want subword tokenization for better OOV handling

---

## ⚙️ Hyperparameters

### Recommended Settings

**Small dataset (< 1k dialogues):**
```bash
python train.py \
  --train data/train.json --test data/test.json \
  --bin-epochs 30 --mc-epochs 30 \
  --bin-lr 1e-5 --mc-lr 2e-4 \
  --bin-patience 8 --mc-patience 10
```

**Medium dataset (1k-10k dialogues):**
```bash
python train.py \
  --train data/train.json --test data/test.json \
  --bin-epochs 50 --mc-epochs 50 \
  --bin-lr 2e-5 --mc-lr 3e-4
```

**Large dataset (> 10k dialogues):**
```bash
python train.py \
  --train data/train.json --test data/test.json \
  --bin-epochs 20 --mc-epochs 20 \
  --bin-lr 3e-5 --mc-lr 5e-4 \
  --bin-batch-size 16 --mc-batch-size 16
```

### Memory Optimization

If you run out of memory:

1. **Reduce batch size:**
   ```bash
   --bin-batch-size 4 --mc-batch-size 4
   ```

2. **Increase gradient accumulation:**
   ```bash
   --bin-grad-accum 16 --mc-grad-accum 16
   ```

3. **Disable gradient checkpointing** (faster but uses more memory):
   ```bash
   --no-grad-ckpt
   ```

4. **Reduce sequence length:**
   ```bash
   --max-turn-len 100 --max-turns 15
   ```

---

## 🐛 Troubleshooting

### Issue: "Import could not be resolved"

**Solution:** Make sure you're in the correct directory and installed dependencies:
```bash
cd baseline2/
pip install -r requirements.txt
python train.py --help
```

### Issue: "CUDA out of memory"

**Solution:** Reduce batch size or use gradient accumulation:
```bash
python train.py ... --bin-batch-size 4 --mc-batch-size 4 --bin-grad-accum 16
```

### Issue: "Model bkai-foundation-models/vietnamese-bi-encoder not found"

**Solution:** The model will be automatically downloaded from HuggingFace on first run. Make sure you have:
1. Internet connection
2. Sufficient disk space (~1-2GB)
3. HuggingFace access token (if model is gated)

If the model is gated, you may need to:
```bash
# Login to HuggingFace
huggingface-cli login

# Or set token
export HUGGING_FACE_HUB_TOKEN=your_token_here
```

### Issue: "No scam scenarios found in training data"

**Solution:** Check your JSON format. Make sure scam dialogues have `"scenario": "A"` (or B/C/D):
```json
{"label": "scam", "scenario": "A", "turns": [...]}
```

### Issue: Training is very slow

**Solutions:**
1. Check if GPU is being used: `watch -n 1 nvidia-smi`
2. Reduce `max_turns` or `max_turn_len`:
   ```bash
   --max-turns 15 --max-turn-len 100
   ```
3. Disable augmentation:
   ```bash
   --no-truncate-aug
   ```

### Issue: Poor validation performance

**Solutions:**
1. **Increase training data** or enable augmentation (default enabled)
2. **Tune learning rate:**
   ```bash
   --bin-lr 1e-5 --mc-lr 2e-4
   ```
3. **Adjust unfreeze epoch** (try unfreezing encoder later):
   ```bash
   --bin-unfreeze-epoch 5 --mc-unfreeze-epoch 8
   ```
4. **Increase patience** to avoid early stopping too soon:
   ```bash
   --bin-patience 10 --mc-patience 15
   ```

---

## 📚 Additional Scripts

### exp1.py - Prompt vs VisCam

Compare synthetic data generation methods:
```bash
python exp1.py --train-source vanilla   # Prompt-based
python exp1.py --train-source viscam    # VisCam pipeline
python exp1.py --train-source both      # Combined
```

### exp2.py & exp3.py

See respective files for experiment-specific configurations.

---

## 📖 About BKAI Foundation

BKAI (Bach Khoa AI) is the AI research lab at Hanoi University of Science and Technology. Their Vietnamese language models are trained on large-scale Vietnamese corpora and optimized for Vietnamese NLP tasks.

**Model card:** https://huggingface.co/bkai-foundation-models/vietnamese-bi-encoder

---

## 📝 Citation

If you use this code or the BKAI model, please cite:

```bibtex
@inproceedings{scamstream2026,
  title={Vietnamese Scam Detection with Two-Stage Streaming Pipeline},
  author={Your Name},
  booktitle={EMNLP},
  year={2026}
}

@misc{bkai-vietnamese-biencoder,
  title={BKAI Vietnamese Bi-Encoder},
  author={BKAI Foundation},
  year={2023},
  url={https://huggingface.co/bkai-foundation-models/vietnamese-bi-encoder}
}
```

---

## 📧 Contact

For questions or issues, please open a GitHub issue or contact [your email].

---

**Happy training! 🚀**
