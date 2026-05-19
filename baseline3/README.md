# Vietnamese Scam Detection - Two-Stage Pipeline (AITeamVN Embedding)

A deep learning system for detecting and classifying Vietnamese scam conversations using **AITeamVN Vietnamese Embedding** (`AITeamVN/Vietnamese_Embedding`) with a two-stage architecture:

1. **Stage 1 (Binary)**: Detect if a conversation is a scam or harmless (streaming per-turn detection)
2. **Stage 2 (Multiclass)**: Classify scam conversations into scenarios (A/B/C/D)

---

## ✨ Features

- **AITeamVN Vietnamese Embedding**: SentenceTransformer model optimized for Vietnamese
- **PyVi tokenization**: Vietnamese word segmentation before encoding
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
cd baseline3/

# Install PyTorch (choose your CUDA version)
# Example for CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

**Manual installation (if needed):**
```bash
pip install torch>=2.0.0 transformers>=4.30.0
pip install sentence-transformers>=2.2.0
pip install pyvi>=0.1
pip install numpy>=1.23.0 scikit-learn>=1.2.0 tqdm>=4.65.0
```

### 3. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from sentence_transformers import SentenceTransformer; print('SentenceTransformer OK')"
python -c "from pyvi import ViTokenizer; print('PyVi OK')"
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
- Use `AITeamVN/Vietnamese_Embedding` encoder

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

---

## 🏗️ Model Architecture

### Overview

```
Input Text (per turn)
    ↓
PyVi Tokenizer (Vietnamese word segmentation)
    ↓
AITeamVN Vietnamese Embedding (SentenceTransformer)
    ↓ [B, T, embed_dim]
Linear Projection → [B, T, 256]
    ↓
CrossTurnAttention (causal)
    ↓ [B, T, 256]
Task-specific Head
```

### AITeamVN Vietnamese Embedding

The **AITeamVN Vietnamese Embedding** (`AITeamVN/Vietnamese_Embedding`) is a SentenceTransformer model specifically trained for Vietnamese text. Key features:

- **Vietnamese-optimized**: Pre-trained on Vietnamese corpus
- **Sentence-level embeddings**: Generates semantic representations for sentences
- **Easy-to-use API**: Simple `.encode()` method
- **PyVi integration**: Works seamlessly with Vietnamese word segmentation

**Usage in this model:**
```python
from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize

model = SentenceTransformer("AITeamVN/Vietnamese_Embedding")

# Tokenize with PyVi
text = "Xin chào Việt Nam"
tokenized = tokenize(text)  # "Xin_chào Việt_Nam"

# Get embeddings
embeddings = model.encode([tokenized])
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

---

## 🔄 Comparison with Other Baselines

| Feature | Baseline1 | Baseline2 | Baseline3 (This) |
|---------|-----------|-----------|------------------|
| **Encoder** | dangvantuan/vietnamese-embedding | BKAI Vietnamese Bi-Encoder | **AITeamVN/Vietnamese_Embedding** |
| **Framework** | SentenceTransformer | Transformers AutoModel | SentenceTransformer |
| **Tokenization** | PyVi | Subword | PyVi |
| **API** | `.encode()` | AutoModel + CLS token | `.encode()` |
| **Dependencies** | sentence-transformers, pyvi | transformers only | sentence-transformers, pyvi |

**Why choose Baseline3:**
- ✅ Simple SentenceTransformer API
- ✅ PyVi word segmentation for Vietnamese
- ✅ AITeamVN model trained specifically for Vietnamese
- ✅ Easy to use and deploy

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

**Output:**
- Binary metrics: Accuracy, F1, AUROC, Detection rate, False alarm rate
- Streaming metrics: Avg detection delay, Early detection stats
- Multiclass metrics: Per-scenario accuracy, Confusion matrix

---

## ⚙️ Hyperparameters

### Recommended Settings

**Small dataset (< 1k dialogues):**
```bash
python train.py \
  --train data/train.json --test data/test.json \
  --bin-epochs 30 --mc-epochs 30 \
  --bin-lr 1e-5 --mc-lr 2e-4
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

---

## 🐛 Troubleshooting

### Issue: "CUDA out of memory"

**Solution:** Reduce batch size or use gradient accumulation:
```bash
python train.py ... --bin-batch-size 4 --mc-batch-size 4 --bin-grad-accum 16
```

### Issue: "Model AITeamVN/Vietnamese_Embedding not found"

**Solution:** The model will be automatically downloaded from HuggingFace on first run. Make sure you have:
1. Internet connection
2. Sufficient disk space (~1GB)

### Issue: Training is very slow

**Solutions:**
1. Check if GPU is being used: `watch -n 1 nvidia-smi`
2. Reduce `max_turns` or `max_turn_len`
3. Disable augmentation: `--no-truncate-aug`

---

## 📖 About AITeamVN

AITeamVN provides Vietnamese NLP models and tools optimized for Vietnamese language processing tasks.

**Model card:** https://huggingface.co/AITeamVN/Vietnamese_Embedding

---

## 📝 Citation

If you use this code, please cite:

```bibtex
@inproceedings{scamstream2026,
  title={Vietnamese Scam Detection with Two-Stage Streaming Pipeline},
  author={Your Name},
  booktitle={EMNLP},
  year={2026}
}
```

---

**Happy training! 🚀**
