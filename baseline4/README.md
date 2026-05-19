# Vietnamese Scam Detection - Two-Stage Pipeline (XLM-RoBERTa-base)

A deep learning system for detecting and classifying Vietnamese scam conversations using **XLM-RoBERTa-base** (`xlm-roberta-base`) - a multilingual transformer model with two stages:

1. **Stage 1 (Binary)**: Detect if a conversation is a scam or harmless (streaming per-turn detection)
2. **Stage 2 (Multiclass)**: Classify scam conversations into scenarios (A/B/C/D)

---

## ✨ Features

- **XLM-RoBERTa-base**: Multilingual transformer pre-trained on 100 languages including Vietnamese
- **Cross-lingual capabilities**: Benefits from multilingual pre-training
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
cd baseline4/

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
- Use `xlm-roberta-base` encoder

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

## 🏗️ Model Architecture

### Overview

```
Input Text (per turn)
    ↓
XLM-RoBERTa Tokenizer (SentencePiece)
    ↓
XLM-RoBERTa-base Encoder (270M parameters)
    ↓ [B, T, 768]
Linear Projection → [B, T, 256]
    ↓
CrossTurnAttention (causal)
    ↓ [B, T, 256]
Task-specific Head
```

### XLM-RoBERTa-base

**XLM-RoBERTa-base** is a multilingual masked language model pre-trained on 2.5TB of CommonCrawl data in 100 languages. Key features:

- **Multilingual pre-training**: Covers 100 languages including Vietnamese
- **Cross-lingual transfer**: Benefits from multilingual representations
- **SentencePiece tokenization**: Language-agnostic subword tokenization
- **270M parameters**: Large model with strong generalization
- **RoBERTa architecture**: Dynamic masking, no NSP task, larger batches

**Why XLM-RoBERTa for Vietnamese:**
- ✅ Strong multilingual capabilities
- ✅ Pre-trained on Vietnamese web data
- ✅ Cross-lingual transfer from high-resource languages
- ✅ No language-specific preprocessing needed
- ✅ Widely used and well-supported

**Usage in this model:**
```python
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
model = AutoModel.from_pretrained('xlm-roberta-base')

# Tokenize
encoded = tokenizer(text, padding=True, truncation=True, return_tensors='pt')

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
  - SupCon: Supervised contrastive learning
  - CrossEntropy: Standard classification loss

---

## 🔄 Comparison with Other Baselines

| Feature | Baseline1 | Baseline2 | Baseline3 | Baseline4 (This) |
|---------|-----------|-----------|-----------|------------------|
| **Encoder** | dangvantuan/vietnamese-embedding | BKAI Vietnamese Bi-Encoder | AITeamVN/Vietnamese_Embedding | **XLM-RoBERTa-base** |
| **Framework** | SentenceTransformer | Transformers | SentenceTransformer | Transformers |
| **Languages** | Vietnamese | Vietnamese | Vietnamese | **100 languages** |
| **Tokenization** | PyVi | Subword | PyVi | SentencePiece |
| **Parameters** | ~100M | ~100M | ~100M | **270M** |
| **Pre-training** | Vietnamese | Vietnamese | Vietnamese | **Multilingual** |

**Why choose Baseline4 (XLM-RoBERTa):**
- ✅ **Larger model** (270M parameters) with stronger representation capacity
- ✅ **Multilingual pre-training** benefits from cross-lingual transfer
- ✅ **Language-agnostic** tokenization (SentencePiece)
- ✅ **Well-established** model from Facebook AI with extensive research support
- ✅ **Strong performance** on multilingual benchmarks

**Trade-offs:**
- ⚠️ Larger model size (~1GB)
- ⚠️ Slower inference than smaller models
- ⚠️ Higher GPU memory requirements

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

XLM-RoBERTa-base is larger than other baselines. If you run out of memory:

1. **Reduce batch size:**
   ```bash
   --bin-batch-size 4 --mc-batch-size 4
   ```

2. **Increase gradient accumulation:**
   ```bash
   --bin-grad-accum 16 --mc-grad-accum 16
   ```

3. **Use gradient checkpointing** (default enabled):
   ```bash
   # Already enabled by default, but you can disable with:
   --no-grad-ckpt  # NOT recommended for this large model
   ```

4. **Reduce sequence length:**
   ```bash
   --max-turn-len 100 --max-turns 15
   ```

5. **Use mixed precision training** (if PyTorch supports):
   ```python
   # This would require code modification
   from torch.cuda.amp import autocast, GradScaler
   ```

---

## 🐛 Troubleshooting

### Issue: "CUDA out of memory"

**Solution:** XLM-RoBERTa is larger than other models. Reduce batch size:
```bash
python train.py ... --bin-batch-size 2 --mc-batch-size 2 --bin-grad-accum 32
```

### Issue: "Model xlm-roberta-base not found"

**Solution:** The model will be automatically downloaded from HuggingFace (~1GB). Make sure you have:
1. Internet connection
2. Sufficient disk space (~2GB)
3. Optional: HuggingFace token for faster downloads
   ```bash
   huggingface-cli login
   ```

### Issue: Training is very slow

**Solutions:**
1. XLM-RoBERTa is larger and slower. This is expected.
2. Check GPU utilization: `watch -n 1 nvidia-smi`
3. Reduce sequence length: `--max-turn-len 100`
4. Use smaller batch size with more accumulation

### Issue: Poor validation performance

**Solutions:**
1. XLM-RoBERTa may need **more training epochs** than smaller models
2. Try **lower learning rate**: `--bin-lr 1e-5 --mc-lr 2e-4`
3. **Unfreeze encoder later**: `--bin-unfreeze-epoch 5 --mc-unfreeze-epoch 8`
4. **Increase patience**: `--bin-patience 10 --mc-patience 15`

---

## 📖 About XLM-RoBERTa

**Paper:** [Unsupervised Cross-lingual Representation Learning at Scale (Conneau et al., 2020)](https://arxiv.org/abs/1911.02116)

**Model card:** https://huggingface.co/xlm-roberta-base

**Key innovations:**
- Pre-trained on 2.5TB of CommonCrawl in 100 languages
- No language-specific preprocessing or embeddings
- Outperforms mBERT and previous multilingual models
- Strong zero-shot cross-lingual transfer

---

## 📝 Citation

If you use this code or XLM-RoBERTa, please cite:

```bibtex
@inproceedings{scamstream2026,
  title={Vietnamese Scam Detection with Two-Stage Streaming Pipeline},
  author={Your Name},
  booktitle={EMNLP},
  year={2026}
}

@inproceedings{conneau2020unsupervised,
  title={Unsupervised Cross-lingual Representation Learning at Scale},
  author={Conneau, Alexis and Khandelwal, Kartikay and Goyal, Naman and Chaudhary, Vishrav and Wenzek, Guillaume and Guzm{\'a}n, Francisco and Grave, Edouard and Ott, Myle and Zettlemoyer, Luke and Stoyanov, Veselin},
  booktitle={ACL},
  year={2020}
}
```

---

## 📧 Contact

For questions or issues, please open a GitHub issue or contact [your email].

---

**Happy training! 🚀**
