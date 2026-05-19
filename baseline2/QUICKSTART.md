# Quick Start Guide - Baseline2 (BKAI Bi-Encoder)

Get started in 5 minutes!

## 1. Install Dependencies (2 minutes)

```bash
cd baseline2/

# Install PyTorch (choose your platform)
# For CUDA 11.8:
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CPU-only:
pip install torch

# Install other packages
pip install -r requirements.txt
```

## 2. Prepare Your Data (1 minute)

Create two JSON files with your training and test data.

**Format:**
```json
[
  {
    "label": "scam",
    "scenario": "A",
    "turns": [
      "Chào bạn, bạn có cần vay tiền không?",
      "Có, tôi cần.",
      "Hãy chuyển 500k phí trước nhé."
    ]
  },
  {
    "label": "harmless",
    "scenario": null,
    "turns": [
      "Hôm nay thời tiết như thế nào?",
      "Trời nắng đẹp!"
    ]
  }
]
```

## 3. Train! (2 minutes to start)

```bash
python train.py --train path/to/train.json --test path/to/test.json
```

Done! The model will train using **BKAI Vietnamese Bi-Encoder** and save to `outputs/experiment/`

---

## What's Different from Baseline1?

| Feature | Baseline1 | Baseline2 |
|---------|-----------|-----------|
| Encoder | dangvantuan/vietnamese-embedding | **BKAI Vietnamese Bi-Encoder** |
| Type | SentenceTransformer | Transformers AutoModel |
| Tokenization | PyVi word segmentation | Subword tokenization |

---

## Common Commands

**Quick test (fewer epochs):**
```bash
python train.py \
  --train data/train.json \
  --test data/test.json \
  --bin-epochs 10 \
  --mc-epochs 10
```

**Custom output:**
```bash
python train.py \
  --train data/train.json \
  --test data/test.json \
  --output-dir results/bkai_exp1
```

**Low memory settings:**
```bash
python train.py \
  --train data/train.json \
  --test data/test.json \
  --bin-batch-size 4 \
  --mc-batch-size 4 \
  --bin-grad-accum 16
```

---

## What Happens During Training?

1. **Model Loading**: Downloads BKAI Vietnamese Bi-Encoder (~1-2GB, first time only)
2. **Data Loading**: Reads your JSON files
3. **Stage 1 (Binary)**: Trains scam vs harmless classifier (50 epochs)
4. **Stage 2 (Multiclass)**: Trains scenario classifier on scam data (50 epochs)
5. **Evaluation**: Tests on your test set
6. **Save**: Stores models in `outputs/experiment/`

---

## Next Steps

- See `README.md` for full documentation
- Run `python train.py --help` for all options
- Use `python eval.py` to evaluate on new data

**Need help?** Check the Troubleshooting section in README.md
