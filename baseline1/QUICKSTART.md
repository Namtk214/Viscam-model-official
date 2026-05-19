# Quick Start Guide

Get started in 5 minutes!

## 1. Install Dependencies (2 minutes)

```bash
cd baseline1/

# Install PyTorch (choose your platform)
# For CUDA 11.8:
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CPU-only:
pip install torch

# Install other packages
pip install -r requirements.txt
```

## 2. Prepare Your Data (1 minute)

Create two JSON files: `data/train.json` and `data/test.json`

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
python train.py --train data/train.json --test data/test.json
```

Done! The model will train and save to `outputs/experiment/`

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
  --output-dir results/exp1
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

1. **Data Loading**: Reads your JSON files
2. **Stage 1 (Binary)**: Trains scam vs harmless classifier (50 epochs)
3. **Stage 2 (Multiclass)**: Trains scenario classifier on scam data (50 epochs)
4. **Evaluation**: Tests on your test set
5. **Save**: Stores models in `outputs/experiment/`

---

## Next Steps

- See `README.md` for full documentation
- Run `python train.py --help` for all options
- Use `python eval.py` to evaluate on new data

**Need help?** Check the Troubleshooting section in README.md
