# 🤖 Transformer NLP Apps — Text Summarization & Translation

A Streamlit web app combining **Text Summarization** (BART) and **Language Translation with BLEU Evaluation** (NLLB-200).

## What This App Does

| Section | Model | What it does |
|---|---|---|
| How Transformers Work | — | Explains self-attention, encoder-decoder, BLEU |
| Text Summarization | BART (facebook/bart-large-cnn) | Summarizes long text into short form |
| Language Translation | NLLB-200 (facebook/nllb-200-distilled-600M) | Translates between 200 languages |
| BLEU Score Evaluator | NLLB-200 + sacrebleu | Translates sentences and scores quality |

## How to Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/transformer-nlp-app.git
cd transformer-nlp-app
pip install -r requirements.txt
streamlit run app.py
```

> ⚠️ First run downloads ~1.6GB of model weights from HuggingFace. Subsequent runs use cached models.

## Project Structure

```
transformer_app/
├── app.py              ← Streamlit web app (4 sections)
├── model.py            ← BART summarizer + NLLB translator + BLEU scorer
├── utils.py            ← Charts (BLEU gauge, compression stats, per-sentence scores)
├── requirements.txt    ← Dependencies
└── README.md           ← This file
```

## Key Concepts

- **BART** — Encoder-Decoder model fine-tuned on CNN/Daily Mail for summarization
- **NLLB-200** — Meta's multilingual translation model supporting 200 languages
- **BLEU Score** — Standard metric for evaluating translation quality (0–100)
- **Self-Attention** — Core mechanism allowing Transformers to understand word relationships
- **Encoder-Decoder** — Architecture where encoder reads input, decoder writes output

## ⚠️ Deployment Note

These models are large (~600MB–1.6GB). For Streamlit Cloud free tier:
- Models download on first startup (takes 2–5 minutes)
- Use **CPU** inference (no GPU needed)
- May hit memory limits on free tier — if so, use HuggingFace Spaces instead

## Course

CST2216 — Machine Learning 2: Advanced Models and Emerging Topics
Algonquin College
