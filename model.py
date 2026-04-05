"""
model.py - ML logic for Text Summarization and Language Translation
Uses HuggingFace Transformers (BART for summarization, NLLB for translation)
"""

import logging
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ── Text Summarization ────────────────────────────────────────

def load_summarizer():
    """Load BART summarization model from HuggingFace."""
    try:
        from transformers import BartForConditionalGeneration, BartTokenizer
        logging.info("Loading BART model...")
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
        logging.info("BART model loaded.")
        return tokenizer, model
    except Exception as e:
        logging.error(f"Failed to load summarizer: {e}")
        raise


def summarize_text(text: str, tokenizer, model,
                   min_length=20, max_length=100,
                   repetition_penalty=5.0, length_penalty=0.3) -> str:
    """Summarize input text using BART."""
    try:
        logging.info("Summarizing text...")
        inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(
            inputs["input_ids"],
            min_length=min_length,
            max_length=max_length,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            num_beams=4,
            early_stopping=True
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        logging.info("Summarization complete.")
        return summary
    except Exception as e:
        logging.error(f"Summarization failed: {e}")
        raise


# ── Language Translation ──────────────────────────────────────

# Supported language pairs for NLLB
LANGUAGE_CODES = {
    "French": "fra_Latn",
    "Spanish": "spa_Latn",
    "German": "deu_Latn",
    "Italian": "ita_Latn",
    "Portuguese": "por_Latn",
    "Arabic": "arb_Arab",
    "Chinese (Simplified)": "zho_Hans",
    "Japanese": "jpn_Jpan",
    "Korean": "kor_Hang",
    "Hindi": "hin_Deva",
    "Russian": "rus_Cyrl",
    "English": "eng_Latn",
}


def load_translator():
    """Load Facebook NLLB translation model from HuggingFace."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

        logging.info("Loading NLLB translation model...")
        model_name = "facebook/nllb-200-distilled-600M"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        device = 0 if torch.cuda.is_available() else -1
        translator = pipeline("translation", model=model, tokenizer=tokenizer, device=device)
        logging.info("NLLB model loaded.")
        return translator
    except Exception as e:
        logging.error(f"Failed to load translator: {e}")
        raise


def translate_text(translator, text: str, src_lang: str, tgt_lang: str,
                   max_length=200) -> str:
    """Translate text from source to target language."""
    try:
        logging.info(f"Translating from {src_lang} to {tgt_lang}...")
        result = translator(
            text,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            max_length=max_length
        )
        translation = result[0]["translation_text"]
        logging.info("Translation complete.")
        return translation
    except Exception as e:
        logging.error(f"Translation failed: {e}")
        raise


def translate_batch(translator, samples: list, src_lang: str, tgt_lang: str) -> list:
    """Translate a batch of sentences. Each sample is a dict with 'src' and 'ref'."""
    try:
        predictions = []
        for sample in samples:
            translated = translate_text(translator, sample["src"], src_lang, tgt_lang)
            predictions.append(translated)
        return predictions
    except Exception as e:
        logging.error(f"Batch translation failed: {e}")
        raise


def calculate_bleu(predictions: list, references: list) -> float:
    """Calculate BLEU score between predictions and reference translations."""
    try:
        import sacrebleu
        refs = [references]  # sacrebleu expects list of lists
        bleu = sacrebleu.corpus_bleu(predictions, refs)
        logging.info(f"BLEU Score: {bleu.score:.2f}")
        return round(bleu.score, 2)
    except Exception as e:
        logging.error(f"BLEU calculation failed: {e}")
        raise


def interpret_bleu(score: float) -> str:
    """Return a human-readable interpretation of a BLEU score."""
    if score >= 90:
        return "🏆 Perfect / Near-human quality"
    elif score >= 60:
        return "✅ Very high quality — fluent and accurate"
    elif score >= 40:
        return "👍 Good quality — some fluency issues"
    elif score >= 20:
        return "⚠️ Understandable but not fluent"
    else:
        return "❌ Poor translation — broken or nonsensical"
