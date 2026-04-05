"""
app.py - Streamlit Web App
Covers: Text Summarization (BART) + Language Translation + BLEU Evaluation (NLLB)
Run with: streamlit run app.py
"""

import streamlit as st
import warnings
warnings.filterwarnings("ignore")

from model import (
    load_summarizer,
    summarize_text,
    load_translator,
    translate_text,
    translate_batch,
    calculate_bleu,
    interpret_bleu,
    LANGUAGE_CODES,
)
from utils import (
    plot_bleu_gauge,
    plot_summary_length_comparison,
    plot_translation_results,
    plot_bleu_interpretation_table,
)

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Transformer NLP Apps",
    layout="wide"
)

st.title("Transformer-Based NLP Apps")
st.markdown("**Text Summarization** with BART and **Language Translation** with NLLB - powered by HuggingFace Transformers.")

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.title("Navigation")
section = st.sidebar.radio("Choose an app:", [
    "How Transformers Work",
    "Text Summarization",
    "Language Translation",
    "BLEU Score Evaluator",
])

# ════════════════════════════════════════════════════════════
# SECTION 1 - How Transformers Work
# ════════════════════════════════════════════════════════════
if section == "How Transformers Work":
    st.header("How Transformers Work")

    st.markdown("""
    ### What is a Transformer?
    A **Transformer** is a type of neural network that processes text using a mechanism called **Attention**.
    Instead of reading words one by one (like older RNN models), Transformers look at **all words at once**
    and figure out which ones matter most for understanding each other.
    """)

    st.subheader("Self-Attention - The Key Idea")
    st.markdown("""
    **Self-Attention** lets the model figure out which words relate to each other in a sentence.

    > *"The animal didn't cross the street because **it** was too tired."*

    When processing the word **"it"**, the model learns that "it" refers to "animal" — not "street".
    This is self-attention in action: the model looks back at all previous words to understand context.
    """)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("Self_Attention.jpg",
                 caption="Self-Attention: 'it' attends to 'animal' and 'street'", use_column_width=True)
    with col2:
        st.markdown("""
        **How it works step by step:**
        1. Each word becomes a vector (list of numbers)
        2. The model computes **attention scores** between every pair of words
        3. Words that are more relevant to each other get higher scores
        4. The model uses these scores to build a rich understanding of meaning

        **Why this is better than older models (RNNs):**
        - RNNs read one word at a time → forget early words in long sentences
        - Transformers read all words at once → no forgetting problem
        - Transformers can run in parallel → much faster to train
        """)

    st.divider()

    st.subheader("Encoder-Decoder Architecture")
    st.markdown("""
    Most translation and summarization models use an **Encoder-Decoder** structure:
    - **Encoder** = reads and understands the input text
    - **Decoder** = generates the output text word by word
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("ED1.jpg", caption="Simple: input → output", use_column_width=True)
    with col2:
        st.image("ED2.jpg", caption="Encoder-Decoder translates step by step", use_column_width=True)
    with col3:
        st.image("ED3.jpg", caption="Attention network powers it", use_column_width=True)

    st.image("ED4.jpg",
             caption="Full picture: Attention mechanism inside the Encoder-Decoder", use_column_width=True)

    st.divider()

    st.subheader("Models Used in This App")
    st.markdown("""
    | Model | Made By | Used For | How it works |
    |---|---|---|---|
    | **BART** | Facebook/Meta | Text Summarization | Encoder reads full text, decoder writes a short summary |
    | **NLLB-200** | Facebook/Meta | Language Translation | Translates between 200 languages using attention |

    ### What is BLEU Score?
    **BLEU (Bilingual Evaluation Understudy)** measures how close a machine translation is to a human reference translation.

    | Score | Quality |
    |---|---|
    | 0–20 | Poor |
    | 20–40 | Understandable |
    | 40–60 | Good |
    | 60–80 | Very High |
    | 80–100 | Near-Perfect |
    """)

    fig = plot_bleu_interpretation_table()
    st.pyplot(fig)

# ════════════════════════════════════════════════════════════
# SECTION 2 - Text Summarization
# ════════════════════════════════════════════════════════════
elif section == "Text Summarization":
    st.header("Text Summarization")
    st.markdown("""
    **BART** (Bidirectional and Auto-Regressive Transformer) reads your full text and writes a shorter summary.
    It was fine-tuned on CNN/Daily Mail news articles.
    """)

    default_text = """BART is a transformer encoder-encoder (seq2seq) model with a bidirectional (BERT-like) encoder
    and an autoregressive (GPT-like) decoder. BART is pre-trained by corrupting text with an arbitrary noising function,
    and learning a model to reconstruct the original text. BART is particularly effective when fine-tuned for text generation
    such as summarization and translation, but also works well for comprehension tasks such as text classification and
    question answering. This particular checkpoint has been fine-tuned on CNN Daily Mail, a large collection of
    text-summary pairs. The model can handle long documents and produces concise, coherent summaries."""

    user_text = st.text_area("Enter text to summarize:", value=default_text, height=180)

    st.sidebar.subheader("Summary Settings")
    min_len = st.sidebar.slider("Min summary length (tokens)", 10, 50, 20)
    max_len = st.sidebar.slider("Max summary length (tokens)", 50, 300, 100)
    rep_penalty = st.sidebar.slider("Repetition Penalty", 1.0, 10.0, 5.0, step=0.5,
        help="Higher = less repetition in output")
    len_penalty = st.sidebar.slider("Length Penalty", 0.1, 2.0, 0.3, step=0.1,
        help="Lower = longer summaries, Higher = shorter summaries")

    if st.button("Summarize"):
        if not user_text.strip():
            st.warning("Please enter some text!")
        else:
            with st.spinner("Loading summarization... (first run may take ~1 min to download)"):
                try:
                    tokenizer, model = load_summarizer()
                    summary = summarize_text(
                        user_text, tokenizer, model,
                        min_length=min_len, max_length=max_len,
                        repetition_penalty=rep_penalty, length_penalty=len_penalty
                    )

                    st.success("Summary generated!")

                    col1, col2 = st.columns(2)
                    col1.subheader("Original Text")
                    col1.write(user_text)

                    col2.subheader("Summary")
                    col2.info(summary)

                    st.subheader("Compression Stats")
                    fig, reduction = plot_summary_length_comparison(user_text, summary)
                    st.pyplot(fig)
                    st.metric("Text Reduction", f"{reduction}%",
                              help="How much shorter the summary is compared to the original")

                except Exception as e:
                    st.error(f"Summarization failed: {e}")
                    st.info("Tip: Make sure `transformers` and `torch` are installed.")

# ════════════════════════════════════════════════════════════
# SECTION 3 - Language Translation
# ════════════════════════════════════════════════════════════
elif section == "Language Translation":
    st.header("Language Translation")
    st.markdown("""
    **NLLB-200** (No Language Left Behind) translates between **200 languages**.
    Built by Meta/Facebook AI.
    """)

    col1, col2 = st.columns(2)
    src_lang_name = col1.selectbox("Source Language", list(LANGUAGE_CODES.keys()), index=0)
    tgt_lang_name = col2.selectbox("Target Language", list(LANGUAGE_CODES.keys()), index=11)

    src_lang = LANGUAGE_CODES[src_lang_name]
    tgt_lang = LANGUAGE_CODES[tgt_lang_name]

    user_text = st.text_area("Text to translate:", value="Bonjour, comment allez-vous ?", height=100)

    if st.button("Translate"):
        if not user_text.strip():
            st.warning("Please enter some text!")
        elif src_lang == tgt_lang:
            st.warning("Source and target languages must be different!")
        else:
            with st.spinner("Loading translation... (first run may take ~2 min to download)"):
                try:
                    translator = load_translator()
                    translation = translate_text(translator, user_text, src_lang, tgt_lang)

                    st.success("Translation complete!")

                    col1, col2 = st.columns(2)
                    col1.subheader(f"{src_lang_name}")
                    col1.info(user_text)
                    col2.subheader(f"{tgt_lang_name}")
                    col2.success(translation)

                except Exception as e:
                    st.error(f"Translation failed: {e}")

# ════════════════════════════════════════════════════════════
# SECTION 4 - BLEU Score Evaluator
# ════════════════════════════════════════════════════════════
elif section == "BLEU Score Evaluator":
    st.header("BLEU Score Evaluator")
    st.markdown("""
    **BLEU (Bilingual Evaluation Understudy)** measures translation quality
    by comparing model output to human reference translations. Score: 0–100.
    """)

    st.subheader("Test with French → English examples (from the notebook)")

    default_samples = [
        {"src": "Bonjour, comment allez-vous ?", "ref": "Hello, how are you?"},
        {"src": "Le chat dort sur le canapé.", "ref": "The cat is sleeping on the couch."},
        {"src": "Il fait très chaud aujourd'hui.", "ref": "It is very hot today."},
    ]

    st.markdown("**Sample Sentences to Translate & Evaluate:**")
    edited_samples = []
    for i, s in enumerate(default_samples):
        col1, col2 = st.columns(2)
        src = col1.text_input(f"Source {i+1} (French)", value=s["src"], key=f"src_{i}")
        ref = col2.text_input(f"Reference {i+1} (English)", value=s["ref"], key=f"ref_{i}")
        edited_samples.append({"src": src, "ref": ref})

    # Add extra sentence option
    if st.checkbox("Add a 4th sentence"):
        col1, col2 = st.columns(2)
        src4 = col1.text_input("Source 4", value="Je suis étudiant en informatique.")
        ref4 = col2.text_input("Reference 4", value="I am a computer science student.")
        edited_samples.append({"src": src4, "ref": ref4})

    src_lang = st.selectbox("Source Language", list(LANGUAGE_CODES.keys()), index=0)
    tgt_lang = st.selectbox("Target Language", list(LANGUAGE_CODES.keys()), index=11)

    if st.button("Translate & Evaluate BLEU"):
        with st.spinner("Loading NLLB model, translating, and scoring..."):
            try:
                translator = load_translator()
                predictions = translate_batch(
                    translator, edited_samples,
                    LANGUAGE_CODES[src_lang], LANGUAGE_CODES[tgt_lang]
                )

                references = [s["ref"] for s in edited_samples]
                bleu_score = calculate_bleu(predictions, references)
                interpretation = interpret_bleu(bleu_score)

                st.success(f"Evaluation complete!")

                # BLEU Score display
                st.subheader("Overall BLEU Score")
                col1, col2 = st.columns(2)
                col1.metric("BLEU Score", f"{bleu_score} / 100")
                col2.metric("Quality", interpretation)

                fig = plot_bleu_gauge(bleu_score)
                st.pyplot(fig)

                # Per-sentence results
                st.subheader("Sentence-by-Sentence Results")
                for i, (sample, pred) in enumerate(zip(edited_samples, predictions)):
                    with st.expander(f"Sentence {i+1}"):
                        st.write(f"**Source:** {sample['src']}")
                        st.write(f"**Reference (human):** {sample['ref']}")
                        st.write(f"**Predicted (model):** {pred}")
                        match = "Good match" if pred.lower().strip() == sample['ref'].lower().strip() else "Partial match"
                        st.caption(match)

                fig2 = plot_translation_results(edited_samples, predictions)
                st.pyplot(fig2)

                # Interpretation guide
                st.subheader("BLEU Score Reference")
                st.markdown("""
                | Score | Interpretation |
                |---|---|
                | 90–100 | Perfect / Near-human |
                | 60–80 | Very high quality |
                | 40–60 | Good quality |
                | 20–40 | Understandable |
                | 0–20 | Poor translation |
                """)

            except Exception as e:
                st.error(f"Evaluation failed: {e}")

# ── Footer ────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("**Project**")
st.sidebar.markdown("Text Summarization + Translation with Transformers")
