"""
utils.py - Visualization helpers for Transformer Apps
"""

import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

logging.basicConfig(level=logging.INFO)


def plot_bleu_gauge(score: float):
    """Horizontal gauge bar for BLEU score (0-100)."""
    try:
        fig, ax = plt.subplots(figsize=(9, 2))

        # Background
        ax.barh(0, 100, color='#EEEEEE', height=0.5)

        # Color zones
        zone_colors = [('#F44336', 20), ('#FF9800', 20), ('#FFC107', 20), ('#8BC34A', 20), ('#4CAF50', 20)]
        left = 0
        for color, width in zone_colors:
            ax.barh(0, width, left=left, color=color, height=0.5, alpha=0.4)
            left += width

        # Score bar
        color = '#4CAF50' if score >= 60 else '#FF9800' if score >= 40 else '#F44336'
        ax.barh(0, score, color=color, height=0.3, alpha=0.9)

        # Score marker
        ax.axvline(score, color='black', linewidth=2)

        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_xticklabels(['0\nPoor', '20', '40\nGood', '60', '80\nExcellent', '100'])
        ax.set_title(f'BLEU Score: {score:.2f} / 100', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"BLEU gauge error: {e}")
        raise


def plot_summary_length_comparison(original: str, summary: str):
    """Bar chart comparing original vs summary word count."""
    try:
        orig_words = len(original.split())
        summ_words = len(summary.split())
        reduction = round((1 - summ_words / orig_words) * 100, 1) if orig_words > 0 else 0

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(['Original Text', 'Summary'], [orig_words, summ_words],
                      color=['#2196F3', '#4CAF50'], edgecolor='white', linewidth=2)

        for bar, val in zip(bars, [orig_words, summ_words]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{val} words', ha='center', fontweight='bold')

        ax.set_ylabel('Word Count')
        ax.set_title(f'Text Compression: {reduction}% reduction', fontweight='bold')
        plt.tight_layout()
        return fig, reduction
    except Exception as e:
        logging.error(f"Length comparison error: {e}")
        raise


def plot_translation_results(samples: list, predictions: list):
    """Table-style bar chart showing per-sentence translation similarity."""
    try:
        import sacrebleu
        per_scores = []
        for pred, sample in zip(predictions, samples):
            b = sacrebleu.sentence_bleu(pred, [sample['ref']])
            per_scores.append(round(b.score, 1))

        labels = [f"Sentence {i+1}" for i in range(len(per_scores))]
        colors = ['#4CAF50' if s >= 60 else '#FF9800' if s >= 30 else '#F44336' for s in per_scores]

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(labels, per_scores, color=colors, edgecolor='white')

        for bar, score in zip(bars, per_scores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f'{score}', ha='center', fontweight='bold')

        ax.set_ylim(0, 110)
        ax.set_ylabel('BLEU Score')
        ax.set_title('Per-Sentence BLEU Scores', fontweight='bold')
        ax.axhline(60, color='green', linestyle='--', alpha=0.5, label='High quality (60)')
        ax.legend()
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Translation results plot error: {e}")
        raise


def plot_bleu_interpretation_table():
    """Static chart explaining BLEU score ranges."""
    try:
        ranges = ['0–20', '20–40', '40–60', '60–80', '80–100']
        labels = ['Poor', 'Understandable', 'Good', 'Very High', 'Near-Perfect']
        colors = ['#F44336', '#FF9800', '#FFC107', '#8BC34A', '#4CAF50']
        values = [20, 20, 20, 20, 20]

        fig, ax = plt.subplots(figsize=(9, 2))
        left = 0
        for i, (val, color, label, rng) in enumerate(zip(values, colors, labels, ranges)):
            ax.barh(0, val, left=left, color=color, height=0.6)
            ax.text(left + val / 2, 0, f'{rng}\n{label}',
                    ha='center', va='center', fontsize=9, fontweight='bold', color='white')
            left += val

        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title('BLEU Score Interpretation Guide', fontweight='bold')
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"BLEU table error: {e}")
        raise
