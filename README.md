# 🤖 AI-Powered Paraphrasing Tool

> A transformer-based NLP tool that paraphrases text while preserving meaning, checking grammar/spelling/fluency, and evaluating output quality using BLEU, ROUGE, and semantic similarity scores.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Pipeline Architecture](#pipeline-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Sample Results](#sample-results)
- [Evaluation Report](#evaluation-report)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)

---

## Overview

This project implements an end-to-end AI paraphrasing pipeline using the **T5 transformer model** fine-tuned on the PAWS (Paraphrase Adversaries from Word Scrambling) dataset. Given any input sentence, the tool generates multiple paraphrase candidates, checks them for quality, scores them on three evaluation metrics, and automatically returns the best-ranked output.

The entire tool runs in a **Google Colab notebook** with console-based input/output — no GUI or web interface required.

---

## Features

- **Transformer-based paraphrasing** using `Vamsi/T5_Paraphrase_Paws` via HuggingFace
- **Diverse output generation** via nucleus sampling with a temperature ladder
- **Grammar checking** using LanguageTool with auto-correction
- **Spelling correction** using TextBlob
- **Fluency scoring** using spaCy dependency tree analysis
- **BLEU score** — measures n-gram overlap (originality indicator)
- **ROUGE-1/2/L** — measures recall-based similarity
- **Semantic Similarity** — cosine similarity via SBERT (`all-MiniLM-L6-v2`)
- **Composite ranking** — automatically selects the best variant
- **Full evaluation report** with aggregate statistics

---

## Tech Stack

| Category | Library | Version |
|----------|---------|---------|
| Deep Learning | PyTorch | 2.x |
| Transformers | HuggingFace Transformers | 5.x |
| Paraphrase Model | T5 (Vamsi/T5_Paraphrase_Paws) | — |
| Similarity Model | SBERT (all-MiniLM-L6-v2) | — |
| Grammar Check | language-tool-python | 3.x |
| Spelling Check | TextBlob | 0.19 |
| NLP / Fluency | spaCy (en_core_web_sm) | 3.8 |
| BLEU Score | NLTK | 3.9 |
| ROUGE Score | rouge-score | 0.1.2 |
| Environment | Google Colab | — |

---

## Project Structure

```
ai-paraphrasing-tool/
│
├── AI_Paraphrasing_Tool.ipynb     # Main Colab notebook (all 6 parts)
│
├── README.md                      # This file
│
└── sample_outputs/
    └── evaluation_report.txt      # Sample results from test run
```

---

## Pipeline Architecture

```
Input Text
    │
    ▼
┌─────────────────────┐
│  ParaphrasingEngine │  ← T5 Transformer + Nucleus Sampling
│  (Part 2)           │    Temperature ladder: 1.2 → 1.6 → 2.0
└────────┬────────────┘
         │  N paraphrase variants
         ▼
┌─────────────────────┐
│   QualityChecker    │  ← Grammar (LanguageTool)
│   (Part 3)          │    Spelling (TextBlob)
└────────┬────────────┘    Fluency  (spaCy)
         │  Quality report per variant
         ▼
┌─────────────────────┐
│  EvaluationMetrics  │  ← BLEU Score (NLTK)
│  (Part 4)           │    ROUGE-1/2/L (rouge-score)
└────────┬────────────┘    Semantic Similarity (SBERT)
         │  Metric scores per variant
         ▼
┌─────────────────────┐
│  Composite Scorer   │  ← 50% Semantic + 30% Fluency
│  + Ranker           │    − Grammar Penalty (up to 20%)
└────────┬────────────┘
         │
         ▼
    Best Paraphrase
    + Full Ranked List
    + Evaluation Report
```

---

## Installation

### Option A — Run in Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `AI_Paraphrasing_Tool.ipynb`
3. Run **Part 1** (installs all dependencies)
4. **Restart the runtime** when prompted
5. Run Parts 2 through 6 in order

### Option B — Run Locally

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/ai-paraphrasing-tool.git
cd ai-paraphrasing-tool

# Install dependencies
pip install transformers torch sentencepiece
pip install language-tool-python textblob rouge-score
pip install sentence-transformers nltk spacy
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
```

> **Note:** First run downloads ~1GB of model weights (T5 + SBERT). These are cached automatically after the first download.

---

## Usage

### In Colab — run all 6 parts sequentially, then use:

```python
# Paraphrase any sentence
result = tool.run("Your sentence goes here.", num_variants=3)

# Get just the best paraphrase
best = result[0]["final_text"]
print(best)
```

### Adjust diversity vs. quality trade-off:

```python
# More conservative (safer, less creative)
variants = engine.paraphrase(text, num_variants=3)
# Edit temperature ladder in paraphrase() to: [1.2, 1.3, 1.4]

# More creative (higher risk of garbled output)
# Edit temperature ladder to: [1.5, 1.8, 2.2]
```

---

## Sample Results

```
INPUT: Artificial intelligence is rapidly changing the way we work and live.

Rank  Score   Semantic  BLEU    Fluency   G.Err  Flag
------------------------------------------------------------
[1]   0.674   0.940     0.4355  Good      0      ✅ GOOD
      The Artificial Intelligence revolution is rapidly changing the way we live and work.

[2]   0.569   0.7148    0.0289  Good      0      REVIEW
      It depends ever more on artificial intelligence to change how you grow up and operate daily.

[3]   0.535   0.7496    0.4863  Fair      0      ✅ GOOD
      Machine learning is drastically changing the way we work and live.

BEST PARAPHRASE:
  The Artificial Intelligence revolution is rapidly changing the way we live and work.
```

---

## Evaluation Report

Results from testing on 4 sentences:

| Sentence | Best Paraphrase (truncated) | Semantic | BLEU | Status |
|----------|----------------------------|----------|------|--------|
| Fox/riverbank | "During the rush, the swift brown fox..." | 0.755 | 0.093 | ⚠️ REVIEW |
| AI sentence | "The AI revolution is rapidly changing..." | 0.940 | 0.436 | ✅ GOOD |
| She/company | "After ten years...she decided to leave" | 0.992 | 0.777 | ⚠️ NEAR-COPY |
| Climate change | "The global climate change poses a grave threat..." | 0.973 | 0.545 | ✅ GOOD |

### Aggregate Statistics

| Metric | Score | Target | Result |
|--------|-------|--------|--------|
| Semantic Similarity | **0.915** | > 0.70 | ✅ Exceeds target |
| BLEU Score | **0.462** | 0.10–0.50 | ✅ Within ideal range |
| Fluency Score | **0.696** | > 0.60 | ✅ Exceeds target |
| Composite Score | **0.641** | > 0.60 | ✅ Exceeds target |
| GOOD paraphrases | **50%** | > 50% | ✅ Meets target |
| Meaning drift cases | **0%** | 0% | ✅ Perfect |

---

## Limitations

- **Short/simple sentences** tend toward near-copies since there is less structural room to rephrase
- **High temperature sampling** (≥ 2.0) occasionally produces garbled or nonsensical output — the quality checker flags these automatically
- **LanguageTool** misses some semantic oddities (e.g., "nearby onto") that are grammatically valid but stylistically poor
- The tool runs on **CPU by default** — enable GPU runtime in Colab for 5–10x faster generation

---

## Future Improvements

- [ ] Swap in a larger model (`humarin/chatgpt_paraphraser_on_T5_base`) for better quality
- [ ] Add a minimum semantic similarity filter to auto-discard low-quality variants
- [ ] Fine-tune the T5 model on a domain-specific paraphrase corpus
- [ ] Add support for paragraph-level (multi-sentence) paraphrasing
- [ ] Export results to CSV for batch processing

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [Vamsi/T5_Paraphrase_Paws](https://huggingface.co/Vamsi/T5_Paraphrase_Paws) model
- [sentence-transformers](https://www.sbert.net/) for semantic similarity
- [LanguageTool](https://languagetool.org/) for grammar checking
- [spaCy](https://spacy.io/) for NLP fluency analysis
