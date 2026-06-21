# A Hybrid Retrieval-Augmented Generation Framework with LoRA-Tuned LLM for Bengali QA

Official repository for **The Dropout Squad**'s submission to the **IEEE CS CUET ML Contest 2.0**. This project implements a high-performance, resource-constrained open-domain Question Answering (QA) system tailored for the Bengali language, achieving a **Recall@2 of 75.5%** and an official server **Token-level F1 score of 0.723** (+33.3 points over the baseline TF-IDF system).

## 👥 Authors
- **Mohammad Sakib Hasan Alif** (u2204051@student.cuet.ac.bd)
- **Ayan Barua** (u2204053@student.cuet.ac.bd)
- *Department of Computer Science and Engineering, Chittagong University of Engineering & Technology (CUET)*

---

## 📌 Table of Contents
1. [Motivation & Key Challenges](#-motivation--key-challenges)
2. [System Architecture](#-system-architecture)
3. [The Four-Stage Retrieval Pipeline](#-the-four-stage-retrieval-pipeline)
4. [QLoRA Fine-Tuning Strategy](#-qlora-fine-tuning-strategy)
5. [Implementation Under 16 GB VRAM](#-implementation-under-16-gb-vram)
6. [Experimental Evaluation & Ablation Study](#-experimental-evaluation--ablation-study)
7. [Error Analysis & Bengali Failure Modes](#-error-analysis--bengali-failure-modes)
8. [Limitations & Future Directions](#-limitations--future-directions)
9. [Acknowledgments](#-acknowledgments)

---

## 🎯 Motivation & Key Challenges

While Retrieval-Augmented Generation (RAG) is highly mature for high-resource languages like English, transitioning it to Bengali exposes three deep, language-specific structural bottlenecks:

1. **Morphological Fragmentation:** Bengali features rich inflectional morphology. A single lemma explodes into dozens of inflected surface forms (e.g., `ঢাকার`, `ঢাকায়`, `ঢাকা`), starving traditional lexical retrievers like BM25 of critical term-frequency signals.
2. **Lexical Mismatch:** Bengali Wikipedia articles predominantly rely on passive, nominalized structural phrasing, whereas human users query the system using active, short factoid expressions. 
3. **Low-Resource Tooling Scarcity:** Modern pre-trained multilingual language models frequently lack fine-grained exposure to language-specific syntactic patterns and regional semantic nuances.

**Our Objective:** Given an un-annotated Bengali query and an unstructured Bengali Wikipedia knowledge base, accurately isolate relevant evidence sentences and extract concise, grammatically sound answers concluding with a Bengali full-stop (`।`).

---

## 🏗️ System Architecture

The architecture operates in two distinct phases: **Offline Data Preparation** and **Online Real-Time Inference**.

```
OFFLINE INDEXING PIPELINE:
[Knowledge Base: 6,988 Articles] ➔ [Parse & Clean] ➔ [Semantic Sentence Chunking]
                                                             │
                                        ┌────────────────────┴────────────────────┐
                                        ▼                                         ▼
                                 [BM25 Index]                            [BGE-M3 Embeddings]

ONLINE INFERENCE PIPELINE:
[Test Question] ➔ ┌───────────────────────────────────────┐
                  │ 1. Suffix-Stemmed BM25  (Top-100)     │
                  │ 2. BGE-M3 Dense Search  (Top-100)     │
                  └───────────────────┬───────────────────┘
                                      ▼
                             [3. RRF Fusion (Top-30)]
                                      ▼
                        [4. Cross-Encoder Reranker]
                                      ▼
                           [Top-2 Context Selection]
                                      ▼
                        [QLoRA Fine-Tuned Llama 3.1 8B]
                                      ▼
                        [5-Step Post-Processing + Fallback] ➔ [Final Concise Answer]
```

---

## 🔍 The Four-Stage Retrieval Pipeline

### 1. Suffix-Stemmed BM25 Lexical Retrieval
- **Text Cleansing:** Removes a curated list of 40 common Bengali stop-words.
- **Rule-Based Stemmer:** Strips 6 high-frequency inflectional suffixes (`এর`, `তে`, `কে`, `রা`, `য়`, `র`) to map surface variants back to their semantic base forms.
- **Parameters:** $k_1 = 1.5$, $b = 0.75$, yielding an average document length ($avgdl$) of 49.6 tokens. Passes the **Top-100** candidates forward.

### 2. BGE-M3 Dense Retrieval
- Maps all 194,427 generated Wikipedia chunks into a 1,024-dimensional semantic vector space (totaling a ~759.5 MB matrix).
- Resolves contextual vocabulary mismatches through cosine similarity profiles. Passes the **Top-100** candidates forward.

### 3. Reciprocal Rank Fusion (RRF)
- Blends lexical accuracy on names/numbers with the conceptual flexibility of dense vectors using a smoothing constant $k = 60$:
  $$RRF(d) = \sum_{r \in R} rac{1}{k + rank_r(d)}$$
- The top-fused **30 candidates** move into the deep scoring matrix.

### 4. Cross-Encoder Reranking
- Uses `BGE-Reranker-v2-M3` to evaluate raw, non-linear query-document interactions.
- Context is aggressively pruned down to the **Top-2 chunks** ($\le 100$ words total) to neutralize the *lost-in-the-middle* phenomenon during LLM execution.

---

## 🤖 QLoRA Fine-Tuning Strategy

The generative model is built upon **Llama 3.1 8B Instruct** utilizing parameter-efficient adapters via Unsloth kernels.

### Fine-Tuning Hyperparameters & Progression
| Hyperparameter / Feature | Model V1 | Model V2 | Model V3 (Final) |
| :--- | :---: | :---: | :---: |
| **Adapter Rank ($r$)** | 16 | 32 | **32** |
| **Adapter Alpha ($lpha$)** | 32 | 64 | **64** |
| **Trainable Parameters** | 41.9M (0.52%) | 83.9M (1.03%) | **83.9M (1.03%)** |
| **Training Epochs** | 1 | 3 | **3** |
| **Negative Sample Injection**| 10% | 25% | **25%** |
| **Prompt Engineering Type** | Strict | Strict | **Adaptive Prompting** |
| **System Prompt Layout** | Inside User Node| Inside User Node| **Dedicated System Node**|
| **Training Duration** | ~1.5 Hours | ~4.9 Hours | **~5.9 Hours** |

### Core Architectural Extensions
- **Negative-Sample Injection (25%):** Purposely pairs 25% of training prompts with non-relevant document contexts, setting the target extraction token to `জানা নেই।` ("Unknown."). This strictly trains out hallucination patterns.
- **Adaptive Prompt Routing:** Automatically identifies and routes queries into two distinct generation lanes:
  - *Factoid Inquiries (Who/When/Where):* Enforces a compact 1-3 word constraint.
  - *Explanatory Inquiries (Why/How):* Grants a wider length budget for necessary contextual completeness.
- **Response-Only Loss Masking:** Computes backpropagation penalties exclusively on output answer tokens using Unsloth masking features, entirely preventing prompt template memorization.

### Post-Processing & Robust Fallback
A deterministic 5-stage cleaning layer cleans output streams:
1. Strips explicit leading answer prefixes (e.g., `উত্তর:`).
2. Truncates strings down to the first sentence boundary.
3. Automatically triggers an overlap-driven fallback extraction protocol if the LLM emits a failure or abstention phrase (`জানা নেই`).
4. Caps string length at 15 words if anomalous sequence extension crosses 20 words.
5. Verifies and appends the standard Bengali Dari punctuation (`।`).

---

## 💻 Implementation Under 16 GB VRAM

To support consumer hardware access and strict infrastructure constraints, the entire code pathway is optimized to run end-to-end on a single **16 GB VRAM GPU**:
- **Sequential Module Management:** Because models cannot fit concurrently in memory, the Dense Embedder, Cross-Encoder, and QLoRA LLM are loaded, utilized, and completely purged from the GPU cache sequentially via explicit `torch.cuda.empty_cache()` and garbage collection steps.
- **Kernel Acceleration:** Integrated `Unsloth` 4-bit quantization wrappers, speeding up learning throughput by 2-5$	imes$.
- **Compute Profiles:** Offline indexing pipeline finishes in **~3 hours**, while online batch inference runs across all 1,500 test questions in **~1.5 hours**.

---

## 📊 Experimental Evaluation & Ablation Study

### Component Performance Progression
Our developmental progression shows a **+33.3 Token F1 point jump** over standard text setups:

| Model Iteration | Component Integration Details | Test Token F1 |
| :--- | :--- | :---: |
| **V1** | Baseline TF-IDF + 400-Word Static Chunks + Base Llama LLM | 0.390 |
| **V2** | + BM25/Dense Hybrid Indexing + RRF Fusion + QLoRA V1 Adapters | 0.610 |
| **V3** | + Semantic Sentence Chunking + Suffix Stemmer + Cross-Encoder Rerank | 0.680 |
| **V4** | + Upgraded QLoRA V2 (Rank=32, Alpha=64, 3 Training Epochs, 25% Negatives) | 0.690 |
| **V5 (Final)** | + Adaptive Prompt Routing + Strict Context Pruning + Fallback Pipelines | **0.723** |

### Gold-Standard Test Set Metrics (1,500 Queries)
- **Official Server Token-level F1:** `0.72302`
- **Exact Match Rate:** 54.8% (822 / 1,500 samples)
- **Partial Match Rate:** 71.2% (1,068 / 1,500 samples)
- **Dari (`।`) Sentence Completion Enforce Rate:** 99.4%
- **Generalization Profiling:** Highly stable distributions with a Public Split F1 of 69.3% vs. a Private Split F1 of 69.7%.

---

## 🔍 Error Analysis & Bengali Failure Modes

Through a manual qualitative assessment of 200 random failure cases sampled out of the 24.5% retrieval miss margin ($1 - 	ext{Recall@2}$), we identified 5 principal vulnerabilities:

1. **Insufficient Stemming (~10%):** Rule-based suffix stripping misses complex verbal inflections (e.g., `প্রতিষ্ঠিত` vs. `প্রতিষ্ঠা`), word compounds, and adjectival derivations.
2. **Cross-Sentence Context Splits (~6%):** Critical semantic answers split across hard boundaries, causing the extracted text chunk to hold isolated pronouns (`তিনি`, `তার`) missing their parent entity reference.
3. **Indirect Wikipedia Phrasing (~5%):** Lexical mismatch where queries request `রাজধানী` ("capital") but reference literature uses `প্রশাসনিক কেন্দ্র` ("administrative center"), bypassing both dense and keyword parameters.
4. **Pool-Size Cutoff Constraints (~2%):** Strict Top-100 pool filters accidentally exclude valid chunks that rank marginally lower before RRF steps can run.
5. **Train/Test Distribution Discrepancy (~3%):** Clean training pairs contrast with test realities where 68.7% of contexts include mixed multi-article content and 13.5% contain heavy pronoun clusters, introducing minor covariate generation shifts.

---

## 🚀 Limitations & Future Directions

- **Advanced Morphological Graphing:** Transitioning from basic regex suffix tools to a dictionary-backed morphological analyzer or implementing BGE-M3's multi-vector ColBERT representation.
- **Noise-Robust LLM Training:** Fine-tuning the adapter layer on intentionally degraded, noisy retrieval candidates rather than pristine contexts to mirror test-time environments.
- **Hypothetical Document Embeddings (HyDE):** Generating synthetic Bengali response drafts first to query dense vector maps, eliminating vocabulary style mismatch.
- **FAISS Sub-Linear Clustering:** Converting vector checks from linear $O(n)$ search loops to FAISS sub-linear sub-spaces to cleanly allow broader Top-500 candidate windows.

---

## 🤝 Acknowledgments

We thank the organizers of the **"Are You Sure LLM Is Enough?"** machine learning tournament, hosted by the **IEEE Computer Society CUET Student Branch Chapter**, for providing the datasets, evaluation platform, and computing benchmarks. 

We also recognize the open-source maintenance teams behind `BM25Okapi`, `BGE-M3`, `Llama 3.1`, `QLoRA`, and `Unsloth` for providing highly efficient, open-domain model layers.here
