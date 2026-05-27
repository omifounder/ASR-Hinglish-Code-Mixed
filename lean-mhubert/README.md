# LeanHuBERT: A Compact, Commercially-Usable HuBERT-style Encoder for Hindi / Hinglish / English

This repo documents my design and implementation plan for a **lean HuBERT-style speech representation model**, inspired by mHuBERT‑147 but built **from scratch on commercially-licensed data** and focused on **Hindi, Hinglish, and English** telephony.

The goal is to get **mHuBERT‑like benefits** (robust multilingual SSL representations, strong downstream ASR, compact model size) while:
- Staying **CPU-friendly** for call center workloads.
- Remaining **safe for commercial use** (no non-commercial weights or data).
- Being transparent and reproducible enough to discuss in interviews and share on GitHub.

---

## 1. Motivation and Design Goals

Modern multilingual SSL models like XLS‑R‑300M and mHuBERT‑147 show that:
- A HuBERT-style multi-iteration SSL objective with discrete units from k-means clustering yields strong, reusable speech representations.
- A compact base-sized model (~95M params) can outperform much larger models on multilingual benchmarks when trained with carefully curated data and efficient clustering.

However:

- mHuBERT‑147 is released under a research-only / non-commercial license, so its weights cannot be used in production.
- Large generic models are often too heavy for CPU-only telephony deployments with 30–100 concurrent calls per server.
- I care specifically about **Hindi / Hinglish / Indian English** in noisy IVR / call center conditions.

**Design goal:** Build a ~100M-parameter HuBERT-style encoder, trained with a multi-iteration HuBERT objective and FAISS-based clustering, only on data I can legally use commercially, and tuned end-to-end for 8 kHz telephony.

---

## 2. High-Level Architecture

The model follows the HuBERT / mHuBERT line of work:

- **Encoder**: HuBERT-base style encoder (~95–110M parameters)  
  - 1D convolutional feature encoder over raw waveform.  
  - Transformer encoder stack (e.g., 12 layers, 768 hidden size).  
- **SSL objective**: HuBERT masked prediction over discrete units produced by k-means clustering of acoustic or hidden features.
- **Multi-iteration training**:
  - Iteration 0: k-means on MFCC/log-Mel features → HuBERT masked prediction on cluster IDs.  
  - Iteration 1: k-means on hidden states from Iteration 0 → improved targets and continued pre-training.
- **Downstream ASR**:
  - Compact CTC head on top of the encoder.  
  - Custom Hinglish tokenizer supporting Devanagari + Latin.  
  - CTC decoding via pyctcdecode + 4-gram KenLM.

**Differences vs. mHuBERT‑147:**

- Only 3 languages (Hindi, Hinglish, English) instead of 147.
- Much smaller, domain-focused data (telephony + Indian conversational speech vs. 90k+ h multi-domain).
- Implementation explicitly optimized for telephony ASR on CPU, not generic foundation model usage.

---

## 3. Data and Licensing Strategy

Strict requirement: **commercial safety**.

### 3.1 Data sources

I use three categories of data:

1. **Proprietary call center / IVR audio**  
   - Hindi, Hinglish, and English calls (8 kHz).  
   - Real-world background noise, accents, code-switching.

2. **Open-license Hindi / English corpora that allow commercial use**  
   - Curated based on license terms (e.g., specific CC variants, explicit commercial permission).

3. **Benchmarking-only corpora (for evaluation, not training)**  
   - MuCS 2021 for multilingual and code-switching ASR benchmarking.  
   - In my local setup, MuCS manifests live under:  
     - `/home/om/asr-benchmarking/hindi/tsvs/` (e.g., `mucs_train.tsv`, `mucs_test.tsv`).

No non-commercial models (e.g., mHuBERT‑147) are used as initialization. I treat them as **papers to learn from**, not checkpoints to reuse.

### 3.2 Preprocessing

- Resample all training audio to **8 kHz mono, 16-bit PCM** (telephony channel).  
- Apply basic VAD and silence trimming.  
- Maintain metadata per utterance:
  - `language` in {`hi`, `en`, `hinglish`}.  
  - `domain` in {`call_center`, `ivr`, `web`}.  
  - `duration_sec`.

For SSL sampling and clustering, I cap per-language hours to prevent English from overwhelming Hindi/Hinglish, inspired by mHuBERT‑147’s language and dataset up-sampling scheme but simplified to 3 languages.

---

## 4. Multi-Iteration HuBERT Training Plan

### 4.1 Iteration 0: k-means on acoustic features

**Goal:** Produce initial discrete units for HuBERT using shallow acoustic features.

- Features:
  - MFCCs or log-Mel filterbanks (25 ms window, 10 ms hop), as in the original HuBERT setup.  
- Sampling:
  - Sample frames from all three languages with per-language caps (e.g., up to N hours per language).  
- Clustering:
  - Use FAISS IVF k-means on all sampled frames to obtain K clusters (K≈100–200).  
  - FAISS allows clustering and label assignment at scale with manageable memory and runtime.  
- Label assignment:
  - For every frame in the training set, assign a discrete label 0..K‑1.
  - Store labels as compressed arrays with consistent alignment to frames.

### 4.2 Iteration 0: HuBERT pre-training

**Architecture (HuBERT-base-like):**

- Conv feature encoder → downsampled sequence of latent frames.  
- Transformer encoder with ~12 layers, hidden size ≈768, 8 attention heads, FFN size ≈3072.

**Objective:**

- Mask contiguous time spans and predict cluster IDs at masked positions.  
- Use the HuBERT loss combining masked and unmasked positions:
  \[
  L = \psi L_m + (1 - \psi)L_u
  \]
  where \(L_m\) and \(L_u\) are cross-entropy losses on masked and unmasked frames respectively.

**Training sketch:**

- Optimizer: AdamW, linear warmup, cosine decay.  
- Masking: ~75 ms spans, mask probability ~0.065 (HuBERT default range).  
- Steps: target ~200–300k steps depending on total hours.

### 4.3 Iteration 1: clustering on hidden states

**Feature extraction:**

- Run the Iteration‑0 encoder on training audio to extract hidden states from a mid-layer (e.g., layer 6 or 8).  
- Sample hidden vectors across languages, again with balanced sampling.

**Clustering:**

- Apply FAISS IVF k-means on these hidden vectors with a larger K (e.g., 500–1000) for richer discrete units.  
- Re-assign labels for all time steps in the dataset.

**Iteration‑1 HuBERT training:**

- Initialize from Iteration‑0 encoder weights.  
- Train with the new discrete units for another ~150–250k steps, same masking and loss.  
- This is analogous to moving from early to later iterations in mHuBERT‑147, but tailored to 3 languages and smaller compute.

The Iteration‑1 encoder is the final **LeanHuBERT encoder** used for downstream tasks.

---

## 5. Downstream ASR and MuCS Benchmarking

### 5.1 Custom Hinglish tokenizer and CTC head (already prototyped)

I already have a working **Hinglish “charish” tokenizer + CTC wrapper** for a HuBERT-style encoder:

- Tokenizer:
  - Built using `tokenizers` BPE with a character-level alphabet:
    - Devanagari vowels and consonants.  
    - Latin a–z.  
    - Digits and basic punctuation.  
  - Wrapped as `PreTrainedTokenizerFast` and saved to a local directory, e.g. `./hinglish_charish_tokenizer`.  
- Processor:
  - `AutoFeatureExtractor.from_pretrained(MHUBERT_DIR)` + custom tokenizer →
    `Wav2Vec2Processor(feature_extractor, tokenizer)`.  
- Model wrapper:
  - A `MHuBERTForCTC` module that:
    - Instantiates the HuBERT encoder from config.  
    - Loads base HuBERT/mHuBERT encoder weights.  
    - Adds a linear CTC head and computes CTC loss with label masking.

This path has already been validated with a dummy forward pass and a small subset of MuCS to ensure end-to-end shape correctness.

### 5.2 MuCS evaluation setup (already working)

For benchmarking on **MuCS 2021 Hindi / Hinglish**:

- MuCS TSV manifests live under:

  - `/home/om/asr-benchmarking/hindi/tsvs/mucs_train.tsv`  
  - `/home/om/asr-benchmarking/hindi/tsvs/mucs_test.tsv`

- I load them via Hugging Face Datasets:

  ```python
  from datasets import load_dataset, Audio

  data_files = {
      "train": "/home/om/asr-benchmarking/hindi/tsvs/mucs_train.tsv",
      "test": "/home/om/asr-benchmarking/hindi/tsvs/mucs_test.tsv",
  }
  ds = load_dataset("csv", data_files=data_files, delimiter="\t")
  ds = ds.cast_column("audio", Audio(sampling_rate=16000))
  ```

- A debug flag allows running on only N samples for rapid iteration:

  ```python
  DEBUG_FEW_SAMPLES = True
  DEBUG_NUM_SAMPLES = 5

  if DEBUG_FEW_SAMPLES:
      train_raw = ds["train"].select(range(min(DEBUG_NUM_SAMPLES, len(ds["train"]))))
      test_raw  = ds["test"].select(range(min(DEBUG_NUM_SAMPLES, len(ds["test"]))))
  else:
      train_raw = ds["train"]
      test_raw  = ds["test"]
  ```

- A `prepare_dataset` function maps:
  - Raw audio → `input_values` (via `Wav2Vec2Processor`).  
  - `text` → `labels` token IDs (with padding and `-100` mask for CTC).

- Evaluation:
  - Model runs in eval mode on GPU or CPU.  
  - Hypotheses are obtained by simple greedy CTC decoding with `processor.batch_decode`.  
  - WER is computed with `jiwer.wer(refs, hyps)` over the test split.

This gives a baseline pipeline to plug in any HuBERT-like encoder and see MuCS WER quickly, including the future LeanHuBERT model.

### 5.3 Telephony CTC decoding (planned)

After ASR fine-tuning:

- Replace greedy decode with CTC beam search:
  - `pyctcdecode` + 4-gram KenLM trained on Hindi/Hinglish/English transcripts.  
- Use adaptive beams depending on speech rate / confidence, to balance accuracy vs latency on CPU.

---

## 6. Training Compute and Practical Feasibility

I am designing this to be trainable with **modest GPU resources**, guided by:

- HuBERT pre-training recipes that show how to cut compute by careful batching and feature handling.  
- mHuBERT‑147’s FAISS-based clustering tricks, which avoid massive RAM requirements when labeling large amounts of data.

Planned practices:

- Use FAISS to keep clustering scalable without storing all features in RAM.  
- Start with a reasonable total hours budget (hundreds to a low thousands) focused on telephony and Indian accents instead of chasing 90k hours.  
- Incrementally scale up data and iterations as compute permits.

---

## 7. Implementation Status

**Already implemented / validated:**

- Local mHuBERT encoder directory (`/home/om/models/mhubert147`) and HF-based loading.  
- Custom Hinglish tokenizer and CTC wrapper (`MHuBERTForCTC`) with a working dummy forward pass.  
- MuCS TSV manifests and dataset loader using Hugging Face Datasets.  
- End-to-end MuCS evaluation script with:
  - Debug mode (5 samples vs full set).  
  - Audio → features → CTC logits → greedy decode → WER with `jiwer`.

**Planned / in-progress:**

- Implement Iteration‑0 clustering with MFCC/log-Mel + FAISS IVF k-means.  
- Train Iteration‑0 HuBERT on Hindi/Hinglish/English pooled data.  
- Implement Iteration‑1 clustering over encoder hidden states and retrain.  
- Fine-tune LeanHuBERT + CTC on MuCS + proprietary call center datasets.  
- Integrate pyctcdecode + KenLM for robust telephony decoding on CPU.  
- Benchmark against `facebook/wav2vec2-xls-r-300m` and other SSL baselines on MuCS and internal test sets.

---

## 8. Licensing and Commercial Safety

To keep this project **commercially usable**:

- I do not use mHuBERT‑147 weights or any non-commercial models as initialization.  
- I treat models like mHuBERT‑147, XLS‑R, MMS, etc. solely as **conceptual references** (architecture and training ideas).  
- All training data are either:
  - Proprietary audio I own or have explicit rights to, or  
  - Datasets whose licenses allow commercial usage.

The repo itself focuses on **code, configuration, and design**, not on distributing any questionable weights or data.

---

## 9. This project highlights:



- **Representation learning**:
  - Why HuBERT-style SSL is a good fit for multilingual telephony.  
  - How multi-iteration clustering with FAISS improves representation quality and multilingual robustness.

- **System design under constraints**:
  - Bridging from a generic SSL encoder to a CPU-efficient production ASR system.  
  - Balancing accuracy, latency, and cost with 8 kHz telephony and limited compute.

- **Licensing and product thinking**:
  - Designing from day one for commercial safety (no research-only weights).  
  - Adapting a large academic model (mHuBERT‑147) into a **domain-specific, deployable** variant for Hindi/Hinglish/English contact centers.

If you’re reading this on GitHub and want to discuss the design or collaborate on LeanHuBERT-style models, feel free to open an issue or reach out.
