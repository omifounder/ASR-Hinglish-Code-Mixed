# Synthetic Data Generation Strategy

## Overview

This pipeline generates synthetic Hindi/Hinglish speech data for fine-tuning ASR models on airline call center audio. The goal is to improve word error rate (WER) on domain-specific named entities — PNR numbers, flight numbers, city names, and airport codes — for an Om Airways contact centre handling Hindi/Hinglish speaking customers.

Synthetic data is mixed at **20% synthetic / 80% real** in the final training set.

---

## Why Synthetic Data

Real call center audio for Hindi/Hinglish airline domains is scarce, expensive to license, and heavily skewed toward a small number of high-frequency intents. Synthetic generation allows:

- Controlled coverage across all 32 intent categories
- Balanced personality and speaking style variation
- Precise entity injection (PNR, flight numbers, dates)
- Repeatable generation across multiple passes at different temperatures
- No privacy or PII concerns

---

## Intent Taxonomy

32 sub-intents across 8 domains, modelled on real Om Airways contact centre call flows:

| Domain | Sub-intents |
|---|---|
| A — Booking & Reservation | New Booking, Change/Modify, Cancel, PNR Retrieval, Name Correction, Missed Flight |
| B — Flight Status | Flight Status, Flight Details, Flight Dispute, Still at Airport |
| C — Check-in & Boarding | Check-in Process, Boarding Pass |
| D — Baggage | Allowance Query, Excess/Extra, Lost/Damaged |
| E — Seat & Cabin | Seat Selection, IFE & Wi-Fi, Onboard Dining |
| F — Special Assistance | Wheelchair, Medical Clearance, Pregnancy, Bassinet/Infant, Pets |
| G — Loyalty & Profile | Om Guest Membership, Credit/Voucher, Profile Update |
| H — Ancillary & General | Chauffeur, Visa/Passport, Agent Transfer, General Enquiry, Closing |

---

## Sentence Distribution

Sentence count per sub-intent is proportional to real-world call frequency, with a minimum floor of 6 sentences per sub-intent:

```
sentences = (distribution_percentage * 10) + 5
```

| Sub-intent | Distribution | Sentences |
|---|---|---|
| B1 Flight Status | 10% | 15 |
| A1 New Booking | 8% | 13 |
| A4 PNR Retrieval | 8% | 13 |
| A2 Change/Modify | 7% | 12 |
| ... | ... | ... |
| F2 Medical Clearance | 1% | 6 |
| F3 Pregnancy | 1% | 6 |
| B4 Still at Airport | 1% | 6 |
| **Total** | **100%** | **~960** |

---

## Generation Approach

### Conversation-Based Generation

Rather than generating isolated sentences, the pipeline generates **full agent-user conversations** and extracts user turns. This produces naturally flowing utterances that reflect real mid-conversation speech patterns — not just scripted opening statements.

```
32 sub-intents × 10 conversations = 320 conversations
  → 7 single-intent conversations (4-6 user turns)
  → 3 multi-intent conversations  (6-10 user turns, 1-2 intent shifts)
  → ~1,600 raw user utterances
  → curated to ~1,000 solid sentences
```

### Two-Pass Generation

Two independent generation passes at different temperatures increase lexical and structural diversity:

| Pass | Temperature | Focus |
|---|---|---|
| Pass 1 | 0.7 | Varied sentence openings |
| Pass 2 | 0.9 | Varied sentence structures and lengths |

---

## Sentence Variation Criteria

### Length Buckets

Every sub-intent is covered across three length buckets:

| Bucket | Word Count | Share |
|---|---|---|
| Short | 5–8 words | ~27% |
| Medium | 9–15 words | ~40% |
| Long | 16–25 words | ~33% |

### Caller Personality

Five personality types are applied with weighted sampling:

| Personality | Weight | Characteristics |
|---|---|---|
| Neutral | 35% | Natural, polite, standard Hinglish |
| Hesitation | 20% | Fillers: umm, matlab, woh, actually. Self-corrects mid-sentence |
| Impatient | 15% | Clipped, direct, no pleasantries, wants quick answers |
| Elderly | 15% | Formal, more Devanagari Hindi, may repeat key information |
| Confused | 15% | Repeats entity twice, seeks confirmation, says "samjhe na?" |

### Language Modes

| Mode | Description |
|---|---|
| `hinglish` (default) | Natural Hindi-English code-switching, 20–50% English per sentence, corpus average ~35% |
| `english` | Full English, airline formal register, personality markers still applied |

Airline domain terms are always in English regardless of mode: PNR, flight, booking, check-in, boarding pass, seat, upgrade, baggage, miles, lounge, visa, cancel, refund.

---

## Entity Injection

Entities are sampled fresh per conversation from fixed pools to ensure realistic variety:

| Entity | Pool Size | Examples |
|---|---|---|
| PNR | 15 | ABC123, XY4521, EY8834 |
| Flight Number | 15 | EY 101, EY 202, EY 303 |
| City | 16 | Mumbai/मुंबई, Dubai/दुबई, London/लंदन |
| Airport Code | 16 | AUH, DXB, BOM, DEL, LHR |
| Date | 18 | 15 January, pandrah January, agle hafte, kal |
| Travel Class | 6 | economy, business class, first class |
| Baggage Weight | 7 | 20 kilo, 23 kilo, 30 kilo |
| Miles | 5 | 5,000 miles, 25,000 miles, 1 lakh miles |

Dates are deliberately varied between English numerals (*15 January*), Hindi numerals (*pandrah January*), and colloquial forms (*15 tarikh*, *agle hafte*) to improve ASR robustness on date entity recognition.

---

## Generation Model

| Parameter | Value |
|---|---|
| Model | Qwen3-32B (q8_0 quantisation) |
| Inference | Ollama, local GPU |
| Thinking mode | Enabled (higher naturalness, better JSON structure) |
| Hardware | NVIDIA GPU |
| Estimated time | ~3.5 hours for full 320 conversations (both passes) |
| License | Apache 2.0 — commercial use permitted |

Two passes run in parallel on separate GPUs:

```bash
# Pass 1 — GPU 0
CUDA_VISIBLE_DEVICES=0 python 00_generate_text_corpus.py \
    --pass_id 1 --temperature 0.7 --port 11434

# Pass 2 — GPU 1
CUDA_VISIBLE_DEVICES=1 python 00_generate_text_corpus.py \
    --pass_id 2 --temperature 0.9 --port 11435
```

---

## Quality Filtering

A separate LLM judge (Qwen3-14B) scores each generated user turn on five criteria:

| Criterion | Description |
|---|---|
| Naturalness | Does it sound like a real caller? |
| Hinglish quality | Is code-switching natural, not forced? |
| Intent alignment | Does the sentence match the stated intent? |
| Entity usage | Are entities injected naturally? |
| TTS readiness | Is it phonetically speakable? |

Each criterion is scored 1–5. Sentences scoring below threshold are flagged for review or rejected before entering the training mix.

---

## Output Format

Each generated user turn is saved with full metadata:

```json
{
  "id": "A1_c01_t02",
  "conversation_id": "A1_c01",
  "primary_intent_id": "A1",
  "primary_sub_intent": "New Booking",
  "is_multi_intent": false,
  "personality": "hesitation",
  "turn_number": 2,
  "text": "umm... Mumbai se Dubai jaana tha, matlab 15 January ko, economy mein, seat milegi kya?",
  "intent": "A1",
  "entities_mentioned": ["CITY_FROM_EN", "CITY_TO_EN", "DATE", "CLASS"],
  "personality_marker": "filler + self-correction",
  "english_ratio": 35,
  "length_bucket": "medium",
  "status": "pending",
  "judge_score": null
}
```

Final corpus is exported as both JSON and CSV for downstream TTS synthesis and ASR training data preparation.

---

## Pipeline Steps

```
00  Text Generation        →  00_generate_text_corpus.py
01  Manifest Generation    →  01_generate_utterances.py
02  Evaluation Framework   →  02_evaluation_framework.py
03  TTS Synthesis          →  03_1_tts_preparation.py
                               03_2_tts_synthesis.py
04  Translation            →  04_en_to_hi_convert.py
05  Augmentation           →  05_telephony_augmentation.py
06  ASR Benchmark          →  06_asr_benchmark.py
07  Filter Training Data   →  07_filter_training_data.py
08  Training Data Prep     →  08_prepare_training_data.py
```
