---
name: emotion-finder-sync-emotions
description: Synchronizes and validates emotion entries across the somatic decision tree, semantic matcher, and bilingual datasets.
tools:
  - view_file
  - grep_search
  - find_by_name
  - write_to_file
  - replace_file_content
---

# Emotion Finder Emotion Synchronization Skill

This skill defines the multi-file synchronization protocol required when adding, modifying, or removing emotion labels in Emotion Finder.

## Architecture & Consistency Invariants

Emotion Finder relies on a strict 1:1 mapping across 4 decoupled components:
1. **Somatic Decision Tree (`decision_tree.py`):** Exactly 64 emotions distributed uniformly as 16 emotions per Russell circumplex quadrant (`alta_positiva`, `alta_negativa`, `baja_positiva`, `baja_negativa`). Each terminal node contains `emotion_es`, `emotion_en`, `emoji`, `description_es`, and `description_en`.
2. **Semantic Emotion Matcher (`emotion_matcher.py`):** TF-IDF definitions dictionary used for direct cosine similarity scoring.
3. **Bilingual Corpus Datasets (`data/emotions_es_v2.csv` and `data/emotions_en_v2.csv`):** Balanced synthetic and curated training samples per quadrant and emotion.
4. **FastHTML UI & Session State (`main.py`):** Display templates rendering leaf cards and breadcrumb navigation.

## Synchronization Protocol

### Step 1: Modify Somatic Decision Tree Nodes
Update the target quadrant tree in `decision_tree.py`:
- Ensure 4-step binary questions preserve the somatic sequence: Body Tension -> Heart Rate -> Cognitive Focus -> Action Impulse.
- Verify leaf payload keys:
  ```python
  {
      "emotion_es": "Nombre en Español",
      "emotion_en": "English Name",
      "emoji": "Icon",
      "description_es": "Clinical somatic definition in Spanish.",
      "description_en": "Clinical somatic definition in English."
  }
  ```

### Step 2: Update Semantic Matcher Definitions
Update `emotion_matcher.py` with expanded semantic descriptions for both languages to maintain TF-IDF matching fidelity.

### Step 3: Rebalance Training CSV Datasets
Synchronize `data/emotions_es_v2.csv` and `data/emotions_en_v2.csv`:
- Ensure row format: `text,quadrant,emotion`.
- Maintain balanced quadrant representation (700 samples per language across quadrants).

### Step 4: Verification and Parity Test Execution
Run test suite to verify tree topology and quadrant integrity:
```bash
pytest tests/test_pipeline.py -k "test_decision_tree" -v
```
Confirm all 64 leaf nodes are reachable with zero broken references or null descriptions.
