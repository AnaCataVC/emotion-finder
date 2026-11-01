---
name: emotion-finder-curate-feedback
description: Curates, audits, and filters human-in-the-loop feedback submissions before promoting samples to retraining datasets.
tools:
  - view_file
  - grep_search
  - write_to_file
  - replace_file_content
  - run_command
---

# Emotion Finder Feedback Curation Skill

This skill guides maintainers through the review, sanitization, and promotion workflow for user feedback records collected via the active learning loop in Emotion Finder.

## Purpose & Scope

User corrections submitted via the leaf card interface are stored in `data/feedback.db` with an initial status of `pending`. Before these records can be merged into `data/emotions_es_v2.csv` or `data/emotions_en_v2.csv` for model retraining, they must pass strict anti-poisoning verification.

## Step-by-Step Workflow

### Step 1: Query Pending Feedback Records
Run a dry-run curation session to assess pending record volume:
```bash
python scripts/curate_feedback.py --status pending --limit 50 --dry-run
```

### Step 2: Quality & Anti-Poisoning Validation Checklist
For each pending candidate, verify:
1. **Linguistic Plausibility:** The input text describes genuine physical, physiological, or emotional sensations in Spanish or English.
2. **Sentiment Consistency:** The corrected quadrant matches the emotional valence and arousal implied by the text.
3. **Whitelist Adherence:** Key sentiment-carrying tokens are not stripped by `preprocessing.py` whitelists.
4. **Volume Ceiling:** Ensure the batch to be approved does not exceed 10% of the baseline dataset size (700 samples per language = maximum 70 approved records per cycle).

### Step 3: Interactive Record Review
Execute the CLI curation tool:
```bash
python scripts/curate_feedback.py --status pending --limit 50
```
- Press `v` to approve and mark record as `verified`.
- Press `r` to reject and mark record as `rejected`.
- Press `s` to skip ambiguous records for later review.
- Press `q` to save progress and exit.

### Step 4: Retraining Dataset Integration
Once verified records are staged, initiate the retrain pipeline:
```bash
python scripts/retrain_from_feedback.py
```
Verify that the output logs confirm successful dataset augmentation and that cross-validation benchmarks pass before committing changes.
