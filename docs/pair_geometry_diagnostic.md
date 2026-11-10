# Development-only pair geometry diagnostic

This report uses the fixed 120 development prompts (2,160 answers, 120 per class) and the saved E10 grouped out-of-fold predictions. The 30 held-out prompts were excluded before feature extraction, vectorizer fitting, scaling, and distance computation. No new classifier was trained and no Kaggle test labels were used.

## Measures

- Character distance is `1 − cosine` for character 3–5-gram TF-IDF. The 100,000-feature vocabulary is fitted on development text only. Style distance is Euclidean distance over the existing 57 standardized style features, divided by √57.
- Closeness percentile ranks each pair's distance among all 153 model pairs *within the same prompt*. 0 is the closest pair and 100 the most distant. Values below 50 mean a pair is more similar than a typical pair in the same-prompt comparison. The table reports means across prompts; 95% prompt-bootstrap intervals and five-fold ranges are in the CSV.
- Direction consistency `R = ||mean(x_a − x_b)|| / sqrt(mean(||x_a − x_b||²))` ranges from 0 to 1. Higher values mean the pair's feature difference points in a more stable direction across prompts. With 120 prompts, random sign flips have R² expectation 1/120, corresponding to an RMS reference of 0.091 for R. Values near this reference offer little evidence of a stable direction. R is a descriptive signal, not a classification score.
- `errors` gives E10 OOF predictions A→B + B→A. The two directions matter; proximity alone does not establish that increasing a loss weight will help.

## Most confused pairs

| pair | errors | char_pct | style_pct | char_R | style_R |
| --- | --- | --- | --- | --- | --- |
| gemini-3.6-flash ↔ gemini-3.7-flash | 19+51 | 17.0 | 26.2 | 0.095 | 0.169 |
| glm-5.3 ↔ glm-5.3-flash | 31+24 | 35.5 | 49.5 | 0.091 | 0.101 |
| gemini-3.7-flash ↔ gemini-3.8-flash | 15+38 | 21.7 | 28.4 | 0.095 | 0.164 |
| gpt-5.6-luna ↔ gpt-5.6-terra | 13+39 | 25.4 | 37.1 | 0.094 | 0.128 |
| gpt-5.6-luna ↔ gpt-5.6-sol | 7+39 | 20.3 | 40.6 | 0.094 | 0.12 |
| gemini-3.1-pro-preview ↔ gemini-3.6-flash | 16+27 | 22.3 | 30.2 | 0.101 | 0.285 |
| claude-fable-5 ↔ glm-5.3 | 23+19 | 29.3 | 43.4 | 0.092 | 0.101 |
| claude-opus-4-8 ↔ claude-sonnet-5 | 17+23 | 25.1 | 44.8 | 0.097 | 0.211 |
| gpt-5.6-luna ↔ gpt-6-astra | 22+17 | 28.9 | 43.8 | 0.101 | 0.2 |
| gemini-3.6-flash ↔ gemini-3.8-flash | 5+30 | 25.0 | 27.9 | 0.097 | 0.219 |
| gpt-5.6-sol ↔ gpt-5.6-terra | 23+12 | 20.9 | 38.5 | 0.093 | 0.114 |
| gpt-5.5 ↔ gpt-5.6-terra | 24+8 | 31.9 | 39.8 | 0.098 | 0.182 |

## Overall relationship

Across all 153 pairs, Spearman correlation between closeness percentile and mutual OOF confusion rate is **-0.740** for character TF-IDF and **-0.592** for style features. Negative values mean closer pairs tend to be confused more often. This is descriptive because the same development set informed the geometry and earlier model choices.

Within the 27 same-family pairs, the correlations are **-0.654** (character) and **-0.621** (style). Across the 126 cross-family pairs they are **-0.588** and **-0.476**, respectively. Thus the overall relationship is not solely a family-versus-non-family effect.

The full 18×18 mean distance matrices, 18×18 E10 confusion matrix, and all 153 pair metrics (including prompt-bootstrap intervals and fold ranges) are in `outputs/pair_geometry_diagnostic/` (Git-ignored).

## Interpretation for later experiments

Use the same-prompt distances to identify candidates, then compare them with directed OOF confusion and direction consistency. A very similar pair can still be separable if its differences are stable; a distant pair can remain hard if its difference changes with prompt. Any proposed loss weights must be estimated inside each training fold without validation labels, fixed before evaluating that fold, and checked against unweighted cross-entropy under the same backbone and training schedule. This diagnostic is not an unbiased estimate of a new loss's gain.

The competition test file has only `id,text`; same-prompt comparison is available in training, but cannot be required at inference.
