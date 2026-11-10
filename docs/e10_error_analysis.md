# E10 grouped OOF error analysis

This analysis uses only the saved, fixed 2,160 development OOF predictions from `run_score_normalization.py`: 120 prompts, five grouped folds, and 120 examples per class. It does not inspect the reserved holdout or hidden Kaggle labels. The full predictions and class metrics are in the Git-ignored `outputs/e10_fusion_normalization/` directory.

Row-standardized fusion changes 447 of 2,160 predictions relative to equal-rank fusion. Of these, 140 change from wrong to correct and 105 from correct to wrong, a net gain of 35. Since all 18 classes have 120 development rows, this equals the balanced-accuracy increase of 35 / 2,160 = 0.016204 (0.383333 to 0.399537). Thirteen classes gain recall and five lose recall.

| True class | Rank recall | Standardized recall | Correct change / 120 |
| --- | ---: | ---: | ---: |
| gpt-5.5 | 24.2% | 30.0% | +7 |
| gpt-5.6-luna | 42.5% | 52.5% | +12 |
| gpt-5.6-sol | 18.3% | 16.7% | -2 |
| gpt-5.6-terra | 27.5% | 25.8% | -2 |
| gpt-6-astra | 71.7% | 73.3% | +2 |
| gemini-3.1-pro-preview | 78.3% | 81.7% | +4 |
| gemini-3.6-flash | 48.3% | 55.8% | +9 |
| gemini-3.7-flash | 40.8% | 30.8% | -12 |
| gemini-3.8-flash | 28.3% | 30.0% | +2 |
| claude-fable-5 | 16.7% | 18.3% | +2 |
| claude-opus-4-6 | 56.7% | 62.5% | +7 |
| claude-opus-4-8 | 34.2% | 36.7% | +3 |
| claude-opus-5 | 53.3% | 55.8% | +3 |
| claude-sonnet-5 | 35.0% | 31.7% | -4 |
| qwen3.8-max | 47.5% | 50.0% | +3 |
| glm-5.3 | 17.5% | 19.2% | +2 |
| glm-5.3-flash | 26.7% | 25.0% | -2 |
| kimi-k3 | 22.5% | 23.3% | +1 |

The largest remaining directed confusions under standardized fusion are `gemini-3.7-flash → gemini-3.6-flash` (51/120, up from 35), `gpt-5.6-terra → gpt-5.6-luna` (39/120, up from 24), `gpt-5.6-sol → gpt-5.6-luna` (39/120, up from 34), and `gemini-3.8-flash → gemini-3.7-flash` (38/120, down from 41). Conversely, `gpt-5.6-luna → gpt-5.6-terra` falls from 21 to 13, and `gemini-3.6-flash → gemini-3.7-flash` falls from 26 to 19. The fusion rule therefore shifts some within-family decisions toward Luna and Gemini 3.6; it does not resolve version attribution symmetrically. These are descriptive counts, not a basis for changing class weights on the same OOF set.

The original equal-rank submission scored 0.376 on Kaggle's public split, 0.0073 below its 0.3833 grouped OOF score. Hidden labels and class composition are unavailable, so this aggregate gap does not identify a failing class or a specific correction. The next meaningful external check is the prespecified row-standardized submission with the same model weights. If its public result is also lower than expected, compare unlabeled train/test text and confidence distributions and seek independently sourced, route-verified examples for the version pairs above before fitting another pair-specific adjustment. E11's pair experts already showed only a small, uncertain OOF gain, and repeated tuning on these 120 prompts would overfit the diagnostic.
