# E11: fixed soft experts for two sibling-version pairs

## Predefined protocol (written before the E11 run)

The saved E0c confusion matrix on the 120 development prompts shows 35+38 mutual errors for `gemini-3.7-flash` versus `gemini-3.8-flash`, and 18+25 for `glm-5.3` versus `glm-5.3-flash`. These are the two pairs for this diagnostic. Choosing them after viewing an earlier development confusion matrix introduces selection bias; the E11 evaluation is exploratory.

For each of the existing five prompt-grouped folds, train one E0c feature pipeline and binary LinearSVC per pair on **only the other four folds' examples from that pair**. Do not use the held-out prompt group in fitting, scaling, margin normalization, feature selection, or model selection. The two pairs have the same fixed recipe; their OOF margins are computed on all 18 responses per validation prompt. Divide each expert's margin by the median absolute training margin (or 1 if that median is zero), then apply `tanh`.

Apply the following fixed adjustment independently to the existing equal-rank and row-z-score E0c + corrected RoBERTa fusion scores:

`delta = 0.5 * exp(-max(0, top_score - best_pair_score) / 2) * tanh(expert_margin / training_median_abs_margin)`.

Add `delta` to the first label of the pair and subtract it from the second. Both versions get a continuous score adjustment; there is no hard family route or label-conditioned fusion weight. The two constants, 0.5 and 2, are fixed before looking at E11 OOF results. Report each pair alone and the combined two-pair adjustment against the respective no-expert fusion, without choosing a weight using OOF labels. The stronger z-score no-expert reference scored 0.399537 and the rank reference scored 0.383333 before this run.

Use the original 2,160 development rows, 120 prompts, and five folds. Report pooled 18-class balanced accuracy, all five fold scores, per-class recall, and 20,000 paired prompt bootstrap draws. The 30-prompt reserved holdout is not scored. Do not use any E11 candidate for Kaggle unless its benefit over the stronger z-score reference is convincing and separately packaged.

## Results

Run `python run_soft_experts.py` from the project root. The script first verifies the original E7e-R rank predictions and the E10 z-score predictions exactly, then fits ten binary experts across the fixed five folds. All candidate margins are OOF, and no parameters were changed after the run. The local result files are in `outputs/e11_soft_experts/`.

| Fusion | No expert | Gemini only | GLM only | Both pairs | Both vs no expert, paired prompt 95% interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| Equal rank | 0.383333 | 0.383796 | 0.384259 | 0.384722 | +0.001389, [-0.004167, 0.006944] |
| Row z-score | 0.399537 | 0.400926 | 0.402778 | 0.404167 | +0.004630, [-0.002315, 0.011111] |

The five z-score fold scores without the experts were 0.368056, 0.402778, 0.449074, 0.372685, and 0.405093. With both experts they were 0.372685, 0.412037, 0.439815, 0.384259, and 0.412037. Thus four folds improved and one declined. The pooled z-score class recalls for the targeted labels changed as follows: Gemini 3.7 from 0.3083 to 0.3250, Gemini 3.8 from 0.3000 to 0.3333, GLM 5.3 from 0.1917 to 0.2500, and GLM Flash from 0.2500 to 0.2917. The adjustment also changed some unrelated predictions: Gemini 3.6 fell from 0.5583 to 0.5333 and Claude Fable 5 from 0.1833 to 0.1667. All 18 per-class recalls and fold scores are saved in the output directory.

The gain is small, and the paired prompt interval spans zero even before accounting for choosing the two pairs from an earlier development confusion matrix and testing multiple variants here. **Do not promote these experts to the Kaggle submission at this stage.** The fixed row-z-score fusion without experts remains the simpler candidate. No reserved holdout score was produced.
