# Reproducing the 0.684 public score on Kaggle

Our reproduction scored **0.684 on the Kaggle public leaderboard**, matching the author's V1 to the three decimal places Kaggle displays. The gain over our previous best, E24+E42 at 0.452, was **0.232**. We submitted notebook version `371798931` once on November 20, 2026 at 13:22:24 UTC. Kaggle marked the hidden-test submission `Succeeded`. [Author’s scored V1](https://www.kaggle.com/code/wadigongren/name-that-llm-0684-clean-baseline?scriptVersionId=368242626)

| Input | What ran | Verified result |
|---|---|---|
| 64 synthetic, unlabeled texts | Full inference with 35 mounted sources on two offline T4s | Completed in 5,319.6 seconds. The complementary and main inference workers and both process supervisors returned exit code 0. An independent CPU audit passed 17 checks, including a finite 64 × 18 complementary score matrix and valid predictions for 64 unique IDs in input order. |
| Official three-row public sample | The author’s short-input path in the submission candidate | Completed in 390.8 seconds. Each of three sparse classifiers produced a 3 × 18 score matrix. The sample path writes a fixed label for all three rows. Independent Parquet readback verified the official IDs, order and schema. |

On the 64 texts, all five main decoder members and DeBERTa completed. The complementary stack had zero rows eligible for its group-only heads. In the separate main ensemble, the Qwen3-14B and Qwen3-8B Gaussian heads each processed two eligible rows. ATLAS was off in the author’s configuration. The final Qwen3-Embedding branch changed eight first choices. The synthetic texts have no reference labels, so the effect on accuracy is unknown.

The reproduction includes checkpoint-path and file-handling fixes. We verified the mounted sources for this run. The score matches at the displayed precision. Whether the two runs produced identical predictions for each hidden row, used identical historical asset bytes, or obtained the same gain from each branch remains unknown.

Our separate local five-fold study used 120 development prompts with 30 previously used holdout prompts excluded. On frozen Qwen3-Embedding-4B features, the Gaussian head raised standalone balanced accuracy from 0.207407 to 0.300000. Adding it to E24+E42 raised balanced accuracy from 0.443519 to 0.452778, below the prespecified gain of 0.010. The local experiment used a different representation and grouping method. Its results measure transfer to our models. [Component transfer results](https://github.com/tannxin/name-that-llm-open/blob/896ae363d958f68fed1861c15ed706d3fb5f0b02/docs/public_0684_component_transfer_2026-11-20.md)
