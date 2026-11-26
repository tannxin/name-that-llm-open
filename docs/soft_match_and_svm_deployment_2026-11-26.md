# Soft matching comparison and SVM deployment checks

2026-11-26. The staged Qwen3-14B and E24 ensemble was evaluated with the same weights and unary scores in two Kaggle submissions.

| Final decision rule | Public score |
| --- | ---: |
| Automatic grouping and fixed soft matching | 0.457 |
| Independent per-answer argmax | 0.457 |

The public leaderboard displays both scores to three decimal places. This comparison does not resolve a difference smaller than its display precision. The public score is separate from the prompt-grouped offline evaluation.

The full-pool four-view SVM was trained on 250,634 external rows, followed by official-data fine-tuning. A private Kaggle CPU notebook with Internet disabled verified the model and code against fixed SHA-256 digests. On four synthetic texts it produced finite 4×18 scores and matched the local reference with maximum absolute difference 0.0. The successful private script version was 374331291.

A separate private CPU resource probe, script version 374336028, processed 4,000 text inputs with 18 scores in 5.711 seconds. It cycled through the official training texts and used a synthetic baseline to exercise fixed-coefficient fusion. The probe was unscored and measures only the SVM and fusion operation. It does not establish the latency or validity of a combined 14B submission.

The 14B and SVM combined pipeline requires a verified mapping from the SVM class order to the competition class order. The actual combined pipeline and its full Kaggle runtime remain to be validated.
