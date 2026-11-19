# E42 official full fit, 19 November 2026

The Qwen3-14B-Base classifier completed three epochs on all 2,699 official training rows across 150 prompts. The training log records 1,009 successful optimizer updates and five skipped updates out of 1,014 planned. The packaging check then stopped the run before it could reload and verify the saved adapter.

The serializer recorded different padding settings in `tokenizer.json`. The full fit saved the tokenizer after single-text inference with padding disabled. The archived fold saved it with padding to the longest sequence in the batch. This field accounted for the entire parsed JSON difference. Offline CPU calls reproduced both byte identities. The other five tokenizer files matched their pinned hashes. The Base, fold and full-fit tokenizers produced identical token IDs and attention masks for all 2,699 official texts when encoding unpadded single texts, training batches of one and inference batches of eight.

A separate CPU check read the saved adapter and verified all 561 tensors, including the 18-class head. All 64,317,440 parameter values were finite. Across 291 supervisor samples, measured local storage growth peaked at 96 KiB and available local space stayed above 258.6 GiB.

The original three pre-save logit vectors, epoch losses and exact training resource measurements existed only in worker memory at failure. The full adapter therefore lacks the evidence required by the original release protocol. This run produced no Kaggle submission or leaderboard score.
