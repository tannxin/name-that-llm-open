# E30 numeric fidelity on RTX 3090 and Tesla T4

Forcing the same PyTorch math attention backend on both GPUs left E30 outside the fixed deployment tolerance on **all 12 training texts**. The default attention operator did differ between the devices, but matching that operator was insufficient to reproduce the encoder outputs or the classifier logits.

## Measured differences

Each comparison below covers the same 12 texts, 4,096 pooled encoder values per text and 18 raw logits per text. The predeclared absolute tolerance was 0.002 with zero relative tolerance. A text exceeded tolerance if any element of the stated output exceeded it.

| Paired comparison | Max absolute pooled difference | Max absolute logit difference | Texts over tolerance, pooled / logits | Same top-1 class |
| --- | ---: | ---: | ---: | ---: |
| 3090 vs. T4, both default | 0.078125 | 0.01144218445 | 12 / 12 | 12 of 12 |
| 3090 vs. T4, both forced math | 0.08203125 | 0.01989364624 | 12 / 12 | 12 of 12 |
| Default vs. forced math on the 3090 | 0.1171875 | 0.01234322786 | 12 / 12 | 12 of 12 |
| Default vs. forced math on the T4 | 0.09375 | 0.01568269730 | 12 / 12 | 12 of 12 |

Within each tolerance-count cell, the first number counts texts for pooled outputs and the second counts texts for logits. All four sets of 12 predicted classes had unique maxima. This top-1 agreement is an unlabelled diagnostic on the E30 component. Accuracy, decisions after fusion with E24, and fidelity on unseen examples remain unmeasured by this experiment.

## Evidence and interpretation

The PyTorch profiler recorded Flash attention for all 12 default 3090 forwards and efficient attention for all 12 default T4 forwards. With math attention explicitly selected, all 24 forwards across the two devices recorded the math operator. The default 3090 outputs matched the archived 3090 baseline bit for bit for both pooled values and logits.

The two captures used the same locked fold-1 model, adapter, classifier parameters, training texts, token IDs, masks, pooling positions, class order, model library versions and NF4 settings. The local audit verified their identities and compared the four finite output matrices from each device. The T4 report and arrays matched the checksums printed by the original private GPU run. The receipt matched the checksum from the subsequent private CPU extraction.

Changing the attention backend changed the outputs on each device. In the controlled cross-device comparison, the largest deviations for both pooled values and logits persisted with the math operator on both GPUs. The experiment does not identify which other numerical operations contribute, or how much of the default drift is attributable to attention. Layer-level or quantized matrix multiplication probes would be separate experiments.

The private Kaggle version completed in 523.9 seconds. The capture ran on GPU 0 of the two-T4 environment. Its outer monitor sampled both cards 36 times over 180.0 seconds, observed peaks of 7,607 MiB and 105 MiB, and recorded a maximum working-space decline of 252 KiB during that interval. The 12-text fidelity gate failed. This diagnostic supplies no basis for promoting E30 to full training or a Kaggle submission.
