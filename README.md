# Target Speech Enhancment through Positive Negative Enrollment with FlowSE

Process:

1. Get `cond_emb` from pretrained model: `proposed-monaural.pt`, which acts as encoder.
2. Pass `cond_emb` to [`PNViT`](./sia_fm_tse/adapter/model/pn_conditioner.py), and combines with FlowSE

But okay, I think we need to *reboot*.
