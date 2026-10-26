---
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: peft
tags:
- lora
---

# CauProbe text expert (LoRA adapter)

LoRA adapter of Qwen2.5-0.5B-Instruct used as the on-device text expert in
*Ask, probe, or act? Learning-aware value of information for self-commissioning
IoT gateways*. Given a gateway's commissioning notes, it answers one yes/no
question per mechanism factor; the logit difference of the two answer tokens is
used, after temperature calibration, as a likelihood ratio in the controller's
posterior. Training and scoring: `causlm/probe/slm_expert.py`,
`experiments/cp_expert.py`. The base model is downloaded separately from
Hugging Face; set `HF_DIR` in `experiments/cp_common.py`.
