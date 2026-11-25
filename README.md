# VLMs on MECCANO: from 3D-CNNs to vision-language models

**Status: work in progress.** An exploratory study of how an open vision-language model (VLM) compares with small, compressed 3D-CNNs for fine-grained worker-activity recognition in industrial assembly. No results yet; the tables below are filled in as experiments finish.

Follows my earlier work on efficient video models for the same task:
- PDNet (VISAPP 2026): https://github.com/mscmalekbaba/PDNet
- Master's thesis code: https://github.com/mscmalekbaba/Master-Thesis-Resource-Efficient-PySlowFast

## Question

PDNet compresses X3D-M with pruning and distillation to 1.80M parameters and reaches 39.46% Top-1 on MECCANO. How does an open VLM do on the same task, what does it cost, and can it help train a small model?

1. **Zero-shot:** how far does a VLM get with no training, only a prompt?
2. **Few-shot adaptation:** how much does LoRA fine-tuning on a small fraction of the labelled clips help?
3. **Efficiency:** accuracy versus parameters, memory and latency, next to PDNet.
4. **Compression:** what does 4-bit quantization cost in accuracy, and can the VLM act as a teacher for a small student?

## Pipeline

The thesis pipeline (train, prune, distill, benchmark) carries over:

| Thesis (3D-CNN) | This repo (VLM) |
| --- | --- |
| Train X3D-M baseline | Zero-shot, then LoRA-adapt Qwen2.5-VL-3B |
| Structured pruning | 4-bit quantization of the VLM |
| Distillation from SlowFast | Distillation from the VLM into X3D-M |
| Benchmark params, FLOPs, CPU time | Same measurements for every model, one table |

To get soft labels from a text-generating model, each of the 61 classes gets a short code, and the VLM's probabilities over the code tokens (one forward pass) serve as the teacher distribution.

## Setup

- **Model:** Qwen2.5-VL-3B-Instruct (a newer small Qwen-VL model if it is a drop-in swap).
- **Data:** MECCANO action recognition, 61 classes, same train/test split as PDNet. The dataset is not included; request access from the [dataset authors](https://github.com/fpv-iplab/MECCANO).
- **Zero-shot protocol:** each clip is sampled to a fixed number of frames and the model is asked to choose one action from the fixed list of 61 classes. The answer is mapped to a class, and Top-1 accuracy is computed.
- **Adaptation:** LoRA fine-tuning on 5-10% of the training clips.
- **Metrics:** Top-1 and mean per-class accuracy, parameters, GPU memory, latency per clip.

## Results

| Model | Training | Top-1 (%) | Mean per-class (%) | Params (M) | Latency / clip |
| --- | --- | --- | --- | --- | --- |
| X3D-M, pruned + KD (PDNet) | full train set | 39.46 | 10.95 | 1.80 | n/a |
| Qwen2.5-VL-3B | zero-shot | TBD | TBD | ~3000 | TBD |
| Qwen2.5-VL-3B | LoRA, few-shot | TBD | TBD | ~3000 | TBD |
| Qwen2.5-VL-3B, 4-bit | LoRA, few-shot | TBD | TBD | ~3000 | TBD |
| X3D-M student | distilled from VLM | TBD | TBD | 3.09 | TBD |

PDNet numbers are from the thesis (Table 5.5). Latency is measured on different hardware, so compare orders of magnitude, not exact values.

## Plan

Core (first):
- [ ] Data loader: MECCANO clips to frames
- [ ] Zero-shot evaluation script and prompt
- [ ] Results table and error analysis (which classes fail)
- [ ] LoRA fine-tuning on a small subset
- [ ] Efficiency measurements

Next:
- [ ] 4-bit quantization of the adapted VLM
- [ ] Distillation from the VLM into X3D-M

Future work:
- [ ] Temporal grounding on longer MECCANO videos: ask when an action happens and compare the predicted time range with the annotated segment (temporal IoU)

## Repository layout (planned)

```
configs/    prompts and experiment settings
src/        data loading, inference, LoRA training, distillation, evaluation
results/    raw predictions and metrics (CSV/JSON)
```

## Limitations

Single runs, no confidence intervals yet. MECCANO is long-tailed and egocentric, which is hard for general-purpose VLMs; low zero-shot numbers are expected and are part of the finding.

## Acknowledgements

[Qwen-VL](https://github.com/QwenLM/Qwen2.5-VL), [MECCANO](https://github.com/fpv-iplab/MECCANO).
