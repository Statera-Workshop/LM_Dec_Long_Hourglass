# LM_Dec_Long_Hourglass

Hourglass - Long family, decoder-only byte-level language model, standardised in the Statera Working Group Language Model library.

## Status

- Verification level: **Benchmarked (4 seed-repeat runs, see results)**
- Failure class: none

## Provenance

- Source repository: LM_2021_hourglass-transformer-pytorch
- Upstream / license: lucidrains/hourglass-transformer-pytorch (MIT)

## Differences from the source ([Unified] protocol)

- depth (2, 2, 2) instead of (4, 2, 4) to match the 6-layer budget

All models share one protocol: 256-byte vocabulary, dim 512, 6 layers, 8 heads x 64, Adam 2e-4 with 200 warm-up steps, gradient clip 0.5, bf16 autocast (automatic fp32 fallback). Numbers rank structures under one budget on one GPU (RTX 3070 Ti 8 GB); they do not reproduce paper results.

## Results

- Parameters: 26.52 M
- Causality check: pass (largest change of earlier outputs 0.0e+00)
- Overfit check (4 x 256 bytes): 5.705 -> 0.001 nats, pass
- LM-02 enwik8 (200M tokens): valid 1.517 / test 1.490 bits per byte
- LM-02 enwik8 (1B tokens): valid 1.354 / test 1.336 bits per byte
- Peak GPU memory at length 2048 (batch 1): train 1387 MB, inference 402 MB
- Peak GPU memory at length 8192 (batch 1): train OOM, inference 5364 MB
- LM-05 recall `lm05v3`: 4 pairs @ 512: 0.26 (keys removed 0.25); 4 pairs @ 2048: 0.25 (keys removed 0.28); 16 pairs @ 512: 0.07 (keys removed 0.06); 16 pairs @ 2048: 0.07 (keys removed 0.07)
- LM-05 recall `lm05v3long`: 4 pairs @ 512: 1.00 (keys removed 0.23); 4 pairs @ 2048: 1.00 (keys removed 0.22); 16 pairs @ 512: 0.07 (keys removed 0.07); 16 pairs @ 2048: 0.07 (keys removed 0.06)
- LM-06 memory update `lm06`: @512: latest 0.70, unchanged 0.04, stale 0.01; @2048: latest 0.63, unchanged 0.05, stale 0.01
- LM-06 memory update `lm06long`: @512: latest 0.74, unchanged 0.05, stale 0.01; @2048: latest 0.67, unchanged 0.05, stale 0.01
- LM-06 memory update `lm06s2`: @512: latest 0.63, unchanged 0.05, stale 0.02; @2048: latest 0.52, unchanged 0.05, stale 0.03
- LM-06 memory update `lm06s3`: @512: latest 0.66, unchanged 0.04, stale 0.01; @2048: latest 0.58, unchanged 0.04, stale 0.02

## Tests

- check: data counts, causality (no look-ahead, outputs use the current byte), parameters
- scaling: peak GPU memory and time vs length 512-8192
- overfit: memorise 4 x 256 bytes below 0.5 nats
- LM-02: enwik8 next-byte prediction (bits per byte)
- LM-05: long-horizon key-value recall with key-removed control (v3: distinct queries)
- LM-06: memory update - latest value vs stale value after a key is reassigned

## Run

```bash
pip install torch==2.5.1 einops numpy
python run.py check hourglass
python run.py lm02 hourglass      # enwik8 is downloaded to data/ on first use
```
