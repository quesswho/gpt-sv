# gpt-sv

Training and inference code for a 430M parameter Swedish language model
trained on 10.5B tokens from FineWeb2-swe.

[write-up](https://quesswho.github.io/miles-blog/2026/09/15/gpt-sv/)

| Task | Metric | ours (430M) | Qwen2.5-0.5B |
|---|---|---|---|
| HellaSwag-sv | acc_norm | **38.2 ± 0.5** | 30.4 ± 0.5 |
| ARC-sv | acc_norm | 28.2 ± 1.3 | 25.8 ± 1.3 |
| Belebele (swe_Latn) | acc | 23.2 ± 1.4 | **30.6 ± 1.5** |
| Global-MMLU-sv | acc | 22.9 ± 0.4 | **33.1 ± 0.4** |

