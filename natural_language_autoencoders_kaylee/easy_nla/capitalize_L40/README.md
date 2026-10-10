# Capitalize round trip at layer 40

An ad hoc follow-up to the NLA reconstruction-robustness experiment
(`notebooks/executor.py`, layer 32). At layer 32 the same-task mismatched
function vector still solved the capitalize task about 40% of the time. The
hypothesis was that the model has enough layers left after layer 32 to recover
from the wrong vector. This reruns the capitalize task with an NLA trained on
Gemma-3-12B-IT layer 40 ([`achand45/gemma-3-12b-it-nla-L40`](https://huggingface.co/achand45/gemma-3-12b-it-nla-L40)).

## Results (n = 50)

| arm | correct |
|---|---|
| baseline | 50 |
| nla_roundtrip | 48 |
| paraphrase_medium | 44 |
| paraphrase_heavy | 46 |
| mismatch_same_task | 0 |
| mismatch_cross_task | 0 |
| random_vec | 1 |

The same-task mismatch drops to 0, which supports the hypothesis.

## Files

| file | what it is |
|---|---|
| `nla_roundtrip.py` | the experiment: runs every arm and draws the bar chart |
| `capitalize_results.json` | correct counts per arm |
| `capitalize_roundtrip.png` | the figure |
| `capitalize_run.log` | the run's output, including each explanation |

## Rerun

From `easy_nla/`, with the layer-40 checkpoint downloaded to `./nla-L40`:

```bash
python capitalize_L40/nla_roundtrip.py --nla-dir ./nla-L40/rl_vllm \
    --data-dir notebooks/datasets --n 50 \
    --out capitalize_L40/capitalize_roundtrip.png \
    --results-json capitalize_L40/capitalize_results.json
```
