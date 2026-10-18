# `data/`

Input files live here. They are **not** committed (see `.gitignore`); supply them
locally before running the notebooks. All notebook paths are relative to the
repository root, so launch Jupyter from the repo root.

## Required

| File | Description |
|---|---|
| `combined_acl_arxiv_with_citations_final.csv` | The master corpus: deduplicated ACL Anthology + arXiv (cs.CL) papers with titles, abstracts, year, and citation counts. **Everything else is derived from this.** |

Download the corpus from Hugging Face:

```python
from datasets import load_dataset
ds = load_dataset("wxyzabcd/acl-arxiv-cscl-citation-impact", split="train")
ds.to_pandas().to_csv("data/combined_acl_arxiv_with_citations_final.csv", index=False)
```

**Label — `is_highly_cited`:** not shipped with the corpus; derive it from the
`citation_count` and `publication_year` columns. A paper is `1` if its citation count is
at or above the **80th percentile within its publication year** (top 20% that year), else
`0`. Percentile-within-year normalization removes the age bias where older papers accrue
more citations. Resulting positive rate ≈ 20.7%. (arXiv citation counts were fetched via
the Semantic Scholar batch API.)

## External (optional, for the LiveIdeaBench evaluation)

| File | Description |
|---|---|
| `liveideabench_hf.csv` | LiveIdeaBench v2 ideas (public HuggingFace dataset). Only needed for `insights/LiveIdea.ipynb`. |

## Generated from the corpus (produced by the notebooks)

| File | Produced by |
|---|---|
| `specter_embeddings.csv` | SPECTER encoder (`insights_utils.generate_specter_embeddings`) |
| `scibert_embeddings.csv`, `bert_embeddings.csv` | respective encoders |
| `targets.csv` | year-normalized citation labels from the corpus |
| `abstract_stylometric_readability_features.csv` | `insights_utils.compute_stylometric_features` |
| `entropy_scores_2.csv` | GPT-2 cross-entropy, written by `finetune_gpt2_scidomain.ipynb` |
| `gpt2_medium_scidomain/` | fine-tuned GPT-2 medium (Surprise model), saved by `finetune_gpt2_scidomain.ipynb` |
| `combined_scores.csv` | `insights/inference.ipynb` |
| `liveidea_filtered.csv` | `insights/LiveIdea.ipynb` |

Files that are already tracked in the repo and need not be regenerated:
`../train_val_test_split.json`, `../kg/mutual_edges@20.npy`, and the trained models
under `../val_pred/results/` and `../insights/results/`.
