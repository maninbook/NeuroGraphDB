# NeuroGraphDB

Graph retrieval for multi-hop QA that builds its graph **without a single LLM call**.

A passage-level graph where edges are literal title mentions ("passage A's body contains
passage B's title"). Retrieval seeds spreading activation from only the **top-5 dense hits**
and lets activation propagate. That is the whole method.

On HotpotQA it matches HippoRAG 2's retrieval quality while indexing the corpus in
**seconds instead of 74 minutes**, because it never calls an LLM to extract entities or triples.

This repository is a **research record**, not a polished library. Most of what we tried failed,
and the failures are documented as carefully as the successes — they are the more useful half.

---

## What is confirmed

All numbers below come from controlled comparisons: same passage pool, same questions in the
same order, same embedder (`BAAI/bge-base-en-v1.5`), same LLM, same top-k. Only retrieval changes.
Significance is McNemar's paired test throughout.

### Retrieval — fraction of questions where *all* supporting passages land in the top 10

n=1000, pool built from the union of all questions' passages (the HippoRAG-family setup).

| | dense only | **this method** |
|---|---|---|
| HotpotQA | 0.857 | **0.940** |
| 2WikiMultihopQA | 0.486 | **0.907** |
| MuSiQue | 0.266 | **0.350** |

### Head-to-head against HippoRAG 2

We ran [HippoRAG 2's published implementation](https://github.com/OSU-NLP-Group/HippoRAG)
ourselves under our conditions (n=500), rather than quoting its paper numbers. Its default
embedder is NV-Embed-v2 (7B); we replaced it with our 110M `bge-base` through its own
`BaseEmbeddingModel` interface so the comparison isolates the graph method rather than the
encoder. Its OpenIE and PPR are untouched.

| all-supporting@10 | dense | **ours** | HippoRAG 2 |
|---|---|---|---|
| HotpotQA | 0.874 | **0.940** | 0.934 |
| 2WikiMultihopQA | 0.534 | **0.920** | 0.796 |
| MuSiQue | 0.330 | 0.394 | **0.454** |

| indexing 4,943 passages | |
|---|---|
| HippoRAG 2 | **74 minutes** (2 LLM calls per passage: NER + triple extraction) |
| this method | **seconds** (string matching, 0 LLM calls) |

**We do not claim higher accuracy.** In end-to-end QA (Qwen2.5-72B, top-10, n=500) we never
beat HippoRAG 2 significantly on any dataset, and it beats us on MuSiQue (p=0.019). The
defensible claim is *comparable answer accuracy at a few thousandths of the indexing cost,
with better retrieval on corpora whose documents cross-reference each other by title.*

---

## The one idea that matters: seed sparsity

Seeding the spread from the top-5 dense hits rather than the top-20 is not a tuning detail.
It is the method.

| seeds | HotpotQA | 2Wiki | MuSiQue |
|---|---|---|---|
| 1 | 0.942 | 0.719 | 0.286 |
| **5** | 0.940 | **0.907** | **0.350** |
| 10 | 0.860 | 0.487 | 0.269 |
| 200 | 0.860 | 0.487 | 0.269 |
| *(dense alone)* | *0.857* | *0.486* | *0.266* |

Past ~10 seeds the graph collapses to plain dense retrieval, to three decimal places.

The reason is mechanical. A node reached by spreading receives `act(source) × w × decay`.
If that node is *already a seed* it holds its own dense score, which is always larger, and the
propagation rule keeps the maximum over paths. **Being a seed makes a node immune to the graph.**
Every seed you add removes one node from the graph's reach. Seed everything and the graph does nothing.

One seed is also bad: if the top dense hit is wrong, the spread starts from the wrong place.
Sparse but not singular.

In hippocampal terms this is pattern separation gating pattern completion — without sparse
coding there is nothing for completion to complete.

---

## The second thing that worked: query gating

Spreading originally ignored the question entirely. A neighbour received the same activation
whether or not it had anything to do with what was asked. Gating the spread by the neighbour's
similarity to the query is free — both vectors are already computed:

```
before   cand = act[u] × w × decay
after    cand = act[u] × w × decay × g(sim(query, v))
```

MuSiQue all-supporting@10 `0.341 → 0.355`, replicated across three fresh seeds
(gains 56, losses 13; p = 0.0043 / 0.0192 / 0.0009). R@20 rises consistently by ~4 points —
the gain is concentrated in deep ranks, which is exactly where diffuse spreading was adding noise.

No effect on HotpotQA or 2Wiki, and the reason is visible in the graph statistics: at 0.35–0.51
edges per node there is almost nothing to suppress. MuSiQue has 0.92. **Gating only matters on
corpora where the graph actually spreads.**

---

## What we tried and rejected

Seven mechanisms, each pre-registered with a primary hypothesis, a safeguard, and a
mechanism prediction written down *before* running. Full records in [RESULTS.md](RESULTS.md).

| mechanism | outcome |
|---|---|
| Hebbian reinforcement (retrieval strengthens edges) | **rejected** — gains 0, losses 20 (p<0.0001) |
| Predicate typing + polarity on edges | **rejected** — harmful at depth ≥2 |
| Proposition argument-sharing edges | **rejected at pre-measurement** — lift 78 vs 952 |
| Title alias keys (strip disambiguators) | **rejected** — mechanism confirmed, dilution won |
| Score-based merging of graph and dense | **rejected** — silently disables the graph |
| ACT-R activation summation | **rejected at pre-measurement** — no gold/noise contrast |
| ACT-R fan discount `S − ln(fan)` | **rejected** — contrast existed, ranking unchanged |

The Hebbian result is worth stating on its own, because adaptive reinforcement is frequently
listed as future work in this literature:

> Co-retrieval Hebbian strengthening has a **structural gain ceiling of zero** on standard
> multi-hop retrieval. Edges can only form between passages that were *already retrieved
> together*, so the mechanism cannot discover anything. Measured: 0 gains, 20 losses across
> 1000 questions, with rare topics hurt worst.

The pattern across all nine attempts is hard to miss. The two that worked (sparse seeding,
query gating) **remove** signal. The seven that failed **add** it.

---

## Measure before you build

Every experiment here was preceded by a cheap diagnostic that runs in seconds on CPU with no
LLM. Several of them killed an experiment before it cost anything, and one of them
(`job_whyedge.py`) found a bug in our own edge builder.

| script | question it answers |
|---|---|
| `job_whyedge.py` | Are the supporting passages even connected in this graph? |
| `job_hops.py` | If not directly, are they reachable in 2–3 hops? |
| `job_missing.py` | *Why* are the unconnected ones unconnected? (classifies + prints samples) |
| `job_degree.py` | Is there degree skew worth suppressing, and does it favour gold or noise? |
| `job_argmeasure.py` | Would a proposed edge type help, or just add noise? (gold vs random lift) |
| `job_converge.py` | Would summing convergent evidence change any ranking? |

The single most useful number is **gold-pair linkage**: the fraction of questions whose
supporting passages are directly connected. It predicts where this method works.

| | gold-pair linkage | random-pair | lift | our gain over dense |
|---|---|---|---|---|
| 2Wiki | 68.8% | 0.016% | 4437 | +38.6 pts |
| HotpotQA | 61.0% | 0.017% | 3587 | +6.6 pts |
| MuSiQue | 32.4% | 0.034% | 952 | +6.4 pts |

Gain requires **both** headroom left by dense retrieval **and** a route through the graph.
2Wiki has both. HotpotQA has the route but dense is already at 0.874. MuSiQue has headroom
but only 37% of gold pairs are reachable at any depth, and dense already delivers 33% —
its ceiling was nearly zero before we started.

Run the linkage check on your corpus before adopting this. It takes nine seconds.

---

## Applicability, stated plainly

Works when documents **cross-reference each other by title** — encyclopedias, internal wikis,
linked documentation, papers with citations. Does not work on corpora assembled from
independently-written passages, which is precisely what MuSiQue is.

Improves ranks **6–20**. A pipeline that reads only the top 5 sees exactly zero benefit —
the top of the ranking is always plain dense.

2Wiki's large margin is partly an artifact: it was generated from Wikidata relation templates,
so its supporting pairs are almost always in a title-mention relation. Do not expect 38 points
on your own corpus.

Retrieval gains do not automatically become answer gains. With a 72B reader, a 12.4-point
retrieval lead on 2Wiki shrank to 2 points of EM. Match the ranks you improve against the
ranks your pipeline actually reads.

---

## Layout

```
neurographdb/src/     C++ core (BM25, dense index, graph spreading) via pybind11
neurographdb/job_*.py self-contained HF Jobs scripts, one per experiment
RESULTS.md            full research log, dated, in Korean — including every failure
RESEARCH.md           literature survey that started the project
PROPOSITION.md        pre-registration for the propositional layer (rejected)
```

Indexing, spreading and scoring run in C++; Python handles data preparation and orchestration.
Raw results for every run are published at
[goethe0101/neurographdb-results](https://huggingface.co/datasets/goethe0101/neurographdb-results).

Every experiment ran on Hugging Face Jobs; each `job_*.py` carries its dependencies in a
PEP 723 header and rebuilds the C++ core inside the container.

```bash
cd neurographdb && make
hf jobs uv run --flavor l4x1 --secrets HF_TOKEN job_graph.py hotpotqa 1000
```

## Reproducibility notes

- `hipporag==2.0.0a4` cannot be installed as published — it pins `openai==1.91.1`, which does
  not exist on PyPI. We override that one pin and nothing else.
- Its released wheel does not accept an injected embedding model, so we install ours through
  its documented `BaseEmbeddingModel` interface.
- Small embedding drift between job runs (≈0.007 on MuSiQue all@10) traces to library versions
  and GPU non-determinism. It affects all conditions in a run identically; every experiment
  re-verifies its control against previously recorded numbers before the result is interpreted.

## License

MIT
