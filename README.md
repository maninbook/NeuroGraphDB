# NeuroGraphDB

Graph retrieval for multi-hop QA that builds its graph **without a single LLM call**.

A passage-level graph where edges are literal title mentions ("passage A's body contains
passage B's title"). Retrieval seeds spreading activation from only the **top-5 dense hits**
and lets activation propagate. That is the whole method.

On HotpotQA it matches HippoRAG 2's retrieval quality at a small fraction of the indexing
cost, because it never calls an LLM to extract entities or triples.

This repository is a **research record**, not a polished library. Most of what we tried failed,
and the failures are documented as carefully as the successes — they are the more useful half.

Where the project is going, and what it has stopped chasing, is in [STRATEGY.md](STRATEGY.md).

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

| 4,943 passages | wall clock |
|---|---|
| HippoRAG 2, whole job | **73.9 minutes** |
| this method, edge construction | **~1 second** |

The 73.9 minutes is the **entire HippoRAG job** — vLLM startup, OpenIE indexing, and retrieval
for 500 queries. The logs do not separate indexing from retrieval, so treat it as an upper
bound on indexing, not a measurement of it. Our ~1 second is derived from a directly measured
rate of 0.2 ms per passage (see the full-Wikipedia run below), not from a stopwatch on this
corpus. Both numbers are honest; only one of them is a clean measurement of indexing alone.

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


### The reranking ceiling

Before adding a diversity or lateral-inhibition term we measured what any top-20 reranker
could possibly gain. A **perfect oracle** reordering the top 20 gains at most:

| | already correct | fixable by reordering | not in top 20 at all |
|---|---|---|---|
| HotpotQA | 94.0% | **2.0%** | 4.0% |
| 2WikiMultihopQA | 90.7% | **1.1%** | 8.2% |
| MuSiQue | 34.1% | **4.7%** | **61.2%** |

And the redundancy signal sits in the wrong place: 2Wiki's failing questions really are more
redundant than its passing ones (ratio 1.094), but only 1.1% of questions are fixable there.
MuSiQue has the headroom and essentially no duplicates to remove — 0.02 pairs above 0.85
similarity per question.

**Reranking of the top 20 is exhausted** — but that is a statement about reordering a
*shallow* candidate list, not about reranking in general. Retrieving deeper changes the picture
completely:

| all-supporting@k | top-10 | top-50 | top-100 | top-200 |
|---|---|---|---|---|
| HotpotQA | 0.940 | 0.983 | 0.988 | 0.991 |
| 2WikiMultihopQA | 0.907 | 0.931 | 0.942 | 0.954 |
| **MuSiQue** | **0.341** | 0.534 | **0.647** | 0.732 |

On MuSiQue a reranker operating over the top 100 has **+30.6 points** of headroom — an order of
magnitude more than anything else we measured. The supporting passages were never missing; they
sat at ranks 11–100 and we never looked. The graph advantage also persists at depth
(0.647 vs 0.559 for dense at k=100), so it helps deep candidate generation too, not just the head.

We then applied it — retrieve top-100, rerank with `BAAI/bge-reranker-v2-m3`, read top-10 —
and **it made every dataset worse**: HotpotQA 0.940 → 0.915, MuSiQue 0.341 → 0.311,
2Wiki 0.907 → **0.467**.

The mechanism separates perfectly. Reranking is a *pure trade*:

| | gold was buried at ranks 11–100 | gold was already in the top 10 |
|---|---|---|
| MuSiQue | +96 / **−0** | **+0** / −126 |
| HotpotQA | +38 / **−0** | **+0** / −63 |
| 2Wiki | +5 / **−0** | **+0** / −445 |

R@1 and R@5 both improve. Only *all-supporting@10* falls. A cross-encoder scores direct
query–passage relevance, but the second-hop passage in a multi-hop question **does not contain
the entities the question mentions** — the article about someone's mother does not mention the
film. Bridge passages are, by construction, not relevant-looking. 2Wiki is the purest case and
loses 44 points.

**"Add a reranker" is default RAG advice, and on multi-hop retrieval it is actively harmful.**
Pointwise relevance is the wrong objective for a coverage metric.

One positive result: our graph still contributes *after* reranking, on all three datasets and
with zero losses (MuSiQue 25/4 p=0.0001, HotpotQA 10/0 p=0.0020, 2Wiki 15/0 p=0.0001). The
method survives inside a standard modern pipeline.

---

## It runs on all of Wikipedia

BEIR's HotpotQA corpus — the one used by MTEB — is **5.23M Wikipedia passages**.

| | |
|---|---|
| passages | 5,233,329 |
| mention edges | **63,895,815** (12.21 per node) |
| **index time** | **23.9 minutes**, CPU, zero LLM calls |
| **gold-pair linkage** | **59.0%** |
| the same corpus under HippoRAG-style OpenIE | **~54 days** (linear extrapolation) |

59.0% linkage sits right where the method works — level with our small HotpotQA pool (61.0%)
and far above MuSiQue (32.4%), where it did not. **The structure survives the scale-up.**

Edge density is 13x higher than in the small pools (12.21 versus 0.35–0.92), which predicts
that query gating — the suppression mechanism that only helped on our densest small corpus —
becomes load-bearing here.

This is the setting LLM-based graph RAG cannot enter, and it is why that literature reports
only on pools of a few thousand passages.

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

## Reproducibility: what you can and cannot check

An audit of the published result files against the claims in this README found that several
numbers **cannot be re-derived from the artifacts we published**, because job scripts wrote to
filenames that omitted the seed, the question count, or the dataset, so later runs silently
overwrote earlier ones.

| claim | artifact status |
|---|---|
| retrieval table (n=1000), all three datasets | **verifiable** — matches `graph_*_n1000.json` exactly |
| HippoRAG head-to-head (n=500) | **verifiable** — matches `hipporag_*_n500.json` exactly |
| propositional ablation at n=1000 | file holds n=500 only |
| query gating replicated over 3 seeds | file holds seed 1 only |
| 7B and 3B reader results per dataset | each file holds one dataset |
| full-Wikipedia run (5.23M, 23.9 min, 59.0%) | **no artifact** — job printed but never uploaded |

Those numbers came from job logs and are reported faithfully, but a third party cannot
currently confirm them. Fixing this is the top item in [NEXT.md](NEXT.md).

**A second finding from the same audit.** MuSiQue rows can contain several paragraphs sharing
one title with *different* text — 1,570 occurrences across our 1,000-question sample. Our
scripts disagreed on which one to keep: `job_graph.py` keeps the first, every later script keeps
the last. 279 of 9,897 pooled passages (2.8%) therefore differ between runs. Within any single
run dense and graph share the same corpus, so **every controlled comparison stands**; only
MuSiQue *absolute* values are incomparable across runs. HotpotQA and 2Wiki are unaffected, which
is why their controls reproduced to three decimals and MuSiQue's did not.

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
