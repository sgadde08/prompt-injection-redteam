# Prompt-Injection Red-Team Harness

A small, honest test bench for measuring how well prompt-injection defenses actually
hold at the **injection-to-generation boundary** of a context-aware (RAG) AI
application.

This is the companion code and data for the DZone article
*"Testing Context-Aware AI Defenses: When Your Red-Team Metrics Mislead You"*
(link: _add the published article URL here_).

If you only read one section, read **"The main finding"** below before trusting any
number in this repo.

## What it tests

It isolates a single question: given that a poisoned document has already reached
the model (retrieval is assumed compromised and skipped), does the defense stop the
model from obeying the instruction hidden inside it?

Every payload is run against three defense levels:

| Level | Defense |
| :-- | :-- |
| `none` | retrieved text concatenated raw (baseline) |
| `delimiters` | XML `<retrieved_context>` tags + a system rule: content inside is data, never instructions |
| `delimiters_plus_stripping` | delimiters + a simple regex strip of common override phrases |

against a corpus of 9 injection payloads across 6 categories (naive override,
roleplay framing, delimiter escape, encoding trick, conditional trigger, authority
spoof), N times each, on three models spanning self-hosted, hosted-frontier, and
hosted-small.

## The main finding (read this before trusting any number)

The obvious metric, "did a unique canary string appear in the output," **cannot
measure obedience.** The same canary lands in the output for three completely
different reasons that a substring match cannot tell apart:

1. the model **obeyed** the injection (a real hijack),
2. the model **explained** the injection while summarizing (benign), or
3. the model **refused** and named the string while declining (a defense success).

So the harness's raw `success` score is an **upper bound**, not an obedience rate.
`rescore.py` separates the layers as far as string logic can, but the honest reading
still requires reading the outputs. The one quantitative signal that is unambiguous
is the **provider API refusal rate** (`stop_reason == "refusal"`), which is a
provider-side safety layer distinct from your own prompt defense and absent entirely
when you self-host. The article walks through all of this.

## Safety and intent

`corpus.json` contains **benign canary payloads for defensive testing.** Each one
asks a model to remove the hyphens from a harmless marker (for example
`QX-7F-3A-N1`) and print the result (`QX7F3AN1`). There is no real exploit, malware,
or harmful instruction anywhere in this repo. The payloads exist only to measure
whether a defense causes the model to treat retrieved text as data or as commands.

## Reproducibility note (important)

The hosted-model numbers under `results/` **will not reproduce exactly.** Provider
refusal behavior is non-deterministic and changes as model snapshots and safety
systems are updated, and the self-hosted model has no provider layer at all. Treat
the files in `results/` and `raw_runs/` as the **recorded evidence from the article's
run** (config hashes below), not as values that will come back identical when you run
it. That variability is itself part of the article's point: you cannot treat a single
run of a probabilistic system as a benchmark.

## Repository layout

```
redteam_harness.py     runner, defenses, backends, scorer, aggregator
rescore.py             refusal-aware re-scoring that separates obey / explain / refuse
corpus.json            the 9 benign injection payloads (data)
results/               per-model attack-success tables and rescored tables
raw_runs/              every request's outcome and full output, with a config hash
```

## Running it

Verify the pipeline with the mock backend (no key, no cost; numbers are placeholders,
NOT findings):

```
python redteam_harness.py --backend mock --n 5 --seed 42
```

Anthropic:

```
pip install anthropic
export ANTHROPIC_API_KEY=sk-...
python redteam_harness.py --backend anthropic --model claude-opus-5-5 --n 5 --sleep 0.5
```

OpenAI-compatible endpoint (also covers a local Ollama model, no key, no cost):

```
pip install openai
ollama pull qwen3:14b
python redteam_harness.py --backend openai --base-url http://localhost:11434/v1 \
    --api-key-env NONE --model qwen3:14b --n 5
```

Then separate the defense layers on any raw-run log:

```
python rescore.py --corpus corpus.json --raw raw_runs/raw_runs_opus-5-5.jsonl --out rescored.csv
```

## The recorded runs (provenance)

| Model | Hosting | config_hash |
| :-- | :-- | :-- |
| Claude Opus 5.5 | hosted API | `9a2405b3b527` |
| Claude Haiku 4.5 (snapshot `claude-haiku-4-5-20251001`) | hosted API | `aa03face93a1` |
| Qwen3-14B | self-hosted (Ollama) | `2f7fd17b7ecb` |

Note: temperature was **not** sent to the Anthropic models (the parameter is
deprecated for them), so they ran at their default. The `temperature: 0.7` recorded
in those run configs is a logging artifact. The local Qwen3 run did use `0.7`.

Scope: 9 payloads, N=5, one run of the full matrix per model (405 total runs).
This is an illustrative red-team study to surface failure modes, not an industry
benchmark. Behavioral classifications (obey / explain / refuse) were read by a single
rater; an independent evaluator would make them more trustworthy.

## License

MIT. See `LICENSE`.
