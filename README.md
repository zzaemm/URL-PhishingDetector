# Phishing URL Detector

Machine learning classifier that predicts whether a URL is phishing or benign
using only lexical features of the URL string — no network requests to the
URLs themselves.

Built as an undergraduate project in AI applied to cyber security.

---

## Status

**Week 1 complete — baseline pipeline working end to end.**

| Model | Precision (phishing) | Recall (phishing) | Accuracy |
|---|---|---|---|
| Logistic regression, 5 lexical features | 0.98 | 0.87 | 0.93 |

These numbers are **not** trustworthy yet. See *Known issues* below.

---

## How it works

A model cannot read a string, so each URL is translated into a small set of
numbers ("features") describing its structure. A classifier then learns which
combinations of those numbers indicate phishing.

Current feature set:

| Feature | What it targets |
|---|---|
| `url_length` | Attackers pad URLs with reassuring words or bury the real domain in a long path |
| `num_dots` | Subdomain abuse, e.g. `paypal.com.verify.evil.ru` |
| `num_digits` | Auto-generated attacker domains and compromised-host paths |
| `has_at` | `http://paypal.com@evil.ru` — browsers ignore everything before the `@` |
| `has_ip` | Raw IP instead of a hostname implies no registered domain |

## Data

| Class | Source | Notes |
|---|---|---|
| Phishing (1) | [OpenPhish](https://openphish.com) community feed | Rotating snapshot of currently-live phishing |
| Benign (0) | [Tranco](https://tranco-list.eu) top-sites ranking | Sampled across 500k ranks, not just the top |

`src/accumulate.py` runs daily and merges each new feed snapshot into a
growing pool, since any single OpenPhish fetch returns only ~300 URLs.

Datasets are gitignored and never committed — the phishing pool contains live
malicious URLs.

## Known issues

**Dataset leakage.** Phishing URLs arrive from a live feed as full URLs with
paths; benign URLs arrive from a domain ranking as bare hostnames. The model
can therefore score well by learning "has a path ⇒ phishing", which is an
artifact of how the two classes were collected rather than anything about
phishing. This is visible in the learned weights, where `url_length` and
`num_dots` dominate. Fixing the collection asymmetry is the next milestone.

**Recall is the weak metric.** At the default 0.5 threshold the model misses
13% of phishing while raising almost no false alarms — the wrong side of the
trade-off for a security tool. Threshold tuning is planned.

## Findings

Two lexical features cited throughout the phishing-detection literature are
close to extinct in a live 2026 feed: only **4 of 300** phishing URLs used the
`@` userinfo trick, and only **1 of 300** used a raw IP address. Current
phishing overwhelmingly abuses reputable hosting instead — OpenPhish reports
~35% of live phishing served from Cloudflare and ~12% from AWS. Features
inherited from older papers should be validated against current data before
being trusted.

## Roadmap

- [x] Week 1 — data collection, 5 features, logistic regression baseline
- [ ] Week 2 — fix collection leakage, expand to ~25 features
- [ ] Week 3 — gradient boosting, threshold tuning, precision/recall curves
- [ ] Week 4 — character-level CNN in PyTorch
- [ ] Week 5 — model comparison and error analysis
- [ ] Week 6 — adversarial evasion testing
- [ ] Week 7 — FastAPI endpoint and CLI
- [ ] Week 8 — writeup

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

python src/accumulate.py      # build/refresh the dataset
python src/features.py        # extract features
python src/train.py           # train and evaluate
```

## Safety note

This project downloads *lists* of phishing URLs as text and never requests
them. Do not add features that fetch the URLs — that sends traffic from your
IP to live attacker infrastructure and exposes you to malware.
