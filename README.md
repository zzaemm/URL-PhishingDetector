# Phishing URL Detector

A machine learning classifier that predicts whether a URL is phishing or benign
from the **text of the URL alone** — no network requests to the URLs themselves.

Built as an undergraduate project in AI applied to cyber security. The
interesting part of this repo is not the accuracy number; it is the record of
finding out that the first accuracy number was fake.

---

## Status

**Week 3 of 8 complete.** Three-way split, gradient boosting, threshold chosen on
validation under an explicit cost policy, test set evaluated once.

| Model | AUC | Precision | Recall | False alarms | Missed attacks |
|---|---|---|---|---|---|
| Logistic regression (25 features) | 0.893 | 0.682 | 0.927 | 494 | 84 |
| **Gradient boosting** | **0.949** | **0.834** | **0.916** | **208** | **96** |

Held-out test set: 2,286 URLs, 1,143 phishing / 1,143 benign. Thresholds (0.24
and 0.29) were chosen on a separate validation split; the test set was evaluated
exactly once.

These numbers are reported at `FALSE_ALARM_COST = 0.5` — a deliberate policy
choice that weights a missed attack twice as heavily as a false alarm. See
[Threshold policy](#threshold-policy) for why, and for what the alternatives
cost.

![Precision-recall curve](reports/pr_curve.png)

Each point on a curve is one possible decision threshold. Gradient boosting sits
above logistic regression at *every* threshold, which is a stronger claim than
comparing the two models at one arbitrary cut-off. The circles mark the reported
operating points. The dotted line at 0.50 is what a model that flags every URL
would score on this balanced test set — the floor any real model must beat.

**Caveat stated up front:** the test set is balanced 50/50. Real traffic is not —
benign URLs outnumber phishing by orders of magnitude. Under that imbalance
recall would hold but precision would fall sharply, because false alarms scale
with the volume of benign traffic while true catches do not. At the 18% false
alarm rate reported above, traffic at 10,000:1 would produce roughly 1,800 false
alarms for every phishing URL encountered.

**So this is a first-stage filter, not a standalone blocker.** It is good at
cheaply ranking URLs by suspicion using nothing but the URL string. Something
more expensive — reputation lookup, page content, a human — should make the
final call. The numbers above are honest for what they measure and should not
be read as deployment performance.

---

## The score arc

The headline result is the shape of this table, not its last row.

| Stage | Accuracy | What it means |
|---|---|---|
| Week 1 baseline, 5 features | 0.92 | **Not real.** A dataset artifact. |
| Week 2, artifact removed | 0.66 | First honest number. |
| Week 2, 25 features | 0.81 | Earned. |
| Week 3, gradient boosting | 0.87 | AUC 0.949. |

### What went wrong in week 1

Week 1 paired phishing URLs from **OpenPhish** against benign domains from
**Tranco**. That looks reasonable and is quietly fatal. OpenPhish supplies full
URLs with paths, averaging ~52 characters. Tranco supplies bare hostnames,
averaging ~20. The two classes were therefore distinguishable by a property that
has nothing whatsoever to do with phishing.

The model scored 0.92 by learning *"does this URL have a slash after the
domain?"* — an artifact of how the data was collected, not a fact about the
world. The evidence was in the weights: `url_length` carried weight 4.50, far
above every other feature.

### Two attempted fixes, one of which worked

**Widening the benign sample** from Tranco's top 20k to a random draw across
500k ranks, so the benign class wasn't only famous domains. This barely moved
the score (0.93 → 0.92), which proved domain fame was never the leak.

**Replacing the dataset entirely** with
[`pirocheto/phishing-url`](https://huggingface.co/datasets/pirocheto/phishing-url)
— 11,429 URLs where both classes come from one source and both are full URLs
with paths. This worked. Accuracy fell to 0.66 and the `url_length` weight
collapsed from 4.50 to 0.12.

Mean URL length by class moved from 20.4 / 51.7 (a 2.53× gap, almost entirely
artifact) to 47.4 / 74.9 (a 1.58× gap, plausibly real signal).

**The lesson:** a suspiciously good score is a bug report. If two classes were
collected in different ways, the model will find that difference before it finds
anything you care about.

---

## Threshold policy

The model outputs a suspicion score between 0 and 1. Something has to decide
where to cut, and that cut-off is a **security policy, not a model property**.

`evaluate.py` picks the threshold that maximises a weighted F-score on the
validation set. The weight comes from one constant, `FALSE_ALARM_COST`. At its
default of 1.0 it asserts that blocking a safe site and letting a credential-
theft page through are equally bad. That is a claim, not a neutral default —
and it is the same blind spot as reporting accuracy, which also counts mistakes
without asking what each one costs.

Same model, same test set, read at five different policy settings:

| Cost | Threshold | Precision | Recall | False alarms | Missed attacks |
|---|---|---|---|---|---|
| 0.10 | 0.05 | 0.635 | 0.990 | 649 | 12 |
| 0.25 | 0.17 | 0.767 | 0.951 | 331 | 56 |
| **0.50** | **0.29** | **0.834** | **0.916** | **208** | **96** |
| 1.00 | 0.41 | 0.872 | 0.873 | 146 | 145 |
| 2.00 | 0.67 | 0.938 | 0.778 | 59 | 254 |

Read it marginally — what does each step down cost per extra attack caught?

- 1.0 → 0.5 saves 49 attacks, costs 62 false alarms. **~1.3 per attack.**
- 0.5 → 0.25 saves 40 attacks, costs 123 false alarms. **~3 per attack.**
- 0.25 → 0.10 saves 44 attacks, costs 318 false alarms. **~7 per attack.**

The price roughly quintuples across that range while the benefit shrinks, so
the knee sits between 0.5 and 0.25. This project reports **0.5**.

The intuition that a security tool should simply refuse to miss anything is
worth pricing before acting on it. Cost 0.10 nearly delivers it — 12 missed
attacks out of 1,143 — but blocks **57% of safe sites**. A detector that noisy
gets switched off, and a switched-off detector has a real-world recall of zero.
Alert fatigue is a security failure, not a UX complaint.

A production system would not use a single threshold at all. It would use two:
a high score blocks, a middle band shows a soft warning, a low score passes.
That confines the cost of uncertainty to two seconds of a user's attention
rather than a hard block. Deferred to week 7.

---

## Features

25 lexical features, extracted by `src/features.py`. Host and path are treated
separately — the same character means different things in different places (see
Findings).

| Group | Features |
|---|---|
| Length | `url_length`, `host_length`, `path_length`, `query_length` |
| Structure | `num_subdomains`, `num_dots`, `num_hyphens`, `num_hyphens_host`, `path_depth`, `num_query_params` |
| Character composition | `digit_ratio`, `digit_ratio_host`, `special_ratio`, `num_percent`, `host_entropy` |
| Classic deception tricks | `has_at`, `has_ip`, `has_port`, `double_slash_in_path`, `https_token_in_host` |
| Reputation proxies | `is_https`, `suspicious_tld`, `is_shortener` |
| Content signals | `num_phish_hints`, `brand_outside_domain` |

Which of these the model actually relies on, measured by permutation importance
(shuffle one feature into noise, see how much performance drops):

```
path_depth              0.0638
path_length             0.0439
num_phish_hints         0.0402
num_subdomains          0.0378
num_dots                0.0330
```

The model leans hardest on the shape of the path. That is a testable prediction
for week 6: flattening a phishing URL's path should hurt it more than any other
single edit.

---

## Data

| Purpose | Source | Size |
|---|---|---|
| Training and evaluation | [`pirocheto/phishing-url`](https://huggingface.co/datasets/pirocheto/phishing-url) (HuggingFace) | 11,429 URLs, both classes one source |
| Live phishing pool | [OpenPhish](https://openphish.com) community feed | 900 URLs and growing, collected daily |
| Benign domain reference | [Tranco](https://tranco-list.eu) | sampled across 500k ranks |

`src/accumulate.py` runs as a daily scheduled job, merging each new OpenPhish
snapshot into a growing pool — any single fetch returns only ~300 URLs. This
pool is reserved for week 5's evaluation against *currently live* phishing.

**Datasets are gitignored and never committed.** The phishing pool contains live
malicious URLs; committing it would get the repo flagged by GitHub's malware
scanning.

---

## Findings

Things that came out of looking at the data rather than reading the score.

**1. Two textbook features are nearly extinct.** In a live 2026 OpenPhish feed,
only 4 of 300 phishing URLs used the `@` userinfo trick and only 1 of 300 used a
raw IP address. Both are canonical features throughout the phishing-detection
literature. Modern phishing abuses reputable hosting instead — OpenPhish reports
~35% of live phishing served from Cloudflare and ~12% from AWS, and the pool is
full of `*.pages.dev` and `backblazeb2.com` subdomains.

But rare is not the same as uninformative: in the larger 2020 reference set,
`has_at` appears in 4.3% of phishing URLs and in *exactly zero* benign ones. The
feature was unlearnable at n=300, not useless. Features inherited from older
papers should be validated against current data before being trusted — and
before being discarded.

**2. Hyphens flip meaning depending on where they are.** `num_hyphens_host` has
weight +0.66 (pushes toward phishing) while `num_hyphens` across the whole URL
has weight −1.23 (pushes toward benign). Hyphens in a hostname look like
`paypal-secure-login.com`. Hyphens anywhere look like a blog slug:
`/how-to-make-homemade-insecticidal-soap/`. This is the concrete argument for
separating host features from path features rather than counting over the whole
string.

**3. Correlated features produce misleading weights.** `has_ip` is roughly 50×
enriched in phishing, yet it acquired a small *negative* linear weight — because
an IP address is entirely digits, so `digit_ratio_host` already captures it. A
worked example of why linear coefficients cannot be read as feature importance.

**4. Accuracy hid a complete behavioural reversal.** Between two week-2 runs,
accuracy moved 0.93 → 0.92 while the model flipped from 1 false alarm and 8
missed attacks to 8 false alarms and 2 missed attacks. Nearly the same accuracy,
opposite security postures. This is the project's strongest argument against
reporting accuracy for a security classifier.

**5. The gap between the two models depends on the policy you evaluate under.**
At `FALSE_ALARM_COST = 1.0` gradient boosting beats logistic regression on
precision by about 4 points (0.872 vs 0.835). At 0.5, where both models are
pushed toward high recall, the gap widens to 15 points (0.834 vs 0.682) — to
reach comparable recall, logistic regression needs 494 false alarms where
gradient boosting needs 208, more than twice as many. The linear model degrades
much faster when asked to catch nearly everything. A single comparison at one
threshold would have understated the difference by a factor of three.

---

## Roadmap

- [x] **Week 1** — data collection, 5 features, logistic regression baseline
- [x] **Week 2** — fix collection leakage, expand to 25 features
- [x] **Week 3** — three-way split, gradient boosting, cost-aware threshold policy, PR curves
- [ ] **Week 4** — character-level CNN in PyTorch
- [ ] **Week 5** — model comparison, error analysis, recall against live phishing
- [ ] **Week 6** — adversarial evasion testing
- [ ] **Week 7** — FastAPI endpoint, CLI, packaging
- [ ] **Week 8** — write-up

Week 6 is the intended headline. The feature list above is, read the other way,
an evasion guide: it says exactly what to change about a URL to slip past this
model. The plan is to apply those mutations to phishing URLs the model currently
catches and measure how far recall collapses.

---

## Repository layout

| File | Purpose |
|---|---|
| `src/accumulate.py` | Daily job. Grows the live OpenPhish pool, caches Tranco monthly. |
| `src/reference_data.py` | Downloads the HuggingFace reference dataset. |
| `src/features.py` | Extracts the 25 lexical features. |
| `src/evaluate.py` | **Current evaluator.** Three-way split, both models, cost-policy sweep, permutation importance. |
| `src/plot_curves.py` | Precision-recall and ROC curves → `reports/`. |
| `src/train.py` | Week 1–2 baseline. Superseded, kept as the reference point the story is told against. |
| `src/get_data.py` | Week 1 OpenPhish+Tranco builder. Superseded — this is the code that produced the artifact. |

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

python src/reference_data.py    # download the training dataset
python src/features.py          # extract features
python src/evaluate.py          # train and evaluate
python src/plot_curves.py       # write PR and ROC curves to reports/
```

`python src/accumulate.py` refreshes the live phishing pool; it is not needed to
reproduce the results above.

---

## Safety note

This project downloads *lists* of phishing URLs as text and never requests them.
Do not add any feature that fetches a URL — that sends traffic from your IP to
live attacker infrastructure, confirms to the operator that the URL is being
examined, and risks serving you malware. Every feature here is computed from the
URL string offline, which is a deliberate design constraint, not a limitation
waiting to be fixed.
