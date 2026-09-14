# Phishing URL Detector

A machine learning classifier that predicts whether a URL is phishing or benign
from the **text of the URL alone** — no network requests to the URLs themselves.

Built as an undergraduate project in AI applied to cyber security. The
interesting part of this repo is not the accuracy number; it is the record of
finding out that the first accuracy number was fake.

---

## Status

**Week 4 of 8 complete.** Three models, a domain-disjoint split, thresholds
chosen on validation under an explicit cost policy, test set evaluated once.

| Model | AUC | Precision | Recall | False alarms | Missed attacks |
|---|---|---|---|---|---|
| Logistic regression (25 features) | 0.866 | 0.726 | 0.859 | 450 | 195 |
| Gradient boosting (25 features) | 0.917 | 0.806 | 0.880 | 293 | 167 |
| **Character-level CNN (raw URL)** | **0.956** | **0.847** | **0.935** | **234** | **90** |

Held-out test set: 2,587 URLs, 53.6% phishing. Every URL from a given registered
domain falls entirely inside one split, so no model has seen any test domain
during training — see [Why the split is domain-disjoint](#why-the-split-is-domain-disjoint).
Reported at `FALSE_ALARM_COST = 0.5`, a deliberate policy choice that weights a
missed attack twice as heavily as a false alarm; see
[Threshold policy](#threshold-policy).

The headline finding: **reading the raw URL beats 25 handcrafted features**, and
the margin widens once domain memorisation is removed from both. Handcrafted
features discard character order — `paypal-secure.evil.com` and
`evil-secure.paypal.com` produce near-identical feature vectors and are entirely
different objects. Convolutions see sequence; counts do not.

![Precision-recall curve](reports/pr_curve.png)

*(Curve currently reflects the earlier row-wise split — regenerating it under the
domain-disjoint split is outstanding.)*

**Caveat stated up front:** the test set is roughly balanced. Real traffic is not
— benign URLs outnumber phishing by orders of magnitude. Under that imbalance
recall would hold but precision would fall sharply, because false alarms scale
with the volume of benign traffic while true catches do not. Gradient boosting
blocks 24% of safe sites at the reported operating point; at 10,000:1 traffic
that is roughly 2,400 false alarms for every phishing URL encountered.

**So this is a first-stage filter, not a standalone blocker.** It is good at
cheaply ranking URLs by suspicion using nothing but the URL string. Something
more expensive — reputation lookup, page content, a human — should make the
final call. The numbers above are honest for what they measure and should not
be read as deployment performance.

---

## Why the split is domain-disjoint

A second collection problem, found in week 4 and affecting every number reported
before it.

URLs are not independent samples. One phishing kit produces hundreds of URLs on
the same host: this dataset has 11,429 URLs across only ~8,100 distinct hosts,
and a single host appears **237 times** — roughly 3% of the corpus is one
attacker's campaign. Split those rows randomly and the same domain lands in
training *and* test. Under the original row-wise split, **51% of test URLs shared
a registered domain with a training URL** and 33% shared an exact hostname.

That is the difference between revising past papers and having seen the exam.
A model can score well by recognising a domain rather than by generalising.

Measured directly, by scoring the week 4 models separately on test URLs whose
domain was or wasn't present in training:

| Subset | CNN AUC | Gradient boosting AUC |
|---|---|---|
| Domain **seen** in training | 0.991 | 0.961 |
| Domain **unseen** | 0.960 | 0.933 |

So `split_three_ways` now groups by registered domain (`GROUP_BY_DOMAIN = True`
in `evaluate.py`), placing every URL from a domain into exactly one split. Every
model loses ground:

| Model | Row-wise AUC | Domain-disjoint AUC |
|---|---|---|
| Logistic regression | 0.893 | 0.866 |
| Gradient boosting | 0.949 | 0.917 |
| Character-level CNN | 0.979 | 0.956 |

The lower numbers are the honest ones. Two costs are worth stating: grouped
splits cannot be stratified, so class balance drifts (the test set is 53.6%
phishing rather than 50%), and precision is base-rate sensitive, so precision is
not strictly comparable across the two splits. AUC is.

Setting `GROUP_BY_DOMAIN = False` reproduces the old behaviour for comparison.

---

## The score arc

The headline result is the shape of this table, not its last row.

| Stage | Accuracy | What it means |
|---|---|---|
| Week 1 baseline, 5 features | 0.92 | **Not real.** A dataset artifact. |
| Week 2, artifact removed | 0.66 | First honest number. |
| Week 2, 25 features | 0.81 | Earned. |
| Week 3, gradient boosting | 0.87 | AUC 0.949 — still domain-leaked. |
| Week 4, domain-disjoint split | 0.82 | AUC 0.917. Leak removed. |
| Week 4, character-level CNN | 0.87 | AUC 0.956. Best honest model. |

Twice the number went *down* and that was the progress: once when a collection
artifact was removed, once when domain leakage was. Both times the lower figure
was the first trustworthy one.

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

Gradient boosting, same test set, read at five different policy settings
(domain-disjoint split; 1,201 benign and 1,386 phishing URLs in test):

| Cost | Threshold | Precision | Recall | False alarms | Missed attacks |
|---|---|---|---|---|---|
| 0.10 | 0.08 | 0.683 | 0.958 | 616 | 58 |
| 0.25 | 0.15 | 0.729 | 0.928 | 479 | 100 |
| **0.50** | **0.31** | **0.806** | **0.880** | **293** | **167** |
| 1.00 | 0.52 | 0.865 | 0.825 | 179 | 242 |
| 2.00 | 0.70 | 0.907 | 0.770 | 110 | 319 |

Read it marginally — what does each step down cost per extra attack caught?

- 1.0 → 0.5 saves 75 attacks, costs 114 false alarms. **~1.5 per attack.**
- 0.5 → 0.25 saves 67 attacks, costs 186 false alarms. **~2.8 per attack.**
- 0.25 → 0.10 saves 42 attacks, costs 137 false alarms. **~3.3 per attack.**

This project reports **0.5**. Worth noting that the curve flattened once domain
leakage was removed — under the old split the marginal price rose sixfold across
this range, here only about twofold, which makes 0.25 more defensible than it
previously looked. The counter-argument is absolute rather than marginal: 0.25
blocks 40% of safe sites against 24% at 0.5, and both are already high.

The intuition that a security tool should simply refuse to miss anything is
worth pricing before acting on it. Cost 0.10 gets closest — 58 missed attacks
out of 1,386 — but blocks **51% of safe sites**. A detector that noisy
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
path_depth              0.0801
num_subdomains          0.0472
path_length             0.0441
special_ratio           0.0365
num_dots                0.0303
num_phish_hints         0.0277
```

The model leans hardest on the shape of the path, and more so under the
domain-disjoint split (0.0638 → 0.0801) than it did before. That is a testable
prediction for week 6: flattening a phishing URL's path should hurt it more than
any other single edit.

The reshuffle underneath is informative too. `num_phish_hints` — a fixed
vocabulary of words like "login" and "secure" — fell from 3rd to 6th once
domain leakage was removed, while the structural `num_subdomains` rose to 2nd.
Vocabulary is campaign-specific and transfers poorly to unseen domains;
structure generalises. Any feature whose importance *drops* when leakage is
removed was partly measuring memorisation.

---

## Data

| Purpose | Source | Size |
|---|---|---|
| Training and evaluation | [`pirocheto/phishing-url`](https://huggingface.co/datasets/pirocheto/phishing-url) (HuggingFace) | 11,429 URLs, both classes one source |
| Live phishing pool | [OpenPhish](https://openphish.com) community feed | 2,095 URLs and growing, collected daily |
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

**5. The gap between models depends on the policy you evaluate under.** Under
the row-wise split at `FALSE_ALARM_COST = 1.0`, gradient boosting beat logistic
regression on precision by about 4 points (0.872 vs 0.835). At 0.5, where both
are pushed toward high recall, the gap widened to 16 points (0.844 vs 0.682) —
logistic regression bought marginally more recall and paid 2.6× as many false
alarms for it. The linear model degrades much faster when asked to catch nearly
everything. A single comparison at one threshold would have understated the
difference fourfold.

**6. Character order carries signal that counting destroys.** The CNN was
expected to lose: 11,429 URLs is a small corpus for a model learning from
scratch, against 25 features designed by hand. It won instead — 0.956 AUC
against 0.917 — and the margin *widened* under the domain-disjoint split (0.030
→ 0.039), which rules out the obvious explanation that it was memorising domain
strings.

The likely reason is that aggregate counts discard sequence.
`paypal-secure.evil.com` and `evil-secure.paypal.com` yield near-identical
feature vectors and are completely different objects. A convolution sees order;
`num_hyphens` cannot. That information is unrecoverable by adding more counting
features, which is the honest limit of the feature-engineering approach here.

**7. A model can be beaten by its own preprocessing.** `cnn.py` truncates URLs
at `MAX_LEN = 200` characters. An attacker who knows that can pad the front of a
URL with innocuous path segments and push the incriminating part past the
cut-off, where the model cannot see it at all. This is a total evasion arising
from an implementation choice rather than from anything about phishing, and it
is cheaper than any mutation on the week 6 list. Gradient boosting does not
share it — its features are computed over the whole string. Scheduled as the
first week 6 test.

---

## Roadmap

- [x] **Week 1** — data collection, 5 features, logistic regression baseline
- [x] **Week 2** — fix collection leakage, expand to 25 features
- [x] **Week 3** — three-way split, gradient boosting, cost-aware threshold policy, PR curves
- [x] **Week 4** — character-level CNN in PyTorch, domain-disjoint split
- [ ] **Week 5** — model comparison, error analysis, learning curves, recall against live phishing
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
| `src/evaluate.py` | **Current evaluator.** Domain-disjoint split, both feature models, cost-policy sweep, permutation importance. Owns the split used by every model. |
| `src/cnn.py` | **Best model.** Character-level CNN in PyTorch. Same split and threshold policy, imported from `evaluate.py`. |
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
