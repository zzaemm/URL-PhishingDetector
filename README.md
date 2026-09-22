# Phishing URL Detector

A machine learning classifier that predicts whether a URL is phishing or benign
from the **text of the URL alone** — no network requests to the URLs themselves.

Built as an undergraduate project in AI applied to cyber security.

**The finding:** a character-level CNN reading the raw URL beats 25
hand-engineered features — because a handwritten list can only contain the
words you already thought of. `num_phish_hints` cannot match
`serviceactivation` or `ingbancoservice` unless someone sat down and added
them. The CNN built its own lexicon from characters, so it isn't limited to
anyone's imagination. Evidence in [Findings](#findings) 6 and 7.

The second thing worth reading here is not an accuracy number at all: it is
the record of two separate occasions where a good-looking score turned out to
be measuring the wrong thing, and what the number became after it was fixed.

---

## Status

**Week 6 of 8 complete.** Four models, a domain-disjoint split, thresholds
chosen on validation under an explicit cost policy, test set evaluated once —
and then attacked on purpose (see
[Adversarial evasion](#adversarial-evasion), where the shipped model is
evaded 97.5% of the time by an attacker who simply looks ordinary).

| Model | AUC | Precision | Recall | False alarms | Missed attacks |
|---|---|---|---|---|---|
| Logistic regression (25 features) | 0.866 | 0.726 | 0.859 | 450 | 195 |
| Gradient boosting (25 features) | 0.917 | 0.806 | 0.880 | 293 | 167 |
| **Character-level CNN (raw URL)** | **0.961** | **0.819** | **0.952** | **291** | **67** |
| CNN + gradient boosting ensemble | 0.964 | 0.887 | 0.908 | 160 | 128 |

All figures are post-truncation-fix (head+tail windowing). Before it, the CNN
scored 0.956 / 0.847 / 0.935. The ensemble edges the CNN on AUC but is **not
shipped** — see [Why the ensemble isn't shipped](#why-the-ensemble-isnt-shipped).

**The shipped model is the CNN, not the ensemble** — see
[Why the ensemble isn't shipped](#why-the-ensemble-isnt-shipped).

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

Every point on a curve is one possible decision threshold. The CNN sits above
both feature models at *every* threshold — a stronger claim than comparing
them at one operating point. Circles mark the reported operating points; the
dotted line is what flagging every URL would score on this test set, the floor
any real model must beat.

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

### The same principle applies to subsampling

`learning_curve.py` shrinks the training set to ask whether collecting more
data would be worth it. There are two ways to shrink it, and they give
opposite answers.

Keep 10% of the **URLs** and you still hold a few URLs from nearly every
domain. Since the test set is domain-disjoint, what is being measured is
generalisation to *unseen* websites — and broad shallow exposure is exactly
what transfers. So that point scores far better than a genuinely 10%-sized
dataset could. Every small point is lifted, the curve flattens, and the
conclusion becomes "more data will not help" when the opposite may be true.

Keep 10% of the **domains** and all their URLs, and the smaller set is
honestly smaller: few sites, known deeply, no secret breadth.

The lumpiness sharpens it — one host carries 237 URLs, so a 10% row sample
still retains ~24 near-identical URLs from that single phishing kit.

Domain-wise sampling also makes the x-axis actionable. This dataset grows by
acquiring new **sites** — that is exactly what `accumulate.py` does daily. So
a point at "4,000 domains" corresponds to a collection effort that could
actually be undertaken. A row-wise x-axis corresponds to nothing anyone could
go and do.

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

## Why the ensemble isn't shipped

Error analysis showed the CNN and gradient boosting fail in opposite
directions — 304 test URLs the CNN rescues, 168 the reverse, with
mechanically different causes (see Findings 8 and 9). That is a textbook case
for ensembling, and the prediction was made before running it.

Nine blending strategies were scored on **validation**; one winner was chosen
there and only that one touched test. Picking the best blend by its test score
would be the same cheat the three-way split exists to prevent, moved one level
up: neither model would have seen test, but the ensemble would have been
fitted to it.

It works — slightly. AUC **0.964** against the CNN's 0.961.

But look at what the operating point actually does:

| | false alarms | missed attacks | weighted cost @ 0.5 |
|---|---|---|---|
| Character CNN | 291 | **67** | 425 |
| Ensemble | **160** | 128 | 416 |

It halves false alarms by **nearly doubling missed attacks**. Under the
project's stated cost policy that nets out to a 2% improvement — and a 2%
improvement is not what "nearly doubling the misses" feels like. This is a
case where the policy constant and the intuition disagree, and it is worth
noticing rather than deferring to the arithmetic.

**Three reasons it is reported rather than shipped:**

*The choice is not robust.* The top two strategies tied at 0.917 on
validation. Concretely: running the identical script on two machines with
different scikit-learn versions selected **different winners** — "mean of
ranks" on one, "mean of probabilities" on the other. A result that flips on a
library patch version is not a finding.

*The gain is near the noise floor.* Validation predicted a larger margin than
test delivered, which is the normal penalty for picking the best of nine
candidates. It is visible at all only because the selection happened on
validation and test was touched once.

*The engineering cost is real.* Shipping the ensemble means two models, torch
*and* scikit-learn at inference, two artefacts to keep in sync — for +0.003
AUC and 61 more phishing URLs through the net. The CNN alone is shipped.

---

## Recall against live phishing

Every result above comes from `pirocheto/phishing-url`, a 2020 research
dataset. Useful for comparing models; silent on whether any of this works on
attacks happening this week.

`src/accumulate.py` has been pulling the OpenPhish live feed daily since
August 2026. `src/live_recall.py` runs the trained models over that pool —
**3,464 phishing URLs, ten collection dates** — using thresholds chosen on
validation, never retuned here.

| Slice | n | Logistic regression | Gradient boosting |
|---|---|---|---|
| **All live phishing** | 3,464 | **88.0%** | **85.7%** |
| domain seen in training | 355 | 89.9% | 91.5% |
| domain unseen *(the honest number)* | 3,109 | 87.8% | 85.0% |

Roughly 85% of currently-live phishing, from a source the models never
trained on, six years after the training data was collected.

### What this is not

**Not a drift study.** Drift requires holding the source constant and varying
time. Here both change: training is pirocheto (2020), the pool is OpenPhish
(2026). A low number could mean phishing has moved on, or simply that
OpenPhish phishing differs from pirocheto phishing. Those cannot be
separated, so the claim stays narrow: *recall against contemporary live
phishing from a different source.*

**Not a model comparison.** Logistic regression scores higher than gradient
boosting here, and that means nothing. Recall alone cannot rank models — a
model flagging everything scores 100%. LR simply sits at a looser threshold
and would raise more false alarms, which this pool cannot measure because it
contains no benign URLs. Comparing *slices within one model* is valid;
comparing models on recall alone is the same mistake as judging by accuracy.

### Length explains more than time does

| URL length | n | LR | GB |
|---|---|---|---|
| ≤ 40 chars | 1,684 | 80.2% | 79.9% |
| 41–80 | 1,472 | 95.0% | 89.7% |
| > 80 | 308 | 97.4% | 98.1% |

Training phishing has a median length of 55 characters; the live pool's is
**41**. Missed URLs have a median of 32, caught ones 42. So a substantial part
of the gap is that live phishing is *shorter* — less string, less lexical
signal — rather than newer. A shorter URL is simply a harder instance of the
same problem.

Across the ten collection dates recall ranges 81–92% with **no trend**. That
comparison does hold source constant, so it is the one clean time signal
available — and a month is far too short a window for real drift, so a flat
line is the expected result.

### A wrong finding, caught

Reading the missed URLs by hand turned up `vercel.app`, `pages.dev`,
`blogspot.com`, `github.io` — which looked like direct confirmation that
week 6's camouflage attack is what live attackers already use. Checked rather
than assumed:

```
recall ON shared platforms : 87.2%  (n=1717)
recall OFF shared platforms: 84.1%  (n=1747)
```

Shared platforms are slightly **easier**, not harder. Half the entire pool is
hosted on them, so half the misses being on them is exactly what chance
predicts. The eyeball sample carried no information — a base-rate error, and
the third time in this project that a plausible reading of a small sample
turned out to be an artifact of how the sample was drawn.

*Outstanding: the CNN is absent from this table — scoring new URLs needs torch,
which Windows Application Control currently blocks on the dev machine.*

---

## Adversarial evasion

Every number above assumes phishing URLs arrive as attackers happen to write
them today. They do not have to. `src/adversarial.py` takes the **1,160
phishing URLs both models correctly caught**, edits each one the way an
adversary would, and measures how much recall survives. Thresholds stay as
validation chose them — an attacker does not get to retune your detector.

Two families of edit:

- **Obfuscation** — hide the incriminating parts. Strip the `phish_hints`
  vocabulary, move off a suspicious TLD, flatten the path, drop it entirely.
  These target the feature list directly.
- **Camouflage** — hide nothing; look ordinary. Rehost on a shared platform
  (`pages.dev`, `weebly.com`) and use a path that reads like a blog article.

| Mutation | CNN recall | GB recall |
|---|---|---|
| *(unmodified)* | 1.000 | 1.000 |
| obfuscate: strip phish words | 0.995 | 0.984 |
| obfuscate: clean TLD | 0.994 | 0.988 |
| obfuscate: flatten path | 0.896 | 0.918 |
| obfuscate: host only | 0.735 | 0.601 |
| camouflage: english path | 0.338 | 0.355 |
| camouflage: shared platform | 0.181 | 0.352 |
| **camouflage: platform + english** | **0.025** | **0.002** |
| truncation: pad w/ english | 0.306 | 0.956 |
| truncation: pad w/ gibberish | 1.000 | 1.000 |

Recall is the fraction of already-caught phishing still caught after the edit.
1.000 means the attack failed; 0.000 is total evasion. Attack surface: 1,188
phishing URLs both models caught unmodified. These are the numbers **after**
the truncation fix described below; the pre-fix figures are in that section.

### Obfuscation barely works

Deleting every word in the `phish_hints` vocabulary leaves recall at 0.99 —
the attack does essentially nothing. Moving off a suspicious TLD does nothing
either. That is a finding about the feature set rather than about attackers:
`num_phish_hints` and `suspicious_tld` are close to decorative, consistent
with `num_phish_hints` having fallen to 6th in permutation importance once
domain leakage was removed.

### Camouflage is near-total, and it is the cheaper attack

Platform hosting plus an English path evades **97.5%** of the time against the
CNN and **99.8%** against gradient boosting. The prediction — made before
running it — was that camouflage would
beat obfuscation, because obfuscation makes a URL *weird* and weird is what
the models were trained to notice. Camouflage makes it ordinary.

An attacker needs no cleverness here. They need a free Weebly account and a
boring page name.

**This cannot be fixed with lexical features, and that is the result.**
Consider:

```
to-our.weebly.com/of-story-tricks-how       <- phishing
my-garden.weebly.com/how-to-grow-tomatoes   <- someone's blog
```

The separating information is not in the string. No architecture, no extra
feature, and no amount of data recovers a signal the input does not carry.
Catching this needs domain age (WHOIS), resolution behaviour (DNS), page
content, or a reputation feed — all of which this project excludes by design.
So the honest conclusion is a boundary: **URL-only detection stops working at
camouflage**, and week 6 measured where that boundary sits.

**Caveat, stated plainly:** the two platform mutations replace host *and*
path, so the result is a fresh URL on shared hosting rather than the original
one edited. The narrower, still-damning claim is that phishing hosted on a
shared platform with an ordinary path carries essentially no lexical signal.
The flatten-path and truncation rows do not have this problem — they preserve
the original URL — which makes truncation the cleanest single result here.

### The truncation attack: found, fixed, verified

The sharpest number in week 6 was self-inflicted. Padding the front of a URL
with ~210 characters of innocuous path dropped **CNN recall from 1.000 to
0.062** while gradient boosting stayed at **0.959**. A 94% evasion caused by
an implementation choice — the CNN's fixed 200-character window — and nothing
to do with phishing. Gradient boosting was unaffected because its features are
computed over the whole string.

Finding 11 predicted this and could not test it: no URL in the dataset has a
benign prefix hiding a payload, because no attacker was targeting this model.
The attack had to be constructed.

**The fix.** `cnn.py` now reads the first 150 characters *and* the last 150,
joined by a separator token, instead of one 200-character window anchored at
the start. Raising the limit to 512 would not have worked — the attacker pads
520 instead. Any single window anchored at one end is defeated by padding the
other. It cost nothing in accuracy: ROC AUC went **0.956 → 0.961**.

**Verifying it was harder than it looks, and this is the methodological part.**
The obvious check — rerun the padding attack — gave 0.306. Better than 0.062,
but far from fixed. The reason is that the padding was built from ordinary
English words, which makes the URL read like a blog: the test was measuring
truncation *and* camouflage at once, and camouflage is the strongest attack in
the table.

There is no neutral filler. Any text either resembles benign content (helping
the attacker) or attacker content (helping the model). So the attack was run
with both, bracketing the answer:

| Padding | CNN recall | bias |
|---|---|---|
| English words | 0.306 | pessimistic — filler is also camouflage |
| Random gibberish | **1.000** | optimistic — filler is also a suspicion signal |

**Gibberish padding is caught 100% of the time. The window hole is closed.**

And the pessimistic bound is explained without appealing to truncation at all:
0.306 for English-padded URLs versus **0.338** for English paths with *no
padding whatsoever*. Statistically the same. The padding contributes nothing
beyond looking ordinary — so what survives is camouflage arriving through a
different door, not a window that is still leaking.

**What the fix does not cover:** an attacker who pads *both* ends pushes the
payload into the middle, out of both windows. More conspicuous and more
expensive, but possible. The fix raises the price; it does not close every
hole. A length-invariant model would be the real answer and is out of scope.

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

**7. The two models fail in mechanically opposite directions.** Gradient
boosting reads *structure*; the CNN reads *vocabulary*. Concretely:

```
                                                            gb     cnn
stackoverflow.com/questions/44481051/relational-db-designing  0.92   0.12
jugendmigrationsdienste.de/                                   0.52   0.01
lender.sandbox.natwest.poweredbydivido.com/                   0.99   0.47
sites.google.com/view/serviceactivation                       0.01   0.52
```

Gradient boosting blocks Stack Overflow because deep path + digits + hyphens
is all it can see. The CNN clears it because `relational-database-designing`
reads as English. But the CNN shrugs at the NatWest URL — a bank name buried
four subdomains deep — which `num_subdomains` catches instantly.

This sharpens finding 6. The CNN's edge is not only character *order*; it is
that `num_phish_hints` is a **closed list written by hand**, so it cannot
contain `serviceactivation` or `ingbancoservice` unless someone thought of
them first. The CNN built its own lexicon from characters. A handwritten
feature can only encode what you already knew.

**8. Complementary errors do not compound.** 472 disagreements looked like
large ensemble headroom. The realised gain was 2%. The reason is that the
disagreements are precisely the URLs where *both* models are least confident
— averaging two uncertain scores yields an uncertain score. Models agree on
the easy cases, and agreement there adds nothing.

**9. The models are most wrong about the sites they exist to protect.** 112
test URLs defeat all three models; 91 are false alarms on legitimate sites,
and 28 of those — a quarter of the entire hard core — are `*.tumblr.com`. A
user subdomain on a shared platform is lexically indistinguishable from
phishing on that same platform, and the domain-disjoint split guarantees the
platform was never seen in training.

The sharpest single case:

```
commbank.com.au/personal/accounts/transaction-accounts.html → flagged by all three
```

The Commonwealth Bank of Australia's real transaction-accounts page. The
models learned that banking vocabulary signals phishing — because phishing
imitates banks — so they penalise the genuine article.

**10. The dataset contains label noise.** `wisegeek.com/what-is-a-form-w-9.htm`
and `poorlydrawnlines.com/comic/fashionable/` are both labelled phishing. One
is a reference site, the other a webcomic. Two obvious errors in a small
sample implies the "hard core" is partly dataset error rather than model
failure — which puts a ceiling on achievable accuracy that no model can cross.
Quantifying that rate is outstanding.

**11. A predicted weakness that the natural data could not detect — and a
constructed attack that confirmed it.** `cnn.py` originally truncated URLs at
200 characters, so an attacker who knew the cut-off could pad the front and
push the payload out of view. The first check looked for this in the existing
data by comparing error rates on long versus short URLs. It found the
opposite:

| model | error rate (≤200 chars) | error rate (>200) |
|---|---|---|
| Logistic regression | 25.4% | 3.4% |
| Gradient boosting | 18.2% | 0.0% |
| Character-level CNN | 12.8% | 0.0% |

Long URLs are *easier* for every model. All 59 in the test set are phishing
with obviously stuffed paths, so the evidence sits well inside the first 200
characters.

The conclusion at the time was "untested rather than disproved": the attack
needs a benign-looking prefix hiding a payload, and no such URL exists in the
data because no attacker was targeting this model. Absence of the attack is
not evidence the attack fails.

Week 6 constructed it, and the vulnerability was real — **94% evasion**. See
[the truncation section](#the-truncation-attack-found-fixed-verified) for the
attack, the fix, and the two-sided verification.

The lesson worth keeping: *a weakness you cannot find in your data may simply
be one nobody has exercised yet.* Looking for evidence of an attack in a
dataset collected before the attack existed will always come up empty.

---

## Roadmap

- [x] **Week 1** — data collection, 5 features, logistic regression baseline
- [x] **Week 2** — fix collection leakage, expand to 25 features
- [x] **Week 3** — three-way split, gradient boosting, cost-aware threshold policy, PR curves
- [x] **Week 4** — character-level CNN in PyTorch, domain-disjoint split
- [x] **Week 5** — error analysis, model disagreement, ensemble, learning curves
- [x] **Week 6** — adversarial evasion testing
- [ ] **Week 7** — FastAPI endpoint, CLI, packaging
- [ ] **Week 8** — write-up

Error analysis changed what week 6 tested. The original plan was obfuscation —
shorten the URL, strip phishing vocabulary, flatten the path. Findings 9 and
10 pointed at a cheaper attack instead: the models cannot distinguish benign
shared hosting from phishing on the same platform, and they clear anything
reading like plain English. Measuring **looking legitimate** rather than
hiding is what produced the 97.5% evasion result.

Outstanding: the CNN's live-phishing recall and its learning curve (both need
torch, currently blocked locally by Windows Application Control), and the
label-noise rate from finding 10.

---

## Repository layout

| File | Purpose |
|---|---|
| `src/accumulate.py` | Daily job. Grows the live OpenPhish pool, caches Tranco monthly. |
| `src/reference_data.py` | Downloads the HuggingFace reference dataset. |
| `src/features.py` | Extracts the 25 lexical features. |
| `src/evaluate.py` | **Current evaluator.** Domain-disjoint split, both feature models, cost-policy sweep, permutation importance. Owns the split used by every model. |
| `src/cnn.py` | **Best model.** Character-level CNN in PyTorch. Same split and threshold policy, imported from `evaluate.py`. |
| `src/ensemble.py` | Blends the CNN with gradient boosting. Nine strategies scored on validation, one chosen, test touched once. |
| `src/error_analysis.py` | Reads the URLs each model gets wrong. Writes a defanged dump to gitignored `data/`. |
| `src/adversarial.py` | **Week 6.** Edits caught phishing the way an attacker would; measures recall collapse. |
| `src/live_recall.py` | Recall against the live OpenPhish pool, with the length/domain/time confounds separated. |
| `src/learning_curve.py` | AUC against training size, subsampled by domain. |
| `colab/week6_adversarial.ipynb` | Runs `cnn.py` and `adversarial.py` on Colab, since Windows Application Control blocks torch locally. |
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
