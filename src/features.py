"""
features.py -- turning a URL string into numbers a model can learn from.

WEEK 2 EXPANSION
The week 1 set had five crude features and scored 0.66 on honest data. This
expands to ~22, grouped by the phishing behaviour each one targets. Every
feature is computed from the URL STRING ONLY -- nothing here visits the URL.

Two design rules worth understanding:

1. RATIOS, NOT COUNTS, wherever length could confound things. Phishing URLs
   average 75 characters vs 47 for benign, so they contain more of everything.
   "9 digits" mostly means "long URL"; "12% digits" means "unusually numeric".

2. SEPARATE THE HOST FROM THE PATH. An attacker controls their own domain
   completely, but when they abuse compromised hosting they control only the
   path. Those are different signals and mixing them loses information.

SECURITY NOTE: this feature list is also an evasion guide. Anything measured
here is something an attacker who reads your repo can optimise against --
which is exactly what week 6 sets out to measure.

Run from the project root:
    python src/features.py                # uses reference.csv
    python src/features.py dataset.csv    # or the live OpenPhish data
"""

import math
import re
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
RAW = PROJECT_ROOT / "data" / "raw"

IN_NAME = sys.argv[1] if len(sys.argv) > 1 else "reference.csv"
IN_PATH = RAW / IN_NAME
OUT_PATH = RAW / "features.csv"

IP_PATTERN = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")

# Cheap generic TLDs, heavily abused because they cost near nothing and can be
# registered in bulk. A legitimate business is far more likely to pay for .com.
SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "top", "xyz", "buzz", "click", "link",
    "work", "casino", "loan", "download", "review", "country", "stream",
    "zip", "mov", "rest", "cyou", "icu", "sbs",
}

# Link shorteners hide the true destination entirely -- the defining property
# an attacker wants. Legitimate use exists, which is why this is one weak
# signal among many rather than a rule.
SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "cutt.ly", "rebrand.ly", "shorturl.at", "rb.gy", "tiny.cc",
}

# Words that appear when a page is trying to make you act on your credentials.
# Real login pages use these too -- but they use them on their OWN domain,
# which is why this pairs with the brand/subdomain features below.
PHISH_HINT_WORDS = [
    "login", "signin", "verify", "secure", "account", "update", "confirm",
    "banking", "password", "credential", "authenticate", "wallet", "recover",
    "suspended", "unlock", "billing", "invoice", "payment",
]

# Brands impersonated most often. If one of these appears somewhere OTHER than
# the registered domain, that is a strong signal -- paypal.com is PayPal, but
# paypal.secure-login.ru is not.
BRANDS = [
    "paypal", "apple", "microsoft", "google", "amazon", "facebook", "netflix",
    "instagram", "whatsapp", "linkedin", "dropbox", "adobe", "docusign",
    "chase", "wellsfargo", "hsbc", "coinbase", "binance", "metamask", "ledger",
    "steam", "roblox", "tiktok", "office365", "outlook", "icloud",
]


def shannon_entropy(text: str) -> float:
    """Measure how random a string looks, in bits per character.

    "google" is predictable -- few distinct characters, repeated patterns --
    and scores low. "x7fq2mzk9p" uses many different characters roughly
    evenly and scores high. Algorithmically generated attacker domains tend
    to look more random than words a human chose.
    """
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    return -sum((n / length) * math.log2(n / length) for n in counts.values())


def safe_divide(numerator: float, denominator: float) -> float:
    """Divide, returning 0.0 instead of crashing when the bottom is zero."""
    return numerator / denominator if denominator else 0.0


def extract_features(url: str) -> dict:
    """Turn one URL string into a dictionary of numeric features."""

    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path or ""
    query = parsed.query or ""

    # Strip the port if present, so "evil.com:8080" -> "evil.com".
    hostname = host.split(":")[0]
    host_parts = hostname.split(".")

    # The registered domain is roughly the last two labels: for
    # "paypal.com.evil.ru" that's "evil.ru" -- the part the attacker owns.
    # (A proper implementation uses the Public Suffix List to handle cases
    # like .co.uk; this approximation is fine for a lexical baseline.)
    registered_domain = ".".join(host_parts[-2:]) if len(host_parts) >= 2 else hostname
    subdomain = ".".join(host_parts[:-2]) if len(host_parts) > 2 else ""
    tld = host_parts[-1] if host_parts else ""

    url_lower = url.lower()
    letters_in_host = "".join(c for c in hostname if c.isalpha())

    return {
        # -- Size ---------------------------------------------------------
        # Kept, but we now know length alone is weak once the data is clean.
        "url_length": len(url),
        "host_length": len(hostname),
        "path_length": len(path),
        "query_length": len(query),

        # -- Structure ----------------------------------------------------
        # Subdomain depth: "paypal.com.secure.verify.evil.ru" stacks labels to
        # push the real domain out of a phone browser's visible address bar.
        "num_subdomains": max(0, len(host_parts) - 2),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        # Hyphens in the DOMAIN specifically: "paypal-secure-login.com".
        # Real brands rarely hyphenate their primary domain.
        "num_hyphens_host": hostname.count("-"),
        "path_depth": path.count("/"),
        "num_query_params": query.count("=") if query else 0,

        # -- Character composition (ratios, not counts) --------------------
        # The honest version of num_digits: is this string unusually numeric,
        # independent of how long it is.
        "digit_ratio": safe_divide(sum(c.isdigit() for c in url), len(url)),
        # Digits in the hostname are a stronger signal than digits anywhere,
        # since legitimate brand domains almost never contain them.
        "digit_ratio_host": safe_divide(sum(c.isdigit() for c in hostname), len(hostname)),
        # Punctuation density: encoded and obfuscated URLs are punctuation-heavy.
        "special_ratio": safe_divide(
            sum(not c.isalnum() for c in url), len(url)
        ),
        # Percent-encoding can hide characters from casual inspection.
        "num_percent": url.count("%"),

        # -- Known tricks --------------------------------------------------
        # Browsers ignore everything before "@" in the authority, so
        # "http://paypal.com@evil.ru" goes to evil.ru. Zero benign URLs in
        # our dataset use this; 4.3% of phishing do.
        "has_at": int("@" in url),
        # A raw IP means no domain was registered at all.
        "has_ip": int(bool(IP_PATTERN.match(hostname))),
        # A non-standard port on a login page is unusual for real businesses.
        "has_port": int(":" in host and not host.endswith(":80") and not host.endswith(":443")),
        # "//" inside the path is an old open-redirect / confusion trick.
        "double_slash_in_path": int("//" in path),
        # The literal word "https" inside the HOSTNAME -- e.g.
        # "https-paypal-secure.com" -- fakes the reassurance of TLS.
        "https_token_in_host": int("https" in hostname),
        # Is the connection actually TLS? Increasingly weak on its own, since
        # free certificates mean most phishing is HTTPS now.
        "is_https": int(parsed.scheme == "https"),

        # -- Domain intelligence -------------------------------------------
        "suspicious_tld": int(tld in SUSPICIOUS_TLDS),
        "is_shortener": int(registered_domain in SHORTENERS),
        # High entropy suggests an algorithmically generated domain rather
        # than a name a human chose.
        "host_entropy": shannon_entropy(letters_in_host),

        # -- Intent words ---------------------------------------------------
        # Credential-harvesting vocabulary anywhere in the URL.
        "num_phish_hints": sum(word in url_lower for word in PHISH_HINT_WORDS),
        # A brand name in the subdomain or path but NOT in the registered
        # domain is impersonation: "paypal.evil.ru" or "evil.ru/paypal/login".
        # This is the single most targeted signal in the set.
        "brand_outside_domain": int(
            any(
                brand in subdomain or brand in path.lower()
                for brand in BRANDS
            )
            and not any(brand in registered_domain for brand in BRANDS)
        ),
    }


def main() -> None:
    df = pd.read_csv(IN_PATH)
    print(f"Loaded {len(df)} URLs from {IN_PATH.name}")

    feature_rows = df["url"].astype(str).apply(extract_features)
    features_df = pd.DataFrame(feature_rows.tolist())
    features_df["label"] = df["label"]

    features_df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(features_df.columns) - 1} features to {OUT_PATH.name}\n")

    # Averages per class. A feature whose two rows look identical is carrying
    # no signal; a feature with a large gap is worth the model's attention.
    summary = features_df.groupby("label").mean().T
    summary.columns = ["benign", "phishing"]
    summary["gap"] = (summary["phishing"] - summary["benign"]).abs()
    print("Feature averages by class, sorted by separation:")
    print(summary.sort_values("gap", ascending=False).round(3).to_string())


if __name__ == "__main__":
    main()
