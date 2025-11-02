import argparse
import math
import random
from datetime import datetime, timedelta
import csv
import numpy as np


def beta_to_posterior_alpha_beta(p, n):
    # helper for later notebooks; not used here directly
    alpha = p * n + 1
    beta = (1 - p) * n + 1
    return alpha, beta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_users", type=int, default=100_000)
    ap.add_argument("--baseline_rate", type=float, default=0.10)
    ap.add_argument(
        "--treatment_lift",
        type=float,
        default=0.05,
        help="relative lift, e.g. 0.05 = +5%",
    )
    ap.add_argument("--start", type=str, default="2025-01-01")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="data/simulated/ab_test.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    start = datetime.fromisoformat(args.start)
    variants = ["control", "treatment"]
    countries = ["US", "GB", "DE", "IN", "BR"]
    devices = ["desktop", "mobile"]

    baseline = args.baseline_rate
    trt_rate = baseline * (1 + args.treatment_lift)

    rows = []
    for i in range(args.n_users):
        uid = i + 1
        variant = rng.choice(variants, p=[0.5, 0.5])
        country = rng.choice(countries, p=[0.35, 0.15, 0.15, 0.2, 0.15])
        device = rng.choice(devices, p=[0.6, 0.4])

        day_offset = rng.integers(0, args.days)
        ts = start + timedelta(
            days=int(day_offset),
            hours=int(rng.integers(0, 24)),
            minutes=int(rng.integers(0, 60)),
        )

        # small heterogeneity by country/device
        country_adj = {
            "US": 0.0,
            "GB": -0.01,
            "DE": -0.015,
            "IN": +0.005,
            "BR": +0.003,
        }[country]
        device_adj = {"desktop": 0.0, "mobile": -0.008}[device]

        p = trt_rate if variant == "treatment" else baseline
        p = max(0.0001, min(0.9999, p + country_adj + device_adj))

        converted = rng.random() < p

        # revenue only if converted; lognormal w/ country scale
        rev_scale = {"US": 60, "GB": 55, "DE": 50, "IN": 15, "BR": 20}[country]
        revenue = (
            float(rng.lognormal(mean=math.log(rev_scale), sigma=0.5))
            if converted
            else 0.0
        )

        rows.append(
            [
                uid,
                ts.isoformat(),
                variant,
                country,
                device,
                int(converted),
                round(revenue, 2),
            ]
        )

    # write
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "user_id",
                "timestamp",
                "variant",
                "country",
                "device",
                "converted",
                "revenue",
            ]
        )
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
