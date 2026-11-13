"""pump.fun top-5 forecaster.

Reads /input/history.ndjson + /input/task.json, writes /output/forecast.json.
Deterministic, stdlib only, no network.

Strategy: pump.fun coins are born constantly and shoot to the top within
minutes, and caps move a long way in hours. A persistence forecast loses.
We spread probability: each place names the current leaders plus fast
risers, but leaves a large share unnamed for newcomers (the practice
showed 2 of the top 5 at the last read were born after the history ended).
Sizes use a wide log-normal around each place's recent cap plus growth.
"""
import json, math, os

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

def clip(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def main():
    with open(os.path.join(INPUT_DIR, "task.json")) as f:
        task = json.load(f)
    places = task["places"]
    edges = task["size"]["edges"]  # 81 edges, 2.00..6.00 step 0.05
    lines = open(os.path.join(INPUT_DIR, "history.ndjson")).read().strip().split("\n")
    marks = [json.loads(ln) for ln in lines if ln.strip()]
    n = len(marks)
    # Per-coin track: mint -> list of (mark_idx, rank, log10cap_SOL, venue)
    coins = {}
    for idx, mk in enumerate(marks):
        for row in mk.get("rows") or []:
            mint = row["mint"]
            # .get with a 0 default: a row missing its cap degrades to a
            # zero-cap entry instead of crashing the whole run.
            mc = int(row.get("market_cap_lamports", 0)) / 1e9
            lg = math.log10(max(mc, 1.0))
            c = coins.setdefault(mint, {"hist": [], "name": row.get("name",""), "symbol": row.get("symbol","")})
            c["hist"].append((idx, row["rank"], lg, row.get("venue","")))
    # Score each coin: recent log-cap (dominant), growth rate, recency of birth.
    scores = {}
    last_idx = n - 1
    for mint, c in coins.items():
        hist = c["hist"]
        # Recent log-cap: weighted toward the latest mark. Weights grow by
        # 1.5x per step so the newest snapshot dominates while older marks
        # still smooth out one-off spikes; only the last 4 marks are used.
        wsum = 0.0; lsum = 0.0
        for k, (idx, rank, lg, venue) in enumerate(hist[-4:]):
            w = 1.5 ** k
            wsum += w; lsum += w * lg
        if wsum == 0:
            continue
        avg_lg = lsum / wsum
        # growth: log-cap change per mark over the coin's history (bounded)
        if len(hist) >= 2:
            g = (hist[-1][2] - hist[0][2]) / max(len(hist)-1, 1)
        else:
            g = 0.0
        g = clip(g, -0.3, 0.6)
        # last rank
        last_rank = hist[-1][1]
        # venue: pool coins are graduated (more stable) than curve coins
        venue_bonus = 0.15 if hist[-1][3] == "pool" else 0.0
        score = avg_lg + 1.5*g + venue_bonus - 0.02*last_rank
        scores[mint] = score
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    pool = [m for m, _ in ranked[:40]]
    # Last mark's top coins (for place anchors)
    last_rows = sorted(marks[-1].get("rows") or [], key=lambda r: r["rank"])
    last_top = [r["mint"] for r in last_rows[:10]]
    # --- probabilities: for each place, spread over candidates ---
    probs = []
    for place in range(1, places+1):
        # Candidate weight: blend of overall score and proximity to this place's rank.
        # We use the coin's recent rank: expected place ~ its rank in the last mark.
        weights = []
        for mint in pool:
            c = coins[mint]
            # expected rank: last rank if present else score-implied
            lr = None
            for (idx, rank, lg, venue) in reversed(c["hist"]):
                if idx == last_idx:
                    lr = rank; break
            if lr is None:
                # not in last mark: infer from score rank among pool
                lr = 15 + pool.index(mint)
            # kernel around this place
            spread = 2.0 + 0.5*place
            d = abs(lr - place)
            wgt = math.exp(-(d*d)/(2*spread*spread))
            # boost by score (bigger cap = likelier to hold a top place)
            wgt *= math.exp(0.5*(scores[mint] - scores[pool[0]]))
            weights.append((mint, wgt))
        weights.sort(key=lambda x: -x[1])
        chosen = weights[:20]
        s = sum(w for _, w in chosen)
        if s <= 0:
            continue
        # Name 0.55 of the place (leave 0.45 unnamed for newcomers/movers)
        named_total = 0.55
        for mint, w in chosen:
            p = named_total * (w / s)
            if p > 0.18: p = 0.18
            if p < 0.005: continue
            probs.append({"mint": mint, "place": place, "p": round(p, 4)})
    # Renormalize each place so named sum <= 0.55 (after capping)
    from collections import defaultdict
    sums = defaultdict(float)
    for r in probs:
        sums[r["place"]] += r["p"]
    # (capping above may reduce sums; that's fine, leaves more unnamed)
    # --- top5: the 5 best-scored coins ---
    top5 = [m for m, _ in ranked[:5]]
    # --- sizes: log-normal per place around recent cap + growth, wide std ---
    # For each place, center = log10 cap of the coin at that place in last mark,
    # plus expected growth over the gap (use median per-mark growth of top coins, capped).
    # std 0.45 (wide, caps move a lot).
    sizes = []
    # median growth per mark among top-10 coins
    growths = []
    for mint in last_top:
        c = coins.get(mint)
        if c and len(c["hist"]) >= 2:
            growths.append((c["hist"][-1][2] - c["hist"][0][2]) / max(len(c["hist"])-1, 1))
    growths.sort()
    med_g = growths[len(growths)//2] if growths else 0.1
    med_g = clip(med_g, 0.0, 0.35)
    # Gap in marks between history end and answer reads: use task reads if present.
    # For practice the gap is ~1-7 marks; for final ~42-78 marks. We widen std with gap.
    # Estimate gap: use the number of marks in history as a proxy (final has 66, practice 6).
    # Use a generous std that grows with the (unknown) gap: 0.45 base.
    std = 0.5
    for place in range(1, places+1):
        # center: log-cap of last-mark coin at this place, plus growth * a few marks
        center = None
        if place <= len(last_rows):
            mc = int(last_rows[place-1]["market_cap_lamports"]) / 1e9
            center = math.log10(max(mc, 1.0)) + med_g * 6
        if center is None:
            center = 3.0
        # bins: 0 = <10^2.00, 1..80 = [edges[i-1], edges[i]), 81 = >=10^6.00
        # edges list has 81 values 2.00..6.00; bin i (1..80) covers [edges[i-1], edges[i])
        # bin 81 covers >= 6.00
        bin_probs = []
        # bin 0: P(X < 2.00)
        p0 = norm_cdf((2.00 - center)/std)
        bin_probs.append((0, p0))
        for i in range(1, 81):
            lo = edges[i-1]; hi = edges[i]
            p = norm_cdf((hi-center)/std) - norm_cdf((lo-center)/std)
            bin_probs.append((i, p))
        # bin 81: P(X >= 6.00)
        p81 = 1.0 - norm_cdf((6.00-center)/std)
        bin_probs.append((81, p81))
        # Name the top bins covering ~0.80 of mass; leave rest unnamed.
        bin_probs.sort(key=lambda x: -x[1])
        acc = 0.0
        for b, p in bin_probs:
            if acc >= 0.80 or p < 0.004:
                continue
            sizes.append({"place": place, "bin": b, "p": round(p, 4)})
            acc += p
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, task.get("output", "forecast.json"))
    with open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps({"probabilities": probs, "top5": top5, "sizes": sizes}) + "\n")
    print("wrote", out, "probs:", len(probs), "sizes:", len(sizes), "top5:", top5)

if __name__ == "__main__":
    main()
