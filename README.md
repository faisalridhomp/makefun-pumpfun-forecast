# pumpfun-top5-forecast-01

Entry for the make.fun bounty "Predict the top 5 new Solana coins by market cap, 24 hours out".

## Program

`main.py` (Python 3.13, standard library only) reads `/input/history.ndjson` (10-minute market-cap snapshots of pump.fun coins) and `/input/task.json`, and writes `/output/forecast.json`: probabilities for each coin at each of places 1-5, a top-5 guess, and size-bin probabilities per place.

## Method

Pump.fun coins are born constantly and shoot to the top within minutes, and caps move a long way in hours, so persistence loses. Each coin is scored by recent log market cap (dominant), bounded growth, and pool graduation. Each place names the leaders plus fast risers but leaves about 45 percent unnamed for newcomers — the practice showed 2 of the top 5 at the last read were born after the history ended. Sizes use a wide log-normal (std 0.5) around each place's recent cap plus growth.

## Files

- `main.py` — the forecaster
- `LICENSE` — MIT

Verified: deterministic, runs twice with identical output.
