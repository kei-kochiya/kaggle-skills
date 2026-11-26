# Rohlik Orders Forecasting Challenge 🏆

> **Track:** Time Series — Retail Multi-Warehouse Demand Forecasting (Community Prediction) · **Scale:** 1,017 teams · \$12,000 · Closed Aug 24, 2024  
> **Core Lesson:** In direct multi-month retail demand forecasting (`61-day` horizon) scored on **MAPE** across heterogeneous warehouses undergoing structural growth, **fixing incomplete holiday calendars**, **multi-window 61-day seasonal/rolling CV** (defending against a noisy 123-row random Public LB), **trimming stale pre-growth history**, **`log`-transform + MAE loss stability** (or horizon-aligned GBDTs), and **holiday proximity/closure/text-embedding features** beat raw target lags and Public-LB overfitting.

---

## 1. Competition DNA & Shakeup Anatomy

- **Task:** Forecast daily grocery delivery `orders` across **7 European warehouses** (Czechia, Hungary, Germany) over a **61-day future horizon** (`2024-03-16` to `2024-05-15`, `397` test rows) from `train.csv` (`7,340` rows, `2020-12-05` to `2024-03-15`) + `train_calendar.csv` / `test_calendar.csv`.
- **Train-Only vs. Test-Available Columns:** Only `warehouse, date, holiday_name, holiday, shops_closed, winter_school_holidays, school_holidays` exist in test; operational/weather columns (`shutdown, mini_shutdown, blackout, mov_change, frankfurt_shutdown, precipitation, snow, user_activity_1/2`) exist **only in train**.
- **Metric:** **Mean Absolute Percentage Error (MAPE)**. Public LB (`31%` = **123 rows**) vs Private LB (`69%` = **274 rows**) were split **randomly across the 61-day test window** rather than sequentially — creating high Public-LB variance.

| Private # | Shake | Team | Private MAPE | Public MAPE (Rank) | Core Breakthroughs |
| :---: | :---: | :--- | :---: | :---: | :--- |
| 🥇 **1st** | **+6** | **Anak Lakarsantri ID** | **0.0360** | 0.0304 (#7) | *(No writeup posted)* |
| 🥈 **2nd** | `-1` | **techniquetwice** | **0.0366** | 0.0292 sel. (`0.0258` #1 unsel.) | **Single LightGBM**: `log` target/features + **MAE (`L1`) loss** + missing holidays + `1.3×` holiday weight + 3-stage seasonal `Mar–May` CV + PL |
| 🥉 **3rd** | **+1** | **DerTeam** | **0.0371** | 0.0295 (#4) | Custom 2-split stable CV (`Mar–May 23`, `Feb–Mar 24`) + added **Easter Sunday** & holiday lags + `VotingRegressor` (`2× LGBM + 2× HistGB`) |
| 🏅 **4th** | `-1` | **yitiaoxiaoyuer** | **0.0374** | 0.0289 (#3) | *(No writeup posted)* |
| 🏅 **5th** | `—` | **ChickenBoy** | **0.0396** *(0.0381 unsel.)* | 0.0297 (#5) | Trimmed history `< 2022-03-01` + 305 holiday/closure/trend features (`delta_days`, 4-day window) + 2-level `XGB+CB+LGB → LGB` stack |
| 🏅 **6th** | **+6** | **Vincent Schuler** | **0.0397** *(0.0362 unsel.)* | 0.0327 (#12) | Stable multi-horizon CV + holiday calendar refinement |
| 🏅 **8th** | **+13** | **vladkha** | **0.0398** | 0.0335 (#21) | **5× 61-day rolling CV** + single LGBM + fixed Easter + **`GPT-4o` + `stella_en_1.5B` holiday-description embeddings (`0.0404→0.0398` Priv)** |
| 🏅 **11th** | **+3** | **Ern711** | **0.0414** *(0.0405 unsel.)* | 0.0332 (#14) | **Lags-as-extra-rows** + multi-offset CatBoost (`t, t±1d, t+2d` + ratios) + `train+test` calendar holiday distances |

---

## 2. EDA & Data Forensics

1. **Missing Dates = Warehouse Closures:** `173` missing dates across the series represent **closed warehouses** (public holidays + **German Sundays**, whereas Prague operates Sundays). Computing holiday/closure distances strictly over the merged `train_calendar + test_calendar` (not just open rows in `train.csv`!) was critical (`#5`, `#11`).
2. **Missing Holidays in Official Calendar:** Key holidays (notably **Easter Sunday**) were absent from `train_calendar` / `test_calendar`. Manually auditing and patching country-specific holidays (`#2`, `#3`, `#8`) yielded immediate CV & Private gains.
3. **Structural Regime Shifts & Non-Stationary Growth:** Younger warehouses opened mid-2021 / early-2022; **`Munich_1`** jumped `~1,000 → ~6,000` orders after the Oct 2023 Bringmeister acquisition. Because tree leaves cannot extrapolate unbounded trends, trimming stale pre-growth history (`>= 2022-03-01`, `#5`) and filtering folds with `< 100` samples/warehouse (`#3`) prevented older warehouse regimes from dominating splits.
4. **Simple Benchmark Sanity Check (`#8`):** Averaging the **previous 61 days by day-of-week** alone placed **#281 on Private LB** — a crucial zero-leakage anchor for validating complex models.

---

## 3. Validation Strategy & MAPE-Aligned Modeling

- **Horizon-Matched Multi-Window Temporal CV:**
  - `#8` validated across **5 non-overlapping 61-day rolling windows** (`May 2023 – Mar 2024`), surging **+13 ranks** on Private LB.
  - `#2` used a **3-stage seasonal `TimeSeriesSplit`** matching the spring `Mar–May` test window (`Mar–May 22`, `Mar–May 23`) to align holiday patterns (Easter/Pentecost/May Day).
  - `#3` paired a year-prior seasonal window (`Mar–May 2023`) with the most recent month (`Feb–Mar 2024`) to capture both spring seasonality and the post-acquisition Munich regime shift.
- **`log(y)` + MAE Loss vs. Native MAPE Tradeoff:**
  - `#2` (`0.0366` Priv, single LGBM) found **`log` target/features + `MAE` (`L1`) loss** stabilized optimization and prevented shakeup (noting `MAPE` gave slightly better CV while `MAE` generalized far better on Private!).
  - Conversely, `#3` (`0.0371` Priv) found `log` + `MAE` degraded their pipeline relative to native MAPE on `LGBM + HistGB` — proving the choice interacts with feature scaling and must be validated strictly on multi-window horizon CV.
- **High-Leverage Feature Engineering:**
  - **Holiday Proximity, Closures & Weighting:** `days_until_next_holiday`, `days_since_last_holiday` (on full calendar!), continuous closed-day run lengths before/after, holiday-type lags (`#3`, `#5`, `#11`), **`1.3×` holiday sample weight** (`#2`), and **LLM-generated holiday description embeddings (`GPT-4o` + `stella_en_1.5B`, `0.0404 → 0.0398` Priv in `#8`)**.
  - **Trend & Multi-Horizon Augmentation:** `delta_days_from_start`, 4-day stacked calendar windows (`#5`), and **lags-as-extra-rows** predicting `t, t±1, t+2` offsets + ratios (`#11`).

---

## 4. Verified Anti-Patterns & Shakeup Traps

1. **Overfitting the 123-Row Random Public LB:** `#2`'s **#1 Public (`0.0258`)** submission collapsed to **`0.0445` Private** (selecting their `0.0292` Public model won #2 at **`0.0366` Private**!); Public #2 (`0.0286`) crashed **-29 ranks** to `#31` (`0.0444`).
2. **Naive Target Lags on Direct 61-Day Horizons (`#3`, `#5`, `#8`):** Raw short-term target lags consistently hurt or failed across all three writeups on a 61-day direct horizon without test-time actuals.
3. **Separate Per-Warehouse Models & `Prophet`/`user_activity_2` Proxying (`#3`, `#8`):** Splitting out outlier warehouses (`Munich`) or predicting train-only `user_activity_2` / Prophet trends degraded Public & Private LB vs a single global GBDT sharing cross-warehouse calendar dynamics.
4. **Full-Data Target Encoding Leakage (`#5`) & Cyclical `sin/cos` / `is_weekend` Redundancy (`#5`, `#8`).**

---

## 5. Verified Solution Sources

- **2nd Place (`techniquetwice`):** `https://www.kaggle.com/competitions/rohlik-orders-forecasting-challenge/writeups/techniquetwice-2nd-place-solution-single-lgbm-mode`
- **3rd Place (`DerTeam`):** `https://www.kaggle.com/competitions/rohlik-orders-forecasting-challenge/writeups/derteam-3rd-place-solution-custom-stable-cv-a-miss`
- **5th Place (`ChickenBoy`):** `https://www.kaggle.com/competitions/rohlik-orders-forecasting-challenge/writeups/chickenboy-5th-place-solution`
- **6th Place (`Vincent Schuler`):** `https://www.kaggle.com/competitions/rohlik-orders-forecasting-challenge/writeups/vincent-schuler-6th-place-solution`
- **8th Place (`vladkha`):** `https://www.kaggle.com/competitions/rohlik-orders-forecasting-challenge/writeups/vladkha-8th-place-solution-potential-5th-place-fun`
