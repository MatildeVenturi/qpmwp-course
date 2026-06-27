# Assignment 5 Submission Notes

## Idea In Brief

The project studies weekly mean reversion in Swiss equities. The core idea is that stocks with unusually weak recent one-week performance may rebound in the following week after temporary price pressure unwinds. A walk-forward Ridge model combines volatility-scaled reversal with residual reversal, volatility, liquidity, value, quality, and momentum features to rank stocks out of sample.

The final backtest evaluates two implementations of the same frozen signal:

- **Continuous Ridge-EWMA:** the best empirical implementation, using smoothed long-only weights from the Ridge ranking.
- **Ridge Black-Litterman:** the formal optimized implementation, using the Ridge basket as a Black-Litterman view with benchmark-aware long-only constraints.

## How To Navigate The Files

1. Start with `assignment_5_mean_reverting_idea.html` for idea generation, signal diagnostics, Ridge calibration, turnover control, and frozen parameter choices.
2. Read `assignment_5_mean_reverting_backtest.html` for the final out-of-sample test from 2020 onward, including costs, SPI comparison, subperiod results, and the final conclusion.
3. Use the matching `.ipynb` files if code inspection or reproducibility is required. All custom functions and classes for the assignment are defined inside the notebooks; the course `src/` files are not modified.

## Included Deliverables

- `assignment_5_mean_reverting_idea.ipynb`
- `assignment_5_mean_reverting_idea.html`
- `assignment_5_mean_reverting_backtest.ipynb`
- `assignment_5_mean_reverting_backtest.html`
- `assignment_5_submission_notes.md`
