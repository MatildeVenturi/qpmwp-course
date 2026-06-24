############################################################################
### QPMwP - BLACK-LITTERMAN HELPER FUNCTIONS
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     20.04.2026
# First version:    13.05.2025
# --------------------------------------------------------------------------




# Load standard libraries
from typing import Optional, Union

# Load third-party libraries
import numpy as np
import pandas as pd






def bl_posterior_mu_sigma(
    mu_prior: pd.Series,
    covmat: pd.DataFrame, 
    P: Union[np.ndarray, pd.DataFrame],
    q: Union[np.ndarray, pd.Series],
    Psi: Union[np.ndarray, pd.DataFrame],
    Omega: Union[np.ndarray, pd.DataFrame],
) -> Union[pd.Series, pd.DataFrame]:
    """
    Computes the posterior mean vector and posterior covariance matrix under the
    Black–Litterman model using the Bayesian update.

    Parameters
    ----------
    mu_prior : pd.Series
        Prior expected returns.
    covmat : pd.DataFrame
        Prior covariance matrix \\( \Sigma_{\text{prior}} \\).
    P : array-like
        Pick matrix encoding the views.
    q : array-like
        Expected returns for the views.
    Psi : array-like
        Prior uncertainty matrix (often \\( \tau \Sigma \\)).
    Omega : array-like
        View uncertainty matrix.

    Returns
    -------
    pd.Series
        Posterior expected returns \\( \mu_{\text{post}} \\).
    pd.DataFrame
        Posterior covariance matrix \\( \Sigma_{\text{post}} \\).
    """

    # Ensure all matrices have the same ordering
    # before converting them to numpy arrays
    ids = mu_prior.index
    if isinstance(mu_prior, pd.Series):
        mu_prior = mu_prior.to_numpy()
    if isinstance(P, pd.DataFrame):
        P = P[ids].to_numpy()
    if isinstance(q, pd.Series):
        q = q.to_numpy()
    if isinstance(Psi, pd.DataFrame):
        Psi = Psi.loc[ids, ids].to_numpy()
    if isinstance(Omega, pd.DataFrame):
        Omega = Omega.to_numpy()

    # Compute the posterior mean and covariance
    Psi_inv = np.linalg.inv(Psi)
    Omega_inv = np.linalg.inv(Omega)
    V_inv = P.T @ Omega_inv @ P + Psi_inv
    V_inv_inv = np.linalg.inv(V_inv)  # //Beware: this reflects uncertainty in the mean estimates, not the variability of returns.

    mu_posterior = V_inv_inv @ (
        P.T @ Omega_inv @ q + Psi_inv @ mu_prior
    )
    sigma_posterior = covmat + pd.DataFrame(V_inv_inv, index=ids, columns=ids)

    # # Alternative formula (computationally more stable, according to Meucci (2010))
    # mu_posterior = mu_prior + (tau * Sigma @ P.T) @ np.linalg.inv(tau * P @ Sigma @ P.T + Omega) @ (q - P @ mu_prior)
    # sigma_posterior = (1 - tau) * Sigma - tau**2 * Sigma @ P.T np.linalg.inv(tau * P @ Sigma @ P.T + Omega) @ (P @ Sigma)

    # Convert to pandas
    mu_posterior = pd.Series(mu_posterior, index=ids)
    # sigma_posterior = pd.DataFrame(sigma_posterior, index=ids, columns=ids)

    return mu_posterior, sigma_posterior




def view_from_scores_quintile_sort(
    scores: pd.Series,
    mu_ref: pd.Series,
    scalefactor: float = 1.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Generate views based on quintile thresholds of scores.

    Parameters:
    -----------
    scores : pd.Series
        The scores used to determine the quintiles.
    mu_ref : pd.Series
        The reference mean vector.
    scalefactor : float, optional
        A scaling factor for the expected returns (default is 1.0).

    Returns:
    --------
    P : pd.DataFrame
        The pick matrix representing the views with equal weights within each quintile.
    q : pd.Series
        The expected returns for the views, scaled by scalefactor.
    """

    # Input validation
    if len(scores) == 0 or len(mu_ref) == 0:
        raise ValueError("Input series cannot be empty")

    # Ensure indices match
    common_index = scores.index.intersection(mu_ref.index)
    if len(common_index) == 0:
        raise ValueError("No common assets between scores and mu_ref")

    scores_aligned = scores[common_index]
    mu_ref_aligned = mu_ref[common_index]

    # Take the negative of scores so that first quintile corresponds to highest scores (best assets)
    scores_aligned = -scores_aligned

    # Define quintile parameters
    n_quintiles = 5
    quintile_percentiles = np.linspace(0, 100, n_quintiles + 1)
    score_thresholds = np.percentile(scores_aligned.dropna(), quintile_percentiles)

    # Create pick matrix P with equal weights within each quintile
    quintile_portfolios = {}

    for q_idx in range(1, len(score_thresholds)):
        quintile_name = f'Q{q_idx}'

        # Determine assets in current quintile
        if q_idx == 1:
            # First quintile: scores <= threshold
            mask = scores_aligned <= score_thresholds[q_idx]
        else:
            # Other quintiles: previous_threshold < scores <= current_threshold  
            mask = (scores_aligned > score_thresholds[q_idx-1]) & (scores_aligned <= score_thresholds[q_idx])

        assets_in_quintile = scores_aligned[mask].index

        # Create equal-weighted portfolio for this quintile
        if len(assets_in_quintile) > 0:
            portfolio_weights = pd.Series(0.0, index=common_index)
            portfolio_weights[assets_in_quintile] = 1.0 / len(assets_in_quintile)
            quintile_portfolios[quintile_name] = portfolio_weights

    # Construct pick matrix
    if not quintile_portfolios:
        raise ValueError("No valid quintile portfolios could be created")

    P = pd.DataFrame(quintile_portfolios).T.fillna(0.0)

    # Compute expected returns for each quintile using corresponding mu_ref quintiles
    mu_ref_thresholds = np.percentile(mu_ref_aligned.dropna(), quintile_percentiles)
    q = pd.Series(index=P.index, dtype=float)

    for q_idx in range(1, len(mu_ref_thresholds)):
        quintile_name = f'Q{q_idx}'

        if quintile_name in q.index:
            # Determine mu_ref values in current quintile
            if q_idx == 1:
                mask = mu_ref_aligned <= mu_ref_thresholds[q_idx]
            else:
                mask = (mu_ref_aligned > mu_ref_thresholds[q_idx-1]) & (mu_ref_aligned <= mu_ref_thresholds[q_idx])

            quintile_mu_values = mu_ref_aligned[mask]
            q[quintile_name] = quintile_mu_values.mean() if len(quintile_mu_values) > 0 else 0.0

    # Apply scaling factor
    q = q * scalefactor

    return P, q


def view_from_scores_longshort_sort(
    scores: pd.Series,
    mu_ref: pd.Series,
    scalefactor: float = 1,
    tail_fraction: float = 0.20,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Generate view on a long-short portfolio based on extreme score tails.

    Parameters:
    -----------
    scores : pd.Series
        The scores used to determine long and short positions.
    mu_ref : pd.Series
        The reference mean vector.
    scalefactor : float, optional
        A scaling factor for the expected returns (default is 1).
    tail_fraction : float, optional
        Fraction in each extreme score tail (default is the original 20%).

    Returns:
    --------
    P : pd.DataFrame
        The pick matrix representing the views.
    q : pd.Series
        The expected returns for the views.
    """
    if not 0 < tail_fraction < 0.5:
        raise ValueError("tail_fraction must be strictly between 0 and 0.5.")

    # Compute configurable extreme-tail thresholds
    tail_percent = 100 * tail_fraction
    lower_threshold, upper_threshold = np.percentile(
        scores.dropna(), [tail_percent, 100 - tail_percent]
    )

    # Identify long and short positions
    s_short = scores[scores <= lower_threshold]
    s_long = scores[scores >= upper_threshold]

    # Create long-short weights
    w_ls = pd.Series(0.0, index=scores.index)
    if not s_short.empty:
        w_ls[s_short.index] = -1 / len(s_short)
    if not s_long.empty:
        w_ls[s_long.index] = 1 / len(s_long)

    # Create the pick matrix (P)
    P = w_ls.to_frame().T.reset_index(drop=True)

    # Compute view portfolio expected return (q) by a long-short
    # portfolio of the best versus worst implied returns
    mu_low, mu_high = np.percentile(
        mu_ref.dropna(), [tail_percent, 100 - tail_percent]
    )
    mu_short = mu_ref[mu_ref <= mu_low]
    mu_long = mu_ref[mu_ref >= mu_high]
    q = pd.Series([mu_long.mean() - mu_short.mean()]) * scalefactor

    return P, q


def view_from_scores_one_sided_sort(
    scores: pd.Series,
    mu_ref: pd.Series,
    scalefactor: float = 1,
    tail_fraction: float = 0.20,
    alpha_high: float = 0.0,
    alpha_low: float = 0.0,
    include_high_view: bool = True,
    include_low_view: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Generate prior-relative one-sided views from extreme score tails.

    `alpha_high` and `alpha_low` are expected-return adjustments relative to
    each basket's prior return and must have the same frequency as `mu_ref`.
    A negative low-score adjustment expresses an unfavourable view; final
    holdings remain determined by the optimizer and its constraints.
    """
    if not 0 < tail_fraction < 0.5:
        raise ValueError("tail_fraction must be strictly between 0 and 0.5.")
    if not include_high_view and not include_low_view:
        raise ValueError("At least one one-sided Black-Litterman view must be enabled.")

    tail_percent = 100 * tail_fraction
    lower_threshold, upper_threshold = np.percentile(
        scores.dropna(), [tail_percent, 100 - tail_percent]
    )

    view_rows = []
    view_returns = []

    if include_high_view:
        high_ids = scores[scores >= upper_threshold].index
        high_weights = pd.Series(0.0, index=scores.index)
        high_weights.loc[high_ids] = 1 / len(high_ids)
        view_rows.append(high_weights)
        view_returns.append(float(high_weights @ mu_ref) + alpha_high)

    if include_low_view:
        low_ids = scores[scores <= lower_threshold].index
        low_weights = pd.Series(0.0, index=scores.index)
        low_weights.loc[low_ids] = 1 / len(low_ids)
        view_rows.append(low_weights)
        view_returns.append(float(low_weights @ mu_ref) + alpha_low)

    P = pd.DataFrame(view_rows).reset_index(drop=True)
    q = pd.Series(view_returns) * scalefactor

    return P, q


def view_from_scores_complete_sort(
    scores: pd.Series,
    mu_ref: pd.Series,
    scalefactor: float = 1,
) -> (pd.DataFrame, pd.Series):
    """
    Generate views based on full ranking of scores.

    Parameters:
    -----------
    scores: pd.Series
        The scores used to determine the ranking.
    mu_ref: pd.Series
        The reference mean vector.
    scalefactor: float, optional
        A scaling factor for the expected returns (default is 1).

    Returns:
    --------
    P: pd.DataFrame
        The pick matrix representing the views.
    q: pd.Series
        The expected returns for the views.
    """

    # Drop NaN values from scores
    scores_clean = scores.dropna()

    # Create the pick matrix
    P = pd.DataFrame(
        np.zeros((len(scores_clean), len(scores))),
        index=scores_clean.index,
        columns=scores.index
    )
    # Set values to 1 for the scores that are not NaN
    for idx in scores_clean.index:
        P.loc[idx, idx] = 1

    if len(scores_clean) == len(mu_ref):

        # Rank the scores in descending order
        scores_rank = scores_clean.rank(ascending=False).astype(int)

        # Align the implied returns with the rank of the scores
        sorted_mu = mu_ref.sort_values(ascending=False)
        q = pd.Series(
            sorted_mu.iloc[scores_rank-1].values,  # Align ranks with sorted returns
            index=mu_ref.index
        ) * scalefactor

    else:
        # Compute the average mu_ref for each quantile
        thresholds = np.quantile(mu_ref, np.linspace(0, 1, len(scores_clean)))
        mu = pd.Series(index=scores_clean.sort_values().index, dtype=float)
        for i in range(len(thresholds)):
            if i == 0:
                idx = mu_ref <= thresholds[i+1]
            elif i == len(thresholds) - 1:
                idx = mu_ref >= thresholds[i-1]
            else:
                idx = (mu_ref >= thresholds[i-1]) & (mu_ref <= thresholds[i+1])
            mu.iloc[i] = mu_ref[idx].mean()

        q = mu[scores_clean.index] * scalefactor

    return P, q


def generate_views_from_scores(
    scores: pd.Series,
    mu_ref: pd.Series,
    method: str = 'quintile_sort',
    scalefactor: float = 1,
    tail_fraction: float = 0.20,
    alpha_high: float = 0.0,
    alpha_low: float = 0.0,
    include_high_view: bool = True,
    include_low_view: bool = True,
) -> (pd.DataFrame, pd.Series):
    """
    Generate views based on scores using the specified method.

    Parameters:
    -----------
    scores: pd.Series
        The scores used to generate views.
    mu_ref : pd.Series
        The reference mean vector.
    method: str, optional
        The method to generate views, including relative-tail
        `longshort_sort` and prior-relative `one_sided_sort` alternatives.
        Default is `quintile_sort`.
    scalefactor: float, optional
        A scaling factor for the expected returns (default is 1).
    tail_fraction: float, optional
        Tail fraction used by tail-based methods (default is 20%).
    alpha_high, alpha_low: float, optional
        Prior-relative adjustments used only by `one_sided_sort`.
    include_high_view, include_low_view: bool, optional
        Select which one-sided views are generated by `one_sided_sort`.

    Returns:
    --------
    P: pd.DataFrame
        The pick matrix representing the views.
    q: pd.Series
        The expected returns for the views.
    """
    if method == 'longshort_sort':
        return view_from_scores_longshort_sort(
            scores, mu_ref, scalefactor, tail_fraction=tail_fraction
        )
    elif method == 'one_sided_sort':
        return view_from_scores_one_sided_sort(
            scores=scores,
            mu_ref=mu_ref,
            scalefactor=scalefactor,
            tail_fraction=tail_fraction,
            alpha_high=alpha_high,
            alpha_low=alpha_low,
            include_high_view=include_high_view,
            include_low_view=include_low_view,
        )
    elif method == 'quintile_sort':
        return view_from_scores_quintile_sort(scores, mu_ref, scalefactor)
    elif method == 'complete_sort':
        return view_from_scores_complete_sort(scores, mu_ref, scalefactor)
    else:
        raise ValueError(
            "Invalid method. Use 'longshort_sort', 'one_sided_sort', "
            "'quintile_sort' or 'complete_sort'."
        )
