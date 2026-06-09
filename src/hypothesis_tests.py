import pandas as pd
import numpy as np
from scipy import stats


def create_kpis(df):
    df = df.copy()
    df["has_claim"] = (df["TotalClaims"] > 0).astype(int)
    df["margin"] = df["TotalPremium"] - df["TotalClaims"]
    return df


def claim_frequency(df, group_col):
    return df.groupby(group_col)["has_claim"].mean().reset_index()


def claim_severity(df, group_col):
    d = df[df["TotalClaims"] > 0]
    return d.groupby(group_col)["TotalClaims"].mean().reset_index()


def margin_by_group(df, group_col):
    return df.groupby(group_col)["margin"].mean().reset_index()


def chi_square_test(df, group_col):
    table = pd.crosstab(df[group_col], df["has_claim"])
    chi2, p, _, _ = stats.chi2_contingency(table)
    return chi2, p


def t_test_groups(df, group_col, value_col, a, b):
    x = df[df[group_col] == a][value_col].dropna()
    y = df[df[group_col] == b][value_col].dropna()
    t, p = stats.ttest_ind(x, y, equal_var=False)
    return t, p


def z_test_proportions(df, group_col, a, b):
    x = df[df[group_col] == a]["has_claim"]
    y = df[df[group_col] == b]["has_claim"]

    pa, pb = x.mean(), y.mean()
    na, nb = len(x), len(y)

    p_pool = (x.sum() + y.sum()) / (na + nb)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/na + 1/nb))
    z = (pa - pb) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))

    return z, p
