import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def get_numeric_columns(df):
    return df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()


def get_categorical_columns(df):
    return df.select_dtypes(include=["object", "category"]).columns


def missing_values_summary(df):
    missing_table = pd.DataFrame({
        "Missing Count": df.isna().sum(),
        "Missing %": (df.isna().mean() * 100)
    })

    missing_table = missing_table.sort_values("Missing %", ascending=False)
    missing_table


def drop_missing_cols(df):
    pct = (df.isna().mean()) * 100
    missing_cols = pct[pct > 50].index
    return df.drop(columns=missing_cols)


def plot_histograms(df, numeric_cols):
    df[numeric_cols].hist(
        figsize=(16, 12),
        xrot=45,
        bins=30
    )

    plt.tight_layout()
    plt.show()


def plot_categorical_cols(df, categorical_cols):
    for col in categorical_cols:
        plt.figure(figsize=(12, 5))
        df[col].value_counts().head(15).plot(kind="bar")
        plt.title(col)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def bivariate_scatter_plot(df, col1, col2):
    plt.figure(figsize=(12, 5))
    df.plot.scatter(x=col1, y=col2)
    plt.title(f"{col1} vs {col2}")
    plt.tight_layout()
    plt.show()


def multivariate_scatter_plot(df, col1, col2, col3):
    plt.figure(figsize=(12, 5))
    sns.scatterplot(
        data=df,
        x=col1,
        y=col2,
        hue=col3
    )

    plt.title(f"{col1} vs {col2} by {col3}")
    plt.tight_layout()
    plt.show()


def correlation_matrix(df, cols):
    numerical_cols = df[cols]
    corr_mat = numerical_cols.corr()
    return corr_mat


def correlation_heatmap(corr_mat):
    plt.figure(figsize=(12, 5))
    sns.heatmap(corr_mat, cmap="coolwarm", center=0)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()


def comparison(df, col1, col2):
    table = pd.crosstab(df[col1], df[col2])
    return table


def comparison_plot(table, x, y):
    plt.figure(figsize=(12, 5))
    table.plot(kind="bar", stacked=True)
    plt.title(f"{x} accross {y}")
    plt.tight_layout()
    plt.show()


def avg_comparison(df, col1, col2):
    comp = df.groupby(col1)[col2].mean().sort_values(ascending=False)
    return comp


def box_plots(df, cols):
    for col in cols:
        plt.figure(figsize=(12, 5))
        sns.boxplot(x=df[col])
        plt.title(f"Box plot of {col}")
        plt.tight_layout()
        plt.show()


def overall_loss_ratio(df):
    return (
        df["TotalClaims"].sum()
        / df["TotalPremium"].sum()
    )


def province_loss_ratio(df):
    plr = df.groupby("Province").agg({
        "TotalClaims": "sum",
        "TotalPremium": "sum"
    })

    plr["LossRatio"] = (
        plr["TotalClaims"]
        / plr["TotalPremium"]
    )

    return plr.sort_values(
        "LossRatio",
        ascending=False
    )


def vehicle_loss_ratio(df):
    vlr = df.groupby("VehicleType").agg({
        "TotalClaims": "sum",
        "TotalPremium": "sum"
    })

    vlr["LossRatio"] = (
        vlr["TotalClaims"]
        / vlr["TotalPremium"]
    )

    return vlr.sort_values(
        "LossRatio",
        ascending=False
    )


def gender_loss_ratio(df):
    glr = df.groupby("Gender").agg({
        "TotalClaims": "sum",
        "TotalPremium": "sum"
    })

    glr["LossRatio"] = (
        glr["TotalClaims"]
        / glr["TotalPremium"]
    )

    return glr
