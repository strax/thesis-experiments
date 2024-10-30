import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure


HUE_ORDER = ["none", "oracle", "gpc_matern52"]
PALETTE = "Dark2"


def median_metrics(df: pd.DataFrame, *, pfa: bool | None = None):
    if pfa is not None:
        df = df[df["pfa"] == pfa]
    df = df.groupby(["constraint", "feasibility_estimator", "rotoscale"])
    df = df[["gskl", "mmtv", "c2st"]].median()
    df = df.unstack("rotoscale")
    df = df.swaplevel(0, 1, 1).sort_index(axis=1)
    return df


def convergence(df: pd.DataFrame):
    # PFA results are identical to non-PFA for inference, so filter them out
    df = df.query("pfa == False")
    df = df.groupby(["constraint", "feasibility_estimator", "rotoscale"])
    df = df[["success"]].sum()
    df = df.unstack("rotoscale")
    df = df.unstack("feasibility_estimator")
    return df


def median_evaluation_failures(df: pd.DataFrame):
    df = df.groupby(["constraint", "feasibility_estimator", "rotoscale"])
    df = df[["failed_evaluations"]].median()
    df = df.unstack("rotoscale")
    df = df.unstack("feasibility_estimator")
    return df


def median_iterations(df: pd.DataFrame):
    df = df.groupby(["constraint", "feasibility_estimator", "rotoscale"])
    df = df[["iterations"]].median()
    df = df.unstack("rotoscale")
    df = df.unstack("feasibility_estimator")
    return df


def median_runtime(df: pd.DataFrame):
    df = df.groupby(["constraint", "feasibility_estimator"])
    df = df[["inference_runtime"]].median()
    df = df.unstack("feasibility_estimator")
    return df


def plot_metrics(df: pd.DataFrame) -> Figure:
    fig = plt.figure(figsize=(10, 12))
    axes = fig.subplots(3, 2, sharey="row")

    df = df.sort_values("constraint", ascending=False)

    plot_kwargs = dict(
        fill=False,
        gap=0.25,
        hue_order=HUE_ORDER,
        palette=PALETTE,
        fliersize=2,
        linewidth=1,
    )

    for i, pfa in enumerate((False, True)):
        x = df[df["pfa"] == pfa]

        sns.boxplot(
            x,
            x="constraint",
            y="gskl",
            hue="feasibility_estimator",
            ax=axes[0, i],
            legend=i == 0,
            log_scale=True,
            **plot_kwargs
        )
        axes[0, i].set_xlabel("")

        sns.boxplot(
            x,
            x="constraint",
            y="mmtv",
            hue="feasibility_estimator",
            ax=axes[1, i],
            legend=False,
            **plot_kwargs
        )
        axes[1, i].set_ylim((0, 1))
        axes[1, i].set_xlabel("")

        sns.boxplot(
            x,
            x="constraint",
            y="c2st",
            hue="feasibility_estimator",
            ax=axes[2, i],
            legend=False,
            **plot_kwargs
        )
        axes[2, i].set_ylim((0.5, 1.0))
        axes[2, i].set_xlabel("")

    for i, ylabel in enumerate(("GSKL", "MMTV", "C2ST")):
        axes[i, 0].set_ylabel(ylabel)

    axes[0, 0].set_title("Without PFA")
    axes[0, 1].set_title("With PFA")

    return fig
