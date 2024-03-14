from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import pandas
from scipy.stats import bootstrap, kendalltau, spearmanr
import seaborn as sns


def calculate_statistics(df):
    stats_dict = {"train": {}, "val": {}, "test": {}}

    for i, sp in enumerate(["train", "val", "test"]):
        df_tmp = df.loc[df["split"] == sp, ["target", "pred"]]

        # Calculate MAE and bootstrapped confidence interval
        mae = (df_tmp["target"] - df_tmp["pred"]).abs().mean()
        conf_interval = bootstrap(
            (df_tmp["target"], df_tmp["pred"]),
            statistic=lambda target, pred: np.abs(target - pred).mean(),
            method="basic",
            confidence_level=0.95,
            paired=True,
        ).confidence_interval
        stats_dict[sp]["mae"] = {
            "value": mae,
            "95ci_low": conf_interval.low,
            "95ci_high": conf_interval.high,
        }

        # Calculate RMSE and bootstrapped confidence interval
        rmse = np.sqrt((df_tmp["target"] - df_tmp["pred"]).pow(2).mean())
        conf_interval = bootstrap(
            (df_tmp["target"], df_tmp["pred"]),
            statistic=lambda target, pred: np.sqrt(np.power(target - pred, 2).mean()),
            method="basic",
            confidence_level=0.95,
            paired=True,
        ).confidence_interval
        stats_dict[sp]["rmse"] = {
            "value": rmse,
            "95ci_low": conf_interval.low,
            "95ci_high": conf_interval.high,
        }

        # Calculate Spearman r and bootstrapped confidence interval
        sp_r = spearmanr(df_tmp["target"], df_tmp["pred"]).statistic
        conf_interval = bootstrap(
            (df_tmp["target"], df_tmp["pred"]),
            statistic=lambda target, pred: spearmanr(target, pred).statistic,
            method="basic",
            confidence_level=0.95,
            paired=True,
        ).confidence_interval
        stats_dict[sp]["sp_r"] = {
            "value": sp_r,
            "95ci_low": conf_interval.low,
            "95ci_high": conf_interval.high,
        }

        # Calculate Kendall's tau and bootstrapped confidence interval
        tau = kendalltau(df_tmp["target"], df_tmp["pred"]).statistic
        conf_interval = bootstrap(
            (df_tmp["target"], df_tmp["pred"]),
            statistic=lambda target, pred: kendalltau(target, pred).statistic,
            method="basic",
            confidence_level=0.95,
            paired=True,
        ).confidence_interval
        stats_dict[sp]["tau"] = {
            "value": tau,
            "95ci_low": conf_interval.low,
            "95ci_high": conf_interval.high,
        }

    return stats_dict


def plot_losses(df, fn=None, splits=["train", "val", "test"]):
    # Subset to only include desired splits
    idx = df["split"].isin(splits)
    df = df.loc[idx, :]

    fig, ax = plt.subplots()
    sns.lineplot(df, x="epoch", y="loss", hue="label", alpha=0.7, ax=ax)

    ax.set_title("Validation Loss Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (Squared pIC$_{50}$)")
    ax.legend(title="Model")

    if fn:
        fig.savefig(fn, dpi=200, bbox_inches="tight")


def plot_model_preds_scatter(loss_df, lab, fn=None, stats_dict={}):
    # Set so the legend looks nicer
    legend_text_mapper = {-1: "Below Range", 0: "In Range", 1: "Above Range"}
    loss_df["Assay Range"] = list(map(legend_text_mapper.get, loss_df["in_range"]))

    in_range_idx = loss_df["in_range"] == 0
    df_in_range = loss_df.loc[in_range_idx, :]
    fg = sns.relplot(
        data=loss_df,
        x="target",
        y="pred",
        col="split",
        col_wrap=2,
        col_order=["train", "na", "val", "test"],
        style="Assay Range",
        markers={"Below Range": "<", "In Range": "o", "Above Range": ">"},
        style_order=["Below Range", "In Range", "Above Range"],
        facet_kws={"sharex": False, "sharey": False},
    )

    # Figure title
    fg.figure.subplots_adjust(top=0.9)
    fg.figure.suptitle(lab)

    # Axes bounds
    min_val = loss_df.loc[:, ["target", "pred"]].values.flatten().min() - 0.5
    max_val = loss_df.loc[:, ["target", "pred"]].values.flatten().max() + 0.5

    # Only set y label for left side plots
    fg.axes[0].set_ylabel("Predicted $\mathrm{pIC}_{50}$")
    fg.axes[2].set_ylabel("Predicted $\mathrm{pIC}_{50}$")

    # Only set x label for bottom plots
    fg.axes[2].set_xlabel("Experimental $\mathrm{pIC}_{50}$")
    fg.axes[3].set_xlabel("Experimental $\mathrm{pIC}_{50}$")

    for sp in ["train", "val", "test"]:
        # Get the right axes
        ax = fg.axes_dict[sp]

        # Set title
        ax.set_title(sp.title())

        # Make it a square
        ax.set_aspect("equal", "box")
        ax.set_xlim((min_val, max_val))
        ax.set_ylim((min_val, max_val))

        # Plot y=x line
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            color="black",
            ls="--",
        )

        # Shade 0.5 pIC50 and 1 pIC50 regions
        ax.fill_between(
            [min_val, max_val],
            [min_val - 0.5, max_val - 0.5],
            [min_val + 0.5, max_val + 0.5],
            color="gray",
            alpha=0.2,
        )
        ax.fill_between(
            [min_val, max_val],
            [min_val - 1, max_val - 1],
            [min_val + 1, max_val + 1],
            color="gray",
            alpha=0.2,
        )

    # Clear stats table axes
    ax = fg.axes_dict["na"]
    ax.clear()

    if stats_dict is not None:
        # Do stats-related stuff
        if not stats_dict:
            stats_dict = calculate_statistics(df_in_range)

        # Plot bar chart
        # Reform dict into DF
        stats_rows = [
            [sp, stat, stat_d["value"], stat_d["95ci_low"], stat_d["95ci_high"]]
            for sp, sp_d in stats_dict.items()
            for stat, stat_d in sp_d.items()
        ]
        stats_df = pandas.DataFrame(
            stats_rows,
            columns=["split", "statistic", "value", "95ci_low", "95ci_high"],
        )
        sns.barplot(
            stats_df,
            x="statistic",
            y="value",
            order=["mae", "rmse", "sp_r", "tau"],
            hue="split",
            hue_order=["train", "val", "test"],
            errorbar=None,
            ax=ax,
        )
        for patch, (sp, statistic) in zip(
            ax.patches,
            product(["train", "val", "test"], ["mae", "rmse", "sp_r", "tau"]),
        ):
            x = patch.get_x() + 0.5 * patch.get_width()
            y = patch.get_height()
            ax.errorbar(
                x=x,
                y=y,
                yerr=[
                    [
                        stats_dict[sp][statistic]["value"]
                        - stats_dict[sp][statistic]["95ci_low"]
                    ],
                    [
                        stats_dict[sp][statistic]["95ci_high"]
                        - stats_dict[sp][statistic]["value"]
                    ],
                ],
                color="black",
            )

        # Fix labels
        ax.set_title("Statistics")
        ax.set_xticklabels(["MAE", "RMSE", "Spearman's $\\rho$", "Kendall's $\\tau$"])
        ax.set_xlabel("")
        ax.set_ylabel("Statistic Value")
        ax.legend(labels=["Train", "Val", "Test"], title="Split")
    else:
        # Just turn off stats
        ax.axis("off")

    if fn:
        plt.savefig(fn, dpi=200, bbox_inches="tight")


def plot_stats_summary(stats_df, fn=None):
    # Reform DF into dict
    stats_dict = {
        lab: (
            g.groupby("split")[["statistic", "value", "95ci_low", "95ci_high"]]
            .apply(lambda x: x.set_index("statistic").to_dict(orient="index"))
            .to_dict()
        )
        for lab, g in stats_df.groupby("label")
    }

    # Make the initial plots
    fg = sns.catplot(
        stats_df,
        x="statistic",
        y="value",
        order=["mae", "rmse", "sp_r", "tau"],
        hue="label",
        hue_order=sorted(stats_df["label"].unique()),
        col="split",
        col_order=["train", "val", "test"],
        kind="bar",
        errorbar=None,
    )

    # Adjust the axes
    for split, ax in zip(["train", "val", "test"], fg.axes[0]):
        # Plot the error bars
        for patch, (lab, statistic) in zip(
            ax.patches,
            product(sorted(stats_df["label"].unique()), ["mae", "rmse", "sp_r", "tau"]),
        ):
            x = patch.get_x() + 0.5 * patch.get_width()
            y = patch.get_height()
            ax.errorbar(
                x=x,
                y=y,
                yerr=[
                    [
                        stats_dict[lab][split][statistic]["value"]
                        - stats_dict[lab][split][statistic]["95ci_low"]
                    ],
                    [
                        stats_dict[lab][split][statistic]["95ci_high"]
                        - stats_dict[lab][split][statistic]["value"]
                    ],
                ],
                color="black",
            )

        # Fix labels
        ax.set_xlabel("Statistic")
        ax.set_xticklabels(["MAE", "RMSE", "Spearman's $\\rho$", "Kendall's $\\tau$"])

    # Only the first ax has a ylabel
    fg.axes[0, 0].set_ylabel("Statistic Value")

    # Adjust legend title
    fg.legend.set_title("Model")

    if fn:
        plt.savefig(fn, dpi=200, bbox_inches="tight")


def reform_loss_dict(loss_dict, pred_epoch=-1):
    """
    Reform a loss_dict into a DataFrame with the following columns:
    * "compound_id": Compound id
    * "split": Which split this compound was in
    * "target": Target (experimental) value of this compound
    * "in_range": Whether this measurement was below (-1) within (0) or above (1) the
                  dynamic range of the assay
    * "pred": Model prediction at the epoch given by `pred_epoch`

    Parameters
    ----------
    loss_dict : dict
        Dict organized as {split: {compound_id: {exp_dict}}}, where exp_dict contains
        at minimum values for "target", "in_range", and "preds"
    pred_epoch : int, default=-1
        Which epoch to take the prediction value at (defaults to last)

    Returns
    -------
    pandas.DataFrame
        Reformed DataFrame, as described above
    """
    all_split = []
    all_compound_id = []
    all_target = []
    all_range = []
    all_pred = []
    for sp, sp_dict in loss_dict.items():
        for compound_id, cpd_dict in sp_dict.items():
            all_split.append(sp)
            all_compound_id.append(compound_id)
            all_target.append(cpd_dict["target"])
            all_range.append(cpd_dict["in_range"])
            all_pred.append(cpd_dict["preds"][pred_epoch])

    df = pandas.DataFrame(
        {
            "compound_id": all_compound_id,
            "split": all_split,
            "target": all_target,
            "in_range": all_range,
            "pred": all_pred,
        }
    )
    return df


def reform_full_loss_dict(loss_dict):
    """
    Reform a loss_dict into a DataFrame with the following columns:
    * "compound_id": Compound id
    * "split": Which split this compound was in
    * "target": Target (experimental) value of this compound
    * "in_range": Whether this measurement was below (-1) within (0) or above (1) the
                  dynamic range of the assay
    * "pred": Model prediction at the epoch given by `pred_epoch`

    Parameters
    ----------
    loss_dict : dict
        Dict organized as {split: {compound_id: {exp_dict}}}, where exp_dict contains
        at minimum values for "target", "in_range", and "preds"
    pred_epoch : int, default=-1
        Which epoch to take the prediction value at (defaults to last)

    Returns
    -------
    pandas.DataFrame
        Reformed DataFrame, as described above
    """
    all_split = []
    all_compound_id = []
    all_target = []
    all_range = []
    all_epoch = []
    all_pred = []
    all_loss = []
    for sp, sp_dict in loss_dict.items():
        for compound_id, cpd_dict in sp_dict.items():
            n_epochs = len(cpd_dict["preds"])
            all_split.extend([sp] * n_epochs)
            all_compound_id.extend([compound_id] * n_epochs)
            all_target.extend([cpd_dict["target"]] * n_epochs)
            all_range.extend([cpd_dict["in_range"]] * n_epochs)
            all_pred.extend(cpd_dict["preds"])
            all_epoch.extend(np.arange(n_epochs) + 1)
            all_loss.extend(cpd_dict["losses"])

    df = pandas.DataFrame(
        {
            "compound_id": all_compound_id,
            "split": all_split,
            "target": all_target,
            "in_range": all_range,
            "pred": all_pred,
            "epoch": all_epoch,
            "loss": all_loss,
        }
    )
    return df
