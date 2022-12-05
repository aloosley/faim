import matplotlib as mpl
import matplotlib.pyplot as plt
from faim.util import util
import numpy as np


def plotKDEPerGroup(data, groups, score_attr, filename):

    mpl.rcParams.update(
        {
            "font.size": 24,
            "lines.linewidth": 3,
            "lines.markersize": 15,
            "font.family": "Times New Roman",
        }
    )
    # avoid type 3 (i.e. bitmap) fonts in figures
    mpl.rcParams["ps.useafm"] = True
    mpl.rcParams["pdf.use14corefonts"] = True
    mpl.rcParams["text.usetex"] = True

    scoresPerGroup = util.scoresByGroup(data, groups, score_attr)
    scoresPerGroup = scoresPerGroup.rename(groups, axis="columns")
    scoresPerGroup.plot.kde()
    score_attr = score_attr.replace("_", "\_")

    plt.xlabel(score_attr)
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.savefig(filename, dpi=100, bbox_inches="tight")


def plotScoreKDEsPerGroup(data, groups, scoreNames, boundary, filename, groupNames):

    mpl.rcParams.update(
        {
            "font.size": 24,
            "lines.linewidth": 3,
            "lines.markersize": 15,
            "font.family": "Times New Roman",
        }
    )
    # avoid type 3 (i.e. bitmap) fonts in figures
    mpl.rcParams["ps.useafm"] = True
    mpl.rcParams["pdf.use14corefonts"] = True
    mpl.rcParams["text.usetex"] = True

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cmap = plt.get_cmap("tab10")

    for attr in scoreNames:
        scoresPerGroup = util.scoresByGroup(data, groups, attr)
        scoresPerGroup = scoresPerGroup.rename(groupNames, axis="columns")
        if "true" in attr:
            scoresPerGroup.columns += "\_true"
            for i, col in enumerate(scoresPerGroup.columns):
                scoresPerGroup[col].plot(kind="kde", ax=ax, linestyle="-", color=cmap(i))
        elif "pred" in attr:
            scoresPerGroup.columns += "\_pred"
            for i, col in enumerate(scoresPerGroup.columns):
                scoresPerGroup[col].plot(kind="kde", ax=ax, linestyle="--", color=cmap(i))

    ax.axvline(boundary, 0, 1, label="decision boundary", color="black")

    plt.xlabel("scores")
    # plt.legend(bbox_to_anchor=(.7, 1.05))
    plt.savefig(filename, dpi=100, bbox_inches="tight")
    plt.cla()


def plotScoreHistsPerGroup(data, binArray, scoreNames, filename, groups, xTickLabels=None, yMin=None, yMax=None):

    mpl.rcParams.update(
        {
            "font.size": 24,
            "lines.linewidth": 3,
            "lines.markersize": 15,
            "font.family": "Times New Roman",
        }
    )
    # avoid type 3 (i.e. bitmap) fonts in figures
    mpl.rcParams["ps.useafm"] = True
    mpl.rcParams["pdf.use14corefonts"] = True
    mpl.rcParams["text.usetex"] = True

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111)
    cmap = plt.get_cmap("tab10")

    for attr in scoreNames:
        scoresPerGroup = util.scoresByGroup(data, list(groups.keys()), attr)
        scoresPerGroup = scoresPerGroup.rename(groups, axis="columns")
        origBinEdges = sorted(data[attr].unique())

        for i, col in enumerate(scoresPerGroup.columns):
            if "truncated" in attr:
                scoresPerGroup[col].plot(
                    kind="hist", ax=ax, bins=origBinEdges, color=cmap(i), alpha=0.5, xticks=binArray
                )
            else:
                scoresPerGroup[col].plot(
                    kind="hist",
                    ax=ax,
                    bins=binArray,
                    color=cmap(i),
                    alpha=1,
                    density=False,
                    xticks=binArray,
                )

    if xTickLabels is not None:
        ax.set_xticklabels(xTickLabels)
    cnt = 0
    for label in ax.xaxis.get_ticklabels():
        if (cnt % 10) != 0:
            label.set_visible(False)
        cnt += 1
    ax.set_ylim(ymin=yMin, ymax=yMax)

    plt.xlabel("scores")
    plt.xticks(rotation=90)
    plt.savefig(filename, dpi=100, bbox_inches="tight")
    plt.cla()
