import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def performancePerBinAndGroup():
    genderData = pd.DataFrame(
        data=[
            [0.788, 0.779, 0.825],
            [0.688, 0.650, 0.830],
            [0.628, 0.610, 0.694],
            [0.570, 0.553, 0.641],
            [0.470, 0.495, 0.387],
            [0.555, 0.580, 0.462],
            [0.582, 0.597, 0.511],
            [0.686, 0.707, 0.550],
            [0.688, 0.699, 0.614],
            [0.779, 0.793, 0.689],
        ],
        columns=["All", "Male", "Female"],
    )
    raceData = pd.DataFrame(
        data=[
            [0.788, 0.796, 0.770, 0.758, 0.833],
            [0.688, 0.691, 0.702, 0.653, 0.643],
            [0.628, 0.659, 0.586, 0.705, 0.615],
            [0.570, 0.602, 0.540, 0.694, 0.476],
            [0.470, 0.442, 0.479, 0.512, 0.524],
            [0.555, 0.560, 0.559, 0.600, 0.400],
            [0.582, 0.595, 0.588, 0.483, 0.538],
            [0.686, 0.727, 0.688, 0.421, 0.727],
            [0.688, 0.688, 0.696, 0.529, 0.700],
            [0.779, 0.700, 0.802, 0.667, 0.818],
        ],
        columns=["All", "White", "Black", "Hispanic", "Other"],
    )
    ageData = pd.DataFrame(
        data=[
            [0.788, 0.814, 0.761, 0.333],
            [0.688, 0.730, 0.685, 0.630],
            [0.628, 0.589, 0.650, 0.603],
            [0.570, 0.622, 0.580, 0.530],
            [0.470, 0.400, 0.448, 0.545],
            [0.555, 0.541, 0.554, 0.562],
            [0.582, 0.548, 0.610, 0.544],
            [0.686, 0.612, 0.717, 0.655],
            [0.688, 0.610, 0.705, 0.682],
            [0.799, 0.750, 0.749, 0.836],
        ],
        columns=["All", "$>45$", "25 -- 45", "$<25$"],
    )

    plotData = {
        os.getcwd() + "/results/compas/gender/accuracyPerScore.png": genderData,
        os.getcwd() + "/results/compas/race/accuracyPerScore.png": raceData,
        os.getcwd() + "/results/compas/age/accuracyPerScore.png": ageData,
    }
    colormap = ["#1E77B4", "#FF7F0E", "#2CA02C", "#D62728", "#7F7F7F"]

    for filename, data in plotData.items():
        cols = data.columns.tolist()
        cols = cols[1:] + [cols[0]]
        data = data[cols]
        colors = colormap[: len(cols) - 1] + [colormap[-1]]

        plott(data, filename, colors)


def plott(data, filename, colormap):
    mpl.rcParams.update(
        {
            "font.size": 30,
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

    data.plot(kind="bar", ax=ax, color=colormap)
    ax.set_xticklabels(range(1, 11))

    plt.xlabel("Compas Scores")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=360)
    plt.legend(bbox_to_anchor=(1.0, 1.03))
    plt.savefig(filename, dpi=100, bbox_inches="tight")
    plt.cla()


if __name__ == "__main__":
    performancePerBinAndGroup()
