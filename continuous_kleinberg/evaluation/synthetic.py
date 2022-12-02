import sklearn.metrics as skmetr
import statsmodels.formula.api as smf
import statsmodels.api as sm
import pandas as pd
import numpy as np
import os

DECIMAL_PLACES = 3


def evaluate(data, pathToResult):
    # calculate model from true scores (decision boundary was set as mean of true scores) for fair label calc
    x = data["true_score"]
    data["true_score"] = (x - min(x)) / (max(x) - min(x))
    boundary = data["true_score"].mean()
    data["fairscoreLabel"] = 0
    data.loc[data["fairScore"] >= boundary, "fairscoreLabel"] = 1

    # convert group column to category
    data["group"] = data["group"].astype("category")

    # start eval string
    resultString = "\n================================================\nSUMMARY FOR RAW SYNTHETIC SCORES \n================================================\n"
    resultString += evaluateWithGLM_trueLabel(data) + "\n"
    resultString += evaluateWithGLM("predictedLabel", data)
    appendString, rawModelPerformance, rawScoreErrorRates = evaluateModelPerformanceAndErrorRates(
        data, "predictedLabel"
    )
    resultString += appendString

    resultString += "\n\n\n================================================\nSUMMARY FOR FAIR SCORES \n================================================\n"
    resultString += evaluateWithGLM("fairscoreLabel", data)
    appendString, fairModelPerformance, fairScoreErrorRates = evaluateModelPerformanceAndErrorRates(
        data, "fairscoreLabel"
    )
    resultString += appendString

    # calc diffs in model performance
    perfDiffs = pd.DataFrame(
        columns=[
            "raw Acc",
            "raw Prec",
            "raw Recall",
            "fair Acc",
            "fair Prec",
            "fair Recall",
            "raw - fair Acc",
            "raw - fair Prec",
            "raw - fair Recall",
        ]
    )
    perfDiffs["raw Acc"] = rawModelPerformance["Accuracy"]
    perfDiffs["raw Prec"] = rawModelPerformance["Precision"]
    perfDiffs["raw Recall"] = rawModelPerformance["Recall"]
    perfDiffs["fair Acc"] = fairModelPerformance["Accuracy"]
    perfDiffs["fair Prec"] = fairModelPerformance["Precision"]
    perfDiffs["fair Recall"] = fairModelPerformance["Recall"]
    perfDiffs["raw - fair Acc"] = rawModelPerformance["Accuracy"] - fairModelPerformance["Accuracy"]
    perfDiffs["raw - fair Prec"] = rawModelPerformance["Precision"] - fairModelPerformance["Precision"]
    perfDiffs["raw - fair Recall"] = rawModelPerformance["Recall"] - fairModelPerformance["Recall"]

    resultString += "\n\n\n================================================\nPERFORMANCE DIFFERENCES \n================================================\n"
    resultString += perfDiffs.to_string()

    # calc diffs in error rates
    errDiffs = pd.DataFrame(columns=["raw FPR", "raw FNR", "fair FPR", "fair FNR", "raw - fair FPR", "raw - fair FNR"])
    errDiffs["raw FPR"] = rawScoreErrorRates["FPR"]
    errDiffs["raw FNR"] = rawScoreErrorRates["FNR"]
    errDiffs["fair FPR"] = fairScoreErrorRates["FPR"]
    errDiffs["fair FNR"] = fairScoreErrorRates["FNR"]
    errDiffs["raw - fair FPR"] = rawScoreErrorRates["FPR"] - fairScoreErrorRates["FPR"]
    errDiffs["raw - fair FNR"] = rawScoreErrorRates["FNR"] - fairScoreErrorRates["FNR"]

    resultString += "\n\n\n================================================\nERROR RATE DIFFERENCES \n================================================\n"
    resultString += errDiffs.to_string()

    # write results to file
    resultFile = os.path.join(os.path.dirname(pathToResult), "eval.txt")
    evalFile = open(resultFile, "w")
    evalFile.write(resultString)
    evalFile.close()


def evaluateModelPerformanceAndErrorRates(data, scoreAttr):
    modelPerformances = pd.DataFrame(columns=["Accuracy", "Precision", "Recall"])
    allErrorRates = pd.DataFrame(columns=["FPR", "FNR"])

    # collect model performances
    report = skmetr.classification_report(data["groundTruthLabel"], data[scoreAttr], output_dict=True)
    modelPerformances = modelPerformances.append(
        pd.Series(
            {
                "Precision": round(report.get("weighted avg").get("precision"), DECIMAL_PLACES),
                "Recall": round(report.get("weighted avg").get("recall"), DECIMAL_PLACES),
                "Accuracy": round(report.get("accuracy"), DECIMAL_PLACES),
            },
            name="all",
        )
    )

    # calc error rates
    confusionMatrixAll = pd.crosstab(data["groundTruthLabel"], data[scoreAttr], margins=True)
    falsePositiveRate = round(confusionMatrixAll.loc[0, 1] / confusionMatrixAll.loc[0, "All"], DECIMAL_PLACES)
    falseNegativeRate = round(confusionMatrixAll.loc[1, 0] / confusionMatrixAll.loc[1, "All"], DECIMAL_PLACES)
    allErrorRates = allErrorRates.append(pd.Series({"FPR": falsePositiveRate, "FNR": falseNegativeRate}, name="all"))

    # print stuff
    resultString = "\nERROR RATES ALL DEFENDANTS \n====================================\n"
    resultString += skmetr.classification_report(data["groundTruthLabel"], data[scoreAttr]) + "\n\n"
    resultString += str(confusionMatrixAll) + "\n\n"

    for group in sorted(data["group"].unique()):
        groupData = data[data["group"] == group]
        # collect model performances
        report = skmetr.classification_report(groupData["groundTruthLabel"], groupData[scoreAttr], output_dict=True)
        modelPerformances = modelPerformances.append(
            pd.Series(
                {
                    "Precision": round(report.get("weighted avg").get("precision"), DECIMAL_PLACES),
                    "Recall": round(report.get("weighted avg").get("recall"), DECIMAL_PLACES),
                    "Accuracy": round(report.get("accuracy"), DECIMAL_PLACES),
                },
                name=group,
            )
        )

        # calc error rates
        confusionMatrixGroup = pd.crosstab(groupData["groundTruthLabel"], groupData[scoreAttr], margins=True)
        falsePositiveRateGroup = round(
            confusionMatrixGroup.loc[0, 1] / confusionMatrixGroup.loc[0, "All"], DECIMAL_PLACES
        )
        falseNegativeRateGroup = round(
            confusionMatrixGroup.loc[1, 0] / confusionMatrixGroup.loc[1, "All"], DECIMAL_PLACES
        )
        allErrorRates = allErrorRates.append(
            pd.Series({"FPR": falsePositiveRateGroup, "FNR": falseNegativeRateGroup}, name=group)
        )

        # print
        resultString += "ERROR RATES GROUP " + str(group) + " DEFENDANTS \n====================================\n"
        resultString += skmetr.classification_report(groupData["groundTruthLabel"], groupData[scoreAttr]) + "\n\n"
        resultString += str(confusionMatrixGroup) + "\n\n"

    return resultString, modelPerformances, allErrorRates


def evaluateWithGLM(scoreAttr, data):
    decPlaces = 6

    # train logistic regression model
    model = smf.glm(formula=scoreAttr + " ~ group + groundTruthLabel", data=data, family=sm.families.Binomial())
    result = model.fit()
    resultString = str(result.summary()) + "\n\n"

    # calculate the odds
    asHtml = result.summary().tables[1].as_html()
    resultFrame = pd.read_html(asHtml, header=0, index_col=0)[0]
    intercept = resultFrame.loc["Intercept", "coef"]
    control = np.exp(intercept) / (1 + np.exp(intercept))

    for groupCat in data["group"].unique():
        row = resultFrame.filter(regex="group..\." + str(groupCat) + ".", axis=0)
        if row.empty:
            continue
        oddsFactor = np.exp(row["coef"]) / (1 - control + (control * np.exp(row["coef"])))
        resultString += (
            "Group "
            + str(groupCat)
            + "'s chance to have a high score compared to non-protected is "
            + str(round(oddsFactor.values[0], decPlaces))
            + "\n"
        )

    return resultString


def evaluateWithGLM_trueLabel(data):
    decPlaces = 6

    # train logistic regression model
    model = smf.glm(formula="groundTruthLabel ~ group", data=data, family=sm.families.Binomial())
    result = model.fit()
    resultString = str(result.summary()) + "\n\n"

    # calculate the odds
    asHtml = result.summary().tables[1].as_html()
    resultFrame = pd.read_html(asHtml, header=0, index_col=0)[0]
    intercept = resultFrame.loc["Intercept", "coef"]
    control = np.exp(intercept) / (1 + np.exp(intercept))

    for groupCat in data["group"].unique():
        row = resultFrame.filter(regex="group..\." + str(groupCat) + ".", axis=0)
        if row.empty:
            continue
        oddsFactor = np.exp(row["coef"]) / (1 - control + (control * np.exp(row["coef"])))
        resultString += (
            "Group "
            + str(groupCat)
            + "'s chance to have a high score compared to non-protected is "
            + str(round(oddsFactor.values[0], decPlaces))
            + "\n"
        )

    return resultString
