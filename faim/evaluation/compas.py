import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.metrics import classification_report

DECIMAL_PLACES = 3


def evaluate(data, pathToResult, filename="eval.txt"):
    # low scores in compas range from 1-4, medium from 5-7, high 8-10
    # since all scores are normalized by now, we have to take that into account
    lowScoreLimit = 0.4
    data["isHighScore_pred"] = 0
    data.loc[data["pred_score_truncated"] > lowScoreLimit, "isHighScore_pred"] = 1
    data["isHighScore_fair"] = 0
    data.loc[data["fairScore"] > lowScoreLimit, "isHighScore_fair"] = 1

    # convert relevant columns to categories
    for col in ["group", "c_charge_degree"]:
        data[col] = data[col].astype("category")

    # start eval string
    resultString = "\n================================================\nSUMMARY FOR RAW COMPAS SCORES \n================================================\n"
    resultString += evaluateWithGLM("isHighScore_pred", data)
    appendString, rawModelPerformance, rawScoreErrorRates = evaluateModelPerformanceAndErrorRates(
        data, "isHighScore_pred"
    )
    resultString += appendString

    resultString += "\n\n\n================================================\nSUMMARY FOR FAIR SCORES \n================================================\n"
    resultString += evaluateWithGLM("isHighScore_fair", data)
    appendString, fairModelPerformance, fairScoreErrorRates = evaluateModelPerformanceAndErrorRates(
        data, "isHighScore_fair"
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
    resultFile = os.path.join(os.path.dirname(pathToResult), filename)
    evalFile = open(resultFile, "w")
    evalFile.write(resultString)
    evalFile.close()


def evaluateModelPerformanceAndErrorRates(data, scoreAttr):
    modelPerformances = pd.DataFrame(columns=["Accuracy", "Precision", "Recall"])
    allErrorRates = pd.DataFrame(columns=["FPR", "FNR"])

    # collect model performances
    report = classification_report(data["groundTruthLabel"], data[scoreAttr], output_dict=True, zero_division=1)
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
    confusionMatrixAll = confusionMatrixAll.reindex(index=[0, 1, "All"], columns=[0, 1, "All"], fill_value=0)
    falsePositiveRate = round(confusionMatrixAll.loc[0, 1] / confusionMatrixAll.loc[0, "All"], DECIMAL_PLACES)
    falseNegativeRate = round(confusionMatrixAll.loc[1, 0] / confusionMatrixAll.loc[1, "All"], DECIMAL_PLACES)
    allErrorRates = allErrorRates.append(pd.Series({"FPR": falsePositiveRate, "FNR": falseNegativeRate}, name="all"))

    # print stuff
    resultString = "\nERROR RATES ALL DEFENDANTS \n====================================\n"
    resultString += classification_report(data["groundTruthLabel"], data[scoreAttr], zero_division=1) + "\n\n"
    resultString += str(confusionMatrixAll) + "\n\n"

    for group in sorted(data["group"].unique()):
        groupData = data[data["group"] == group]
        # collect model performances
        report = classification_report(
            groupData["groundTruthLabel"], groupData[scoreAttr], output_dict=True, zero_division=1
        )
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
        confusionMatrixGroup = pd.crosstab(
            groupData["groundTruthLabel"], groupData[scoreAttr], dropna=False, margins=True
        )
        confusionMatrixGroup = confusionMatrixGroup.reindex(index=[0, 1, "All"], columns=[0, 1, "All"], fill_value=0)
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
        resultString += (
            classification_report(groupData["groundTruthLabel"], groupData[scoreAttr], zero_division=1) + "\n\n"
        )
        resultString += str(confusionMatrixGroup) + "\n\n"

    return resultString, modelPerformances, allErrorRates


def evaluateWithGLM(scoreAttr, data):
    # train logistic regression model
    model = smf.glm(
        formula=scoreAttr + " ~ group + groundTruthLabel + c_charge_degree + priors_count",
        data=data,
        family=sm.families.Binomial(),
    )
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
            + str(round(oddsFactor.values[0], DECIMAL_PLACES))
            + "\n"
        )

    return resultString


def evaluatePerBin(data, pathToResult):
    for score in range(1, 11):
        temp = data[data["pred_score"] == score].copy()
        # low scores in compas range from 1-4, medium from 5-7, high 8-10
        # since all scores are normalized by now, we have to take that into account
        lowScoreLimit = 0.4
        temp["isHighScore_pred"] = 0
        temp.loc[temp["pred_score_truncated"] > lowScoreLimit, "isHighScore_pred"] = 1
        temp["isHighScore_fair"] = 0
        temp.loc[temp["fairScore"] > lowScoreLimit, "isHighScore_fair"] = 1

        # convert relevant columns to categories
        for col in ["group", "c_charge_degree"]:
            temp[col] = temp[col].astype("category")

        # start eval string
        resultString = (
            "\n================================================\nSUMMARY FOR RAW COMPAS SCORES IN BIN "
            + str(score)
            + "\n================================================\n"
        )
        appendString, rawModelPerformance, rawScoreErrorRates = evaluateModelPerformanceAndErrorRates(
            temp, "isHighScore_pred"
        )
        resultString += appendString

        resultString += (
            "\n\n\n================================================\nSUMMARY FOR FAIR SCORES IN BIN "
            + str(score)
            + " \n================================================\n"
        )
        appendString, fairModelPerformance, fairScoreErrorRates = evaluateModelPerformanceAndErrorRates(
            temp, "isHighScore_fair"
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
        errDiffs = pd.DataFrame(
            columns=["raw FPR", "raw FNR", "fair FPR", "fair FNR", "raw - fair FPR", "raw - fair FNR"]
        )
        errDiffs["raw FPR"] = rawScoreErrorRates["FPR"]
        errDiffs["raw FNR"] = rawScoreErrorRates["FNR"]
        errDiffs["fair FPR"] = fairScoreErrorRates["FPR"]
        errDiffs["fair FNR"] = fairScoreErrorRates["FNR"]
        errDiffs["raw - fair FPR"] = rawScoreErrorRates["FPR"] - fairScoreErrorRates["FPR"]
        errDiffs["raw - fair FNR"] = rawScoreErrorRates["FNR"] - fairScoreErrorRates["FNR"]

        resultString += "\n\n\n================================================\nERROR RATE DIFFERENCES \n================================================\n"
        resultString += errDiffs.to_string()

        # write results to file
        resultFile = os.path.join(os.path.dirname(pathToResult), "eval_bin=" + str(score) + ".txt")
        evalFile = open(resultFile, "w")
        evalFile.write(resultString)
        evalFile.close()
