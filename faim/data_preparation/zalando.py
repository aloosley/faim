import os
from pathlib import Path
import pandas as pd


class ZalandoDataset:
    def __init__(self, input_file, output_path):
        raw = pd.read_csv(input_file)
        cols = ["group", "groundTruthLabel", "pred_score"]
        df = raw.rename(columns={"raw_score": "pred_score", "label": "groundTruthLabel"})[cols]
        df.group = df.group.replace({"high" : 0, "low" : 1})
        Path(output_path).mkdir(parents=True, exist_ok=True)
        df.to_csv(os.path.join(output_path, "data.csv"), index=False, header=True)
