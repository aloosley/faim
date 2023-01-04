import os
from pathlib import Path
import pandas as pd


class ZalandoDataset:
    def __init__(self, input_filepath: Path, output_dir: Path) -> None:
        raw = pd.read_csv(input_filepath)
        cols = ["group", "groundTruthLabel", "pred_score"]
        df = raw.rename(columns={"raw_score": "pred_score", "label": "groundTruthLabel"})[cols]
        df.group = df.group.replace({"high": 0, "low": 1})

        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_dir / "data.csv", index=False, header=True)
