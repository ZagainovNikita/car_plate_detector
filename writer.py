import pandas as pd
from typing import Literal
import config


class RecordManager:
    def __init__(
        self, file_path=config.RECORDS,
        checkpoint_freq=config.CHECKPOINTS_FREQ, initialize=False
    ):
        self.file_path = file_path
        self.checkpoint_freq = checkpoint_freq
        self.initialize = initialize
        if not self.initialize:
            self.df = pd.read_csv(file_path)
        else:
            self.df = pd.DataFrame(
                data=[],
                columns=config.RECORD_COLUMNS
            )
        self._last_save = 0

    def add_record(
        self, video_id, timestamp, car_id, car_box, plate_box, transcript, confidence
    ):

        self.df.loc[len(self.df)] = [
            video_id, timestamp, car_id, car_box, plate_box, transcript, confidence
        ]

        if len(self.df) - self._last_save > self.checkpoint_freq:
            self.df.to_csv(self.file_path, index=False)

    def save(self):
        self.df.to_csv(self.file_path, index=False)

    def get_records(self, video_id, criterion: Literal["confidence", "most_common"] = "most_common"):
        video_records = self.df[self.df["video_id"] == video_id]
        if criterion == "confidence":
            result = video_records.loc[video_records.groupby("car_id")["confidence"].idxmax().values][
                ["car_id", "transcript", "confidence"]
            ]

            return result

        if criterion == "most_common":
            grouped = video_records.groupby(["car_id", "transcript"], as_index=False).count()[
                ["car_id", "transcript", "timestamp"]]
            sums = grouped.groupby(["car_id"], as_index=False).sum()[
                ["car_id", "timestamp"]].rename(columns={"timestamp": "total"})
            grouped = grouped.merge(sums, "left", "car_id").rename(
                columns={"timestamp": "support"})
            grouped["frequency"] = grouped["support"] / grouped["total"]
            result = grouped.loc[grouped.groupby("car_id")["frequency"].idxmax().values][[
                "car_id", "transcript", "frequency", "support"]].reset_index(drop=True)

            return result


if __name__ == "__main__":
    recorder = RecordManager()
    print("#" * 50)
    print(recorder.get_records("./data/sample.mp4", criterion="confidence"))
    print("#" * 50)
    print(recorder.get_records("./data/sample.mp4", criterion="most_common"))
    print("#" * 50)
