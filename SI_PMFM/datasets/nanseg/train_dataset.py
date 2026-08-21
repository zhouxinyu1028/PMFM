from .common import HanSegSliceDataset


class TrainDataset(HanSegSliceDataset):
    def __init__(self, data_args):
        super().__init__(data_args, split="train")

