from .common import HanSegSliceDataset


class ValDataset(HanSegSliceDataset):
    def __init__(self, data_args):
        super().__init__(data_args, split="val")

