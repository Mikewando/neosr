from torch.utils import data
from neosr.utils.registry import DATASET_REGISTRY
import numpy as np
import json
from pathlib import Path

UINT16_MAX = 2**16 - 1

@DATASET_REGISTRY.register()
class paired_tbc(data.Dataset):
    """Use .tbc file as lq and .npy files as gt
    """

    def __init__(self, opt):
        super(paired_tbc, self).__init__()
        self.opt = opt

        self.gt_folder = opt['dataroot_gt']
        self.gt_list = sorted(Path(self.gt_folder).glob("*.npy"))
        self.lq_tbc = Path(opt['tbc'])
        if self.lq_tbc.stat().st_size != len(self.gt_list) * 526 * 910 * 2:
            raise Exception('GT count does not match expected TBC size')
        with open(self.lq_tbc.with_suffix('.tbc.json')) as tbc_json_file:
            tbc_json = json.load(tbc_json_file)
        self.active_start = tbc_json['videoParameters']['activeVideoStart']
        self.active_end = tbc_json['videoParameters']['activeVideoEnd']

    def __getitem__(self, index):
        # Load gt images. Dimension order: CHW; channel order: YUV;
        # image range: [0, 1], float32.
        img_gt = np.load(self.gt_list[index])

        # Load TBC for lq input. Stored as uint16 samples in sequence.
        # Each complete frame is 525(+1 empty) rows of 910 samples.
        # The frames are stored with the bottom field stacked on top of
        # the top field. We split the fields and then interleave them TFF.
        bottom, top = np.split(
            (np.fromfile(
                self.lq_tbc,
                dtype=np.uint16,
                count=910 * 526,
                offset=index * 910 * 526 * 2).astype(np.float32) / UINT16_MAX
            ).reshape(526, 910), 2)
        img_lq = np.zeros((526, 910), dtype=np.float32)
        img_lq[0::2] = top
        img_lq[1::2] = bottom

        # Convert lq input. Dimension order: CHW;
        # signal range: [0, 1], float32.
        img_lq = np.stack([img_lq[39:525:, self.active_start:self.active_end:]])

        return {'lq': img_lq, 'gt': img_gt}

    def __len__(self):
        return len(self.gt_list)


@DATASET_REGISTRY.register()
class single_tbc(data.Dataset):
    """Use .tbc file as lq
    """

    def __init__(self, opt):
        super(single_tbc, self).__init__()
        self.opt = opt

        self.lq_tbc = Path(opt['tbc'])
        with open(self.lq_tbc.with_suffix('.tbc.json')) as tbc_json_file:
            tbc_json = json.load(tbc_json_file)
        self.active_start = tbc_json['videoParameters']['activeVideoStart']
        self.active_end = tbc_json['videoParameters']['activeVideoEnd']
        self.frames = opt['frames']

    def __getitem__(self, index):
        bottom, top = np.split(
            (np.fromfile(
                self.lq_tbc,
                dtype=np.uint16,
                count=910 * 526,
                offset=self.frames[index] * 910 * 526 * 2).astype(np.float32) / UINT16_MAX
            ).reshape(526, 910), 2)
        img_lq = np.zeros((526, 910), dtype=np.float32)
        img_lq[0::2] = top
        img_lq[1::2] = bottom
        img_lq = np.stack([img_lq[39:525:, self.active_start:self.active_end:]])
        return {'lq': img_lq, 'lq_path': str(self.lq_tbc.with_stem(f'{self.lq_tbc.stem}_frame_{self.frames[index]}'))}

    def __len__(self):
        return len(self.frames)