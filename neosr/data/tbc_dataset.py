from torch.utils import data
from neosr.utils.registry import DATASET_REGISTRY
import numpy as np
from pathlib import Path

UINT16_MAX = 2**16 - 1

@DATASET_REGISTRY.register()
class paired_tbc(data.Dataset):
    """Use .tbc file as lq and .npy files as gt
    """

    def __init__(self, opt):
        super(paired_tbc, self).__init__()
        self.opt = opt

        self.field = opt['field']
        self.gt_folder = opt['dataroot_gt']
        self.gt_list = sorted(Path(self.gt_folder).glob("*.npy"))
        self.lq_tbc = opt['tbc']

    def __getitem__(self, index):
        # Load gt images. Dimension order: CHW; channel order: YUV;
        # image range: [0, 1], float32.
        img_gt = np.load(self.gt_list[index])

        # Load lq images. Dimension order: CHW;
        # signal range: [0, 1], float32.
        if self.field != "top":
            offset = 0
        else:
            offset = 243
        img_lq = np.stack([
            (np.fromfile(
                self.lq_tbc,
                dtype=np.uint16,
                count=910 * 526,
                offset=index * 910 * 526 * 2).astype(np.float32) / UINT16_MAX
            ).reshape(526, 910)[39 + offset:39 + 243 + offset:, 147:905:]
        ])

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
        self.field = opt['field']
        self.lq_tbc = Path(opt['tbc'])
        self.frames = opt['frames']

    def __getitem__(self, index):
        if self.field != "top":
            offset = 0
        else:
            offset = 243
        img_lq = np.stack([
            (np.fromfile(
                self.lq_tbc,
                dtype=np.uint16,
                count=910 * 526,
                offset=self.frames[index] * 910 * 526 * 2).astype(np.float32) / UINT16_MAX
            ).reshape(526, 910)[39 + offset:39 + 243 + offset:, 147:905:]
        ])
        return {'lq': img_lq, 'lq_path': str(self.lq_tbc.with_stem(f'{self.lq_tbc.stem}_frame_{self.frames[index]}'))}

    def __len__(self):
        return len(self.frames)