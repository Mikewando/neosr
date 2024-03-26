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
        self.lq_tbcs = [Path(tbc) for tbc in opt['tbc']]
        for tbc in self.lq_tbcs:
            if tbc.stat().st_size != len(self.gt_list) * 526 * 910 * 2:
                raise Exception('GT count does not match expected TBC size')
        # TODO only the luma will have one, this enforces luma first,
        # we should be more explicit or flexible I guess
        with open(self.lq_tbcs[0].with_suffix('.tbc.json')) as tbc_json_file:
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
        stacked_lq = []
        #for lq_tbc in self.lq_tbcs:
        #    bottom, top = np.split(
        #        (np.fromfile(
        #            lq_tbc,
        #            dtype=np.uint16,
        #            count=910 * 526,
        #            offset=index * 910 * 526 * 2).astype(np.float32) / UINT16_MAX
        #        ).reshape(526, 910), 2)
        #    # Convert lq input. Dimension order: CHW;
        #    # signal range: [0, 1], float32.
        #    img_lq = np.zeros((486, self.active_end - self.active_start), dtype=np.float32)
        #    img_lq[0::2] = top[19:262:, self.active_start:self.active_end:]
        #    img_lq[1::2] = bottom[20::, self.active_start:self.active_end:]
        #    stacked_lq.append(img_lq)
        if len(self.lq_tbcs) == 2:
            luma_tbc, chroma_tbc = self.lq_tbcs
        else:
            luma_tbc, = self.lq_tbcs
            chroma_tbc = None

        src = np.fromfile(
            luma_tbc,
            dtype=np.uint16,
            count=910 * 526,
            offset=index * 910 * 526 * 2
        ).astype(np.float32) / UINT16_MAX
        src = src.reshape(526, 910)

        bottom, top = np.split(src, 2)

        img_lq = np.zeros((486, self.active_end - self.active_start), dtype=np.float32)
        img_lq[0::2] = top[19:262:, self.active_start:self.active_end:]
        img_lq[1::2] = bottom[20::, self.active_start:self.active_end:]
        stacked_lq.append(img_lq)

        if chroma_tbc:
            src = np.fromfile(
                chroma_tbc,
                dtype=np.uint16,
                count=910 * 526,
                offset=index * 910 * 526 * 2
            ).astype(np.float32) / UINT16_MAX
            src = src.reshape(526, 910)

            bottom, top = np.split(src, 2)
            # Normalize the phase between lines in field
            bottom[20::2, self.active_start:self.active_end:] = bottom[20::2, self.active_start:self.active_end:] * -1 + 1
            top[19:262:2, self.active_start:self.active_end:] = top[19:262:2, self.active_start:self.active_end:] * -1 + 1

            img_lq = np.zeros((486, self.active_end - self.active_start), dtype=np.float32)
            img_lq[0::2] = top[19:262:, self.active_start:self.active_end:]
            img_lq[1::2] = bottom[20::, self.active_start:self.active_end:]
            stacked_lq.append(img_lq)

        return {'lq': np.stack(stacked_lq), 'gt': img_gt}

    def __len__(self):
        return len(self.gt_list)


@DATASET_REGISTRY.register()
class single_tbc(data.Dataset):
    """Use .tbc file as lq
    """

    def __init__(self, opt):
        super(single_tbc, self).__init__()
        self.opt = opt

        self.lq_tbcs = [Path(tbc) for tbc in opt['tbc']]
        with open(self.lq_tbcs[0].with_suffix('.tbc.json')) as tbc_json_file:
            tbc_json = json.load(tbc_json_file)
        self.active_start = tbc_json['videoParameters']['activeVideoStart']
        self.active_end = tbc_json['videoParameters']['activeVideoEnd']
        self.frames = opt['frames']

    def __getitem__(self, index):
        stacked_lq = []
        if len(self.lq_tbcs) == 2:
            luma_tbc, chroma_tbc = self.lq_tbcs
        else:
            luma_tbc, = self.lq_tbcs
            chroma_tbc = None

        src = np.fromfile(
            luma_tbc,
            dtype=np.uint16,
            count=910 * 526,
            offset=self.frames[index] * 910 * 526 * 2
        ).astype(np.float32) / UINT16_MAX
        src = src.reshape(526, 910)

        bottom, top = np.split(src, 2)

        img_lq = np.zeros((486, self.active_end - self.active_start), dtype=np.float32)
        img_lq[0::2] = top[19:262:, self.active_start:self.active_end:]
        img_lq[1::2] = bottom[20::, self.active_start:self.active_end:]
        stacked_lq.append(img_lq)

        if chroma_tbc:
            src = np.fromfile(
                chroma_tbc,
                dtype=np.uint16,
                count=910 * 526,
                offset=self.frames[index] * 910 * 526 * 2
            ).astype(np.float32) / UINT16_MAX
            src = src.reshape(526, 910)

            bottom, top = np.split(src, 2)
            # Normalize the phase between lines in field
            bottom[20::2, self.active_start:self.active_end:] = bottom[20::2, self.active_start:self.active_end:] * -1 + 1
            top[19:262:2, self.active_start:self.active_end:] = top[19:262:2, self.active_start:self.active_end:] * -1 + 1

            img_lq = np.zeros((486, self.active_end - self.active_start), dtype=np.float32)
            img_lq[0::2] = top[19:262:, self.active_start:self.active_end:]
            img_lq[1::2] = bottom[20::, self.active_start:self.active_end:]
            stacked_lq.append(img_lq)

        return {'lq': np.stack(stacked_lq), 'lq_path': str(self.lq_tbcs[0].with_stem(f'{self.lq_tbcs[0].stem}_frame_{self.frames[index]}'))}

    def __len__(self):
        return len(self.frames)