from torch.utils import data
from torchvision.transforms.functional import normalize

from neosr.data.transforms import basic_augment, paired_random_crop
from neosr.utils import bgr2ycbcr, img2tensor
from neosr.utils.registry import DATASET_REGISTRY

import vapoursynth as vs
import numpy as np

@DATASET_REGISTRY.register()
class vapoursynth(data.Dataset):
    """Load data from vapoursynth scripts
    """

    def __init__(self, opt):
        super(vapoursynth, self).__init__()
        self.opt = opt
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_script, self.lq_script = opt['script_gt'], opt['script_lq']

        with open(self.gt_script, "rb") as hr_file:
            code = compile(hr_file.read(), self.gt_script, "exec")
        exec(code)
        self.gt_clip, _, _ = vs.get_output(0)

        with open(self.lq_script, "rb") as lr_file:
            code = compile(lr_file.read(), self.lq_script, "exec")
        exec(code)
        self.lq_clip, _, _ = vs.get_output(0)

    def __getitem__(self, index):
        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        frame_gt = self.gt_clip.get_frame(index)
        img_gt = np.stack([np.asarray(frame_gt[plane]) for plane in range(frame_gt.format.num_planes)], axis=2)
        frame_lq = self.lq_clip.get_frame(index)
        img_lq = np.stack([np.asarray(frame_lq[plane]) for plane in range(frame_lq.format.num_planes)], axis=2)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq = paired_random_crop(
                img_gt, img_lq, gt_size, scale)
            # flip, rotation
            img_gt, img_lq = basic_augment(
                [img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] *
                              scale, 0:img_lq.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor(
            [img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt}

    def __len__(self):
        return len(self.lq_clip)