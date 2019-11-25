import os
from sklearn.utils import shuffle
import cv2
from skimage import io
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True
Tensor = torch.cuda.FloatTensor

def load_unit(path, preproc):
    # Load
    file_suffix = path.split('.')[-1].lower()
    if file_suffix in ['jpg', 'png']:
        try:
            unit = cv2.cvtColor(io.imread(path).astype(np.uint8), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print('{} load exception:\n'.format(path), e)
            unit = cv2.cvtColor(np.array(Image.open(path).convert('RGB')), cv2.COLOR_RGB2BGR)
    else:
        print('Unsupported file type.')
        return None
    # Preprocessing
    if 'BF' in preproc and 'train_vid_frames' in path:
        unit = cv2.bilateralFilter(unit, 9, 75, 75)
    if 'resize' in preproc:
        unit = cv2.resize(unit, (384, 384), interpolation=cv2.INTER_LANCZOS4)
    elif 'downsample' in preproc:
        unit = cv2.resize(unit, unit.shape[1]//2, unit.shape[0]//2, interpolation=cv2.INTER_LANCZOS4)
    unit = cv2.cvtColor(unit, cv2.COLOR_BGR2RGB)
    try:
        unit = unit.astype(np.float32) / 127.5 - 1.0
    except Exception as e:
        print('EX:', e, unit.shape, unit.dtype)

    unit = np.transpose(unit, (2, 0, 1))
    return unit


def unit_postprocessing(unit, residual=None, vid_size=None):
    unit = unit.squeeze()
    if residual is not None:
        if residual.shape[-2:] != unit.shape[-2:]:
            residual = resize2d(residual, unit.shape[-2:])
        residual = residual.squeeze()
        unit = torch.clamp(unit - residual, -1.0, 1.0)
    unit = unit.cpu().detach().numpy()
    unit = np.round((np.transpose(unit, (1, 2, 0)) + 1.0) * 127.5).astype(np.uint8)
    if unit.shape[:2][::-1] != vid_size:
        unit = cv2.resize(unit, vid_size, interpolation=cv2.INTER_CUBIC)
    return unit


def get_paths_ABC(config, mode):
    if mode in ('train', 'test_on_trainset'):
        dir_root = config.dir_train
    elif mode == 'test_on_testset':
        dir_root = config.dir_test
    else:
        val_vid = mode.split('_')[-1]
        dir_root = eval('config.dir_{}'.format(val_vid))
    paths_A, paths_C, paths_skip_intermediate, paths_skip, paths_mag = [], [], [], [], []
    if config.cursor_end > 0 or 'test' in mode:
        dir_A = os.path.join(dir_root, 'frameA')
        files_A = sorted(os.listdir(dir_A), key=lambda x: int(x.split('.')[0]))
        paths_A = [os.path.join(dir_A, file_A) for file_A in files_A]
        if mode == 'train' and isinstance(config.cursor_end, int):
            paths_A = paths_A[:config.cursor_end]
        paths_C = [p.replace('frameA', 'frameC') for p in paths_A]
        paths_mag = [p.replace('frameA', 'amplified') for p in paths_A]
    else:
        paths_A, paths_C, paths_skip_intermediate, paths_skip = [], [], [], []
    if 'test' not in mode:
        path_vids = os.path.join(config.dir_train, 'train_vid_frames')
        dirs_vid = [os.path.join(path_vids, p, 'frameA') for p in config.videos_train]
        for dir_vid in dirs_vid[:config.video_num]:
            vid_frames = [
                os.path.join(dir_vid, p) for p in sorted(
                    os.listdir(dir_vid), key=lambda x: int(x.split('.')[0])
                )]
            if config.skip < 0:
                lst = [p.replace('frameA', 'frameC') for p in vid_frames]
                for idx, _ in enumerate(lst):
                    skip_rand = np.random.randint(min(-config.skip, 2), -config.skip+1)
                    idx_skip = min(idx + skip_rand, len(lst) - 1)
                    paths_skip.append(lst[idx_skip])
                    paths_skip_intermediate.append(lst[idx_skip//2])
            paths_A += vid_frames
        paths_C = [p.replace('frameA', 'frameC') for p in paths_A]
    paths_B = [p.replace('frameC', 'frameB') for p in paths_C]
    return paths_A, paths_B, paths_C, paths_skip, paths_skip_intermediate, paths_mag


class DataGen():
    def __init__(self, paths, config, mode):
        self.is_test = 'test' in mode
        self.anchor = 0
        self.paths = paths
        self.batch_size = config.batch_size_test if self.is_test else config.batch_size
        self.data_len = len(paths)
        self.load_all = False
        self.data = []
        self.preproc = config.preproc
        self.coco_amp_lst = config.coco_amp_lst

    def gen(self, anchor=None):
        batch_A = []
        batch_C = []
        batch_M = []
        batch_B = []
        batch_amp = []
        batch_ori = []
        if anchor is None:
            anchor = self.anchor

        for _ in range(self.batch_size):
            unit_A = load_unit(self.paths[anchor], preproc=self.preproc)
            unit_C = load_unit(self.paths[anchor].replace('frameA', 'frameC'), preproc=self.preproc)
            unit_M = load_unit(self.paths[anchor].replace('frameA', 'amplified'), preproc=self.preproc)
            unit_B = load_unit(self.paths[anchor].replace('frameA', 'frameB'), preproc=self.preproc)
            unit_amp = self.coco_amp_lst[anchor]
            batch_A.append(unit_A)
            batch_C.append(unit_C)
            batch_M.append(unit_M)
            batch_B.append(unit_B)
            batch_amp.append(unit_amp)

            self.anchor = (self.anchor + 1) % self.data_len

        batch_A = numpy2cuda(batch_A)
        batch_C = numpy2cuda(batch_C)
        batch_M = numpy2cuda(batch_M)
        batch_B = numpy2cuda(batch_B)
        batch_amp = numpy2cuda(batch_amp)
        for _ in range(4 - len(batch_amp.shape)):
            batch_amp = batch_amp.unsqueeze(-1)
        return batch_A, batch_B, batch_C, batch_M, batch_amp

    def gen_test(self, anchor=None):
        batch_A = []
        batch_C = []
        if anchor is None:
            anchor = self.anchor

        for _ in range(self.batch_size):
            unit_A = load_unit(self.paths[anchor], preproc=self.preproc)
            unit_C = load_unit(self.paths[anchor].replace('frameA', 'frameC'), preproc=self.preproc)
            batch_A.append(unit_A)
            batch_C.append(unit_C)

            self.anchor = (self.anchor + 1) % self.data_len

        batch_A = numpy2cuda(batch_A)
        batch_C = numpy2cuda(batch_C)
        return batch_A, batch_C


def get_gen_ABC(config, mode='train'):
    paths_A = get_paths_ABC(config, mode)[0]
    gen_train_A = DataGen(paths_A, config, mode)
    return gen_train_A


def cuda2numpy(tensor):
    array = tensor.detach().cpu().squeeze().numpy()
    return array


def numpy2cuda(array):
    tensor = torch.from_numpy(np.asarray(array)).float().cuda()
    return tensor


def resize2d(img, size):
    with torch.no_grad():
        img_resized = (F.adaptive_avg_pool2d(Variable(img, volatile=True), size)).data
    return img_resized