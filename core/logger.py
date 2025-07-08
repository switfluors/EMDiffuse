import os
from PIL import Image
import importlib
from datetime import datetime
import logging
import pandas as pd

import core.util as Util

class InfoLogger():
    """
    use logging to record log, only work on GPU 0 by judging global_rank
    """
    def __init__(self, opt):
        self.opt = opt
        self.rank = opt['global_rank']
        self.phase = opt['phase']

        self.setup_logger(None, opt['path']['experiments_root'], opt['phase'], level=logging.INFO, screen=False)
        self.logger = logging.getLogger(opt['phase'])
        self.infologger_ftns = {'info', 'warning', 'debug'}

    def __getattr__(self, name):
        if self.rank != 0: # info only print on GPU 0.
            def wrapper(info, *args, **kwargs):
                pass
            return wrapper
        if name in self.infologger_ftns:
            print_info = getattr(self.logger, name, None)
            def wrapper(info, *args, **kwargs):
                print_info(info, *args, **kwargs)
            return wrapper
    
    @staticmethod
    def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False):
        """ set up logger """
        l = logging.getLogger(logger_name)
        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
        log_file = os.path.join(root, '{}.log'.format(phase))
        fh = logging.FileHandler(log_file, mode='a+')
        fh.setFormatter(formatter)
        l.setLevel(level)
        l.addHandler(fh)
        if screen:
            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
            l.addHandler(sh)

class VisualWriter():
    """ 
    use tensorboard to record visuals, support 'add_scalar', 'add_scalars', 'add_image', 'add_images', etc. funtion.
    Also integrated with save results function.
    """
    def __init__(self, opt, logger):
        log_dir = opt['path']['tb_logger']
        self.result_dir = opt['path']['results']
        enabled = opt['train']['tensorboard']
        self.rank = opt['global_rank']
        self.task = opt['task']

        self.writer = None
        self.selected_module = ""

        if enabled and self.rank==0:
            log_dir = str(log_dir)

            # Retrieve vizualization writer.
            succeeded = False
            for module in ["tensorboardX", "torch.utils.tensorboard"]:
                try:
                    self.writer = importlib.import_module(module).SummaryWriter(log_dir)
                    succeeded = True
                    break
                except ImportError:
                    succeeded = False
                self.selected_module = module

            if not succeeded:
                message = "Warning: visualization (Tensorboard) is configured to use, but currently not installed on " \
                    "this machine. Please install TensorboardX with 'pip install tensorboardx', upgrade PyTorch to " \
                    "version >= 1.1 to use 'torch.utils.tensorboard' or turn off the option in the 'config.json' file."
                logger.warning(message)

        self.epoch = 0
        self.iter = 0
        self.phase = ''

        self.tb_writer_ftns = {
            'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio',
            'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding'
        }
        self.tag_mode_exceptions = {'add_histogram', 'add_embedding'}
        self.custom_ftns = {'close'}
        self.timer = datetime.now()

    def set_iter(self, epoch, iter, phase='train'):
        self.phase = phase
        self.epoch = epoch
        self.iter = iter

    def save_images(self, results, norm=True, percent=False):
        result_path = os.path.join(self.result_dir, self.phase)
        os.makedirs(result_path, exist_ok=True)
        result_path = os.path.join(result_path, str(self.epoch))
        os.makedirs(result_path, exist_ok=True)

        from PIL import Image
        import numpy as np
        import torch
        import matplotlib.pyplot as plt

        def rescale_to_uint16(img):
            img = img.astype(np.float32)
            img_min = img.min()
            img_max = img.max()
            if img_max - img_min < 1e-8:
                return np.zeros_like(img, dtype=np.uint16)
            img = (img - img_min) / (img_max - img_min)
            return (img * 65535).astype(np.uint16)

        try:
            names = results['name']
            outputs = results['result']

            for i in range(len(names)):
                img = outputs[i]
                if isinstance(img, torch.Tensor):
                    img = img.detach().cpu().numpy()

                if img.ndim == 3 and img.shape[0] == 1:
                    img = img.squeeze(0)

                img = rescale_to_uint16(img)

                # 🔬 Optional debug visualization
                # plt.figure(figsize=(6, 3))
                # plt.imshow(img, cmap='gray')
                # plt.title(f"Saved: {names[i]}")
                # plt.axis('off')
                # plt.tight_layout()
                # plt.show()

                Image.fromarray(img).save(os.path.join(result_path, names[i]))

        except Exception as e:
            raise NotImplementedError(
                'You must specify the context of name and result in save_current_results functions of model.'
            ) from e

    # def save_images(self, results, norm=True, percent=False):
    #     result_path = os.path.join(self.result_dir, self.phase)
    #     os.makedirs(result_path, exist_ok=True)
    #     result_path = os.path.join(result_path, str(self.epoch))
    #     os.makedirs(result_path, exist_ok=True)
    #
    #     from PIL import Image
    #     import numpy as np
    #     import torch
    #
    #     def denormalize_to_uint16(tensor_img):
    #         # Expecting input in range [-1, 1]
    #         img = tensor_img.astype(np.float32)
    #         img = (img + 1.0) / 2.0  # Scale [-1, 1] to [0, 1]
    #         img = np.clip(img, 0, 1)
    #         return (img * 65535).astype(np.uint16)
    #
    #     try:
    #         names = results['name']
    #         outputs = results['result']  # raw tensors before postprocess
    #
    #         for i in range(len(names)):
    #             img = outputs[i]
    #             if isinstance(img, torch.Tensor):
    #                 img = img.detach().cpu().numpy()
    #
    #             if img.ndim == 3 and img.shape[0] == 1:
    #                 img = img.squeeze(0)  # shape: (H, W)
    #
    #             img = denormalize_to_uint16(img)  # uint16 image in [0, 65535]
    #
    #             # ✅ No transpose needed — keep as (128, 16)
    #             Image.fromarray(img).save(os.path.join(result_path, names[i]))
    #
    #     except Exception as e:
    #         raise NotImplementedError(
    #             'You must specify the context of name and result in save_current_results functions of model.'
    #         ) from e


    # def save_images(self, results, norm=True, percent=False):
    #     result_path = os.path.join(self.result_dir, self.phase)
    #     os.makedirs(result_path, exist_ok=True)
    #     result_path = os.path.join(result_path, str(self.epoch))
    #     os.makedirs(result_path, exist_ok=True)
    #
    #     from PIL import Image
    #     import numpy as np
    #     import torch
    #
    #     def denormalize_to_uint16(tensor_img):
    #         # Expecting input in range [-1, 1]
    #         img = tensor_img.astype(np.float32)
    #         img = (img + 1.0) / 2.0  # Scale [-1, 1] to [0, 1]
    #         img = np.clip(img, 0, 1)
    #         return (img * 65535).astype(np.uint16)
    #
    #     try:
    #         names = results['name']
    #         outputs = results['result']  # raw tensors before postprocess
    #
    #         for i in range(len(names)):
    #             img = outputs[i]
    #             if isinstance(img, torch.Tensor):
    #                 img = img.detach().cpu().numpy()
    #
    #             if img.ndim == 3 and img.shape[0] == 1:
    #                 img = img.squeeze(0)  # shape: (H, W)
    #
    #             img = denormalize_to_uint16(img)  # Now uint16 in [0, 65535]
    #
    #             # 🔁 Transpose explicitly to (128, 16) for saving
    #             img = img.T  # if shape is (H=128, W=16), this swaps to (16, 128) to fix saved display
    #
    #             Image.fromarray(img).save(os.path.join(result_path, names[i]))
    #
    #     except Exception as e:
    #         raise NotImplementedError(
    #             'You must specify the context of name and result in save_current_results functions of model.'
    #         ) from e
    #
    #     except Exception as e:
    #         raise NotImplementedError(
    #             'You must specify the context of name and result in save_current_results functions of model.'
    #         ) from e

    def close(self):
        self.writer.close()
        print('Close the Tensorboard SummaryWriter.')

        
    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return a blank function handle that does nothing
        """
        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)
            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    # add phase(train/valid) tag
                    if name not in self.tag_mode_exceptions:
                        tag = '{}/{}'.format(self.phase, tag)
                    add_data(tag, data, self.iter, *args, **kwargs)
            return wrapper
        else:
            # default action for returning methods defined in this class, set_step() for instance.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError("type object '{}' has no attribute '{}'".format(self.selected_module, name))
            return attr


class LogTracker:
    """
    record training numerical indicators.
    """
    def __init__(self, *keys, phase='train'):
        self.phase = phase
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return {'{}/{}'.format(self.phase, k):v for k, v in dict(self._data.average).items()}
