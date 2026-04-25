r""" Logging during training/testing """
import datetime
import logging
import os

from tensorboardX import SummaryWriter
import torch


class AverageMeter:
    r""" Stores evaluation results per class_name (str) """
    def __init__(self, dataset=None):
        # dataset 参数保留但不使用
        self.intersections = {}  # dict[str -> Tensor([2])]
        self.unions = {}         # dict[str -> Tensor([2])]
        self.loss_buf = []

    def update(self, inter_b, union_b, class_name, loss):
        """
        inter_b: Tensor([2, B]) or Tensor([2])
        union_b: Tensor([2, B]) or Tensor([2])
        class_name: str / list[str] / tuple[str]
        """
        if isinstance(class_name, str):
            class_name = [class_name]
        elif isinstance(class_name, tuple):
            class_name = list(class_name)

        if inter_b.dim() == 1:
            inter_b = inter_b.unsqueeze(1)
        if union_b.dim() == 1:
            union_b = union_b.unsqueeze(1)

        assert inter_b.shape[1] == union_b.shape[1] == len(class_name), \
            f"Batch size mismatch: inter={inter_b.shape}, union={union_b.shape}, class_name={len(class_name)}"

        for i, cls_name in enumerate(class_name):
            if cls_name not in self.intersections:
                self.intersections[cls_name] = torch.zeros_like(inter_b[:, i])
                self.unions[cls_name] = torch.zeros_like(union_b[:, i])

            self.intersections[cls_name] += inter_b[:, i]
            self.unions[cls_name] += union_b[:, i]

        if loss is None:
            loss = torch.tensor(0.0, device=inter_b.device)
        self.loss_buf.append(loss.detach())

    def compute_iou(self):
        ious = []
        fb_ious = []
        class_iou_map = {}

        for class_name in self.intersections:
            inter = self.intersections[class_name]
            union = self.unions[class_name]
            iou = inter / (union + 1e-10)
            ious.append(iou[1])  # foreground IoU
            fb_ious.append(iou.mean())
            class_iou_map[class_name] = iou[1].item() * 100

        mean_iou = torch.stack(ious).mean() * 100
        mean_fb_iou = torch.stack(fb_ious).mean() * 100

        return mean_iou, mean_fb_iou, class_iou_map  # dict[class_name → iou]

    def write_result(self, split, epoch):
        miou, fb_iou, class_ious = self.compute_iou()
        loss_buf = torch.stack(self.loss_buf)

        msg = f'\n*** {split} [@Epoch {epoch:02d}] '
        msg += f'Avg L: {loss_buf.mean():6.5f}  '
        msg += f'mIoU: {miou:5.2f}   '
        msg += f'FB-IoU: {fb_iou:5.2f}   '

        for cls_name, iou_val in class_ious.items():
            msg += f'|  {cls_name}: {iou_val:5.2f}   '

        msg += '***\n'
        Logger.info(msg)

    def write_process(self, batch_idx, datalen, epoch, write_batch_idx=20):
        if batch_idx % write_batch_idx == 0:
            msg = f'[Epoch: {epoch:02d}] ' if epoch != -1 else ''
            msg += f'[Batch: {batch_idx+1:04d}/{datalen:04d}] '
            miou, fb_iou, class_ious = self.compute_iou()

            if epoch != -1:
                loss_buf = torch.stack(self.loss_buf)
                msg += f'L: {loss_buf[-1]:6.5f}  '
                msg += f'Avg L: {loss_buf.mean():6.5f}  '

            msg += f'mIoU: {miou:5.2f}  |  FB-IoU: {fb_iou:5.2f}'
            for cls_name, iou_val in class_ious.items():
                msg += f' |  {cls_name}: {iou_val:5.2f}   '

            Logger.info(msg)



class Logger:
    r""" Writes evaluation results of testing """
    @classmethod
    def initialize(cls, args, root='logs'):
        logtime = datetime.datetime.now().strftime('%m%d_%H%M%S')
        logname = '_TEST_' + logtime

        cls.logpath = os.path.join(root, logname)
        os.makedirs(cls.logpath, exist_ok=True)

        cls.logfile = os.path.join(cls.logpath, 'log.txt')
        logging.basicConfig(
            filename=cls.logfile,
            filemode='w',
            level=logging.INFO,
            format='%(message)s',
            datefmt='%m-%d %H:%M:%S'
        )

        # Console logging
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

        # Tensorboard writer
        cls.tbd_writer = SummaryWriter(os.path.join(cls.logpath, 'tbd', 'runs'))

        # Print args to log
        logging.info('\n:=========== Few-shot Seg. with Matcher ===========')
        for arg_key in vars(args):
            logging.info(f'| {arg_key:20s}: {str(getattr(args, arg_key))}')
        logging.info(':================================================\n')

    @classmethod
    def info(cls, msg):
        logging.info(msg)

    @classmethod
    def save_model_miou(cls, model, epoch, val_miou):
        save_path = os.path.join(cls.logpath, 'best_model.pt')
        torch.save(model.state_dict(), save_path)
        cls.info(f'Model saved @epoch {epoch} with val mIoU: {val_miou:.2f}\n')

    @classmethod
    def log_params(cls, model):
        backbone_param = 0
        learner_param = 0
        for k, v in model.state_dict().items():
            n_param = v.numel()
            if k.startswith('backbone') and not any(x in k for x in ['classifier', 'fc']):
                backbone_param += n_param
            else:
                learner_param += n_param
        total = backbone_param + learner_param
        cls.info(f'Backbone # param.: {backbone_param}')
        cls.info(f'Learnable # param.: {learner_param}')
        cls.info(f'Total # param.: {total}')