import argparse
import os
import numpy as np
from PIL import Image

import torch

from model.dc_sam import DC_SAM
from model.dc_sam_prior import DC_SAM_Prior
from common.logger_batch import AverageMeter
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset
from SAM_plugin import SAM_plugin
from tqdm import tqdm


def flatten_support_names(support_names):
    out = []
    for names in support_names:
        if isinstance(names, (list, tuple)):
            out.extend(names)
        else:
            out.append(names)
    return out


def save_pred_masks(pred_mask, query_names, class_names, save_dir):
    if save_dir is None or save_dir == "":
        return

    if pred_mask.dim() == 4:
        pred_mask = pred_mask.squeeze(1)

    pred_mask = pred_mask.detach().cpu().numpy().astype(np.uint8) * 255

    for i, (query_path, class_name) in enumerate(zip(query_names, class_names)):
        class_save_dir = os.path.join(save_dir, class_name)
        os.makedirs(class_save_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(query_path))[0] + ".png"
        save_path = os.path.join(class_save_dir, base_name)
        Image.fromarray(pred_mask[i]).save(save_path)


def evaluate(args, model, sam_model, dataloader):
    utils.fix_randseed(args.seed)
    model.eval()
    sam_model.eval()
    average_meter = AverageMeter()

    for batch in tqdm(dataloader):
        batch = utils.to_cuda(batch)

        with torch.no_grad():
            bs = batch['query_img'].size(0)

            support_imgs = batch['support_imgs']
            support_masks = batch['support_masks']

            # support_imgs: [B, shot, 3, H, W]
            # support_masks: [B, shot, H, W]
            if support_imgs.dim() == 5:
                nshot = support_imgs.size(1)
                support_imgs = support_imgs.reshape(bs * nshot, *support_imgs.shape[2:])
                support_masks = support_masks.reshape(bs * nshot, *support_masks.shape[2:])
            else:
                nshot = 1

            supp_names = flatten_support_names(batch['support_names'])

            query_sam = sam_model.get_feat_from_np(batch['query_img'], batch['query_name'])
            support_sam = sam_model.get_feat_from_np(support_imgs, supp_names)

            protos, _, q_feat, s_feat = model(
                (batch['query_img'], support_imgs, support_masks, query_sam, support_sam),
                stage=1
            )

            _, pre_mask = sam_model(batch['query_img'], batch['query_name'], protos)
            protos = model((q_feat, pre_mask, s_feat, protos), stage=2)

            low_masks, pred_mask = sam_model(batch['query_img'], batch['query_name'], protos)
            logit_mask = low_masks

            pred_mask = (torch.sigmoid(logit_mask) > 0.5).float()

            loss = model.compute_objective(logit_mask, batch['query_mask'])

            if args.save_dir is not None and args.save_dir != "":
                save_pred_masks(pred_mask, batch['query_name'], batch['class_name'], args.save_dir)

            area_inter, area_union = Evaluator.classify_prediction(pred_mask.squeeze(1), batch)
            average_meter.update(area_inter, area_union, batch['class_name'], loss.detach())

    average_meter.write_result('Test', 0)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou, _ = average_meter.compute_iou()

    print(f'Loss: {avg_loss:.5f}, mIoU: {miou:.2f}, FB-IoU: {fb_iou:.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DC-SAM FSS Batch Test')

    # === Dataset ===
    parser.add_argument('--datapath', type=str, default='/path/to/dataset')
    parser.add_argument('--dataset-name', type=str, default='ISIC')
    parser.add_argument('--img-size', type=int, default=512)

    # 保留 nshot；内部自动转 support_indices
    parser.add_argument('--nshot', type=int, default=1)
    parser.add_argument('--support-indices', type=str, default=None, help='Optional, e.g. "[0]" or "[0,1]"')

    # === Test ===
    parser.add_argument('--bsz', type=int, default=2)
    parser.add_argument('--nworker', type=int, default=4)
    parser.add_argument('--seed', type=int, default=321)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--save-dir', type=str, default='', help='Save predicted masks by class')

    # === Model ===
    parser.add_argument('--sam_version', type=int, default=2, choices=[1, 2])
    parser.add_argument('--num_query', type=int, default=25)
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['vgg16', 'resnet50', 'resnet101'])
    parser.add_argument('--prior', action='store_true', help='Use Prior')
    parser.add_argument('--sam1-ckpt', type=str, default='/datanas01/nas01/Student-home/2025D_ShiGuangze/Code/GPO_1/GPO/segmenter/checkpoint/sam_vit_h_4b8939.pth')
    parser.add_argument('--sam2-config', type=str, default='configs/sam2.1/sam2.1_hiera_l.yaml')
    parser.add_argument('--sam2-ckpt', type=str, default='/datanas01/nas01/Student-home/2025D_ShiGuangze/Code/IFP/Segment_Anything2/checkpoints/sam2.1_hiera_large.pt')

    args = parser.parse_args()

    if args.support_indices is None:
        args.support_indices = list(range(args.nshot))
    else:
        args.support_indices = eval(args.support_indices)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Evaluator.initialize()
    utils.fix_randseed(args.seed)

    if args.prior:
        model = DC_SAM_Prior(args, args.backbone, False)
    else:
        model = DC_SAM(args, args.backbone, False)

    sam_model = SAM_plugin(args)

    state_dict = torch.load(args.ckpt, map_location='cpu')

    # 兼容 DataParallel / 非 DataParallel
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)

    sam_model.to(device)
    model.to(device)

    FSSDataset.initialize(img_size=args.img_size, datapath=args.datapath)
    dataloader_val = FSSDataset.build_dataloader_fss(
        dataset_name=args.dataset_name,
        bsz=args.bsz,
        support_indices=args.support_indices,
        resize=(args.img_size, args.img_size),
        nworker=args.nworker
    )

    evaluate(args, model, sam_model, dataloader_val)