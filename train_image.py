import argparse
import torch
import torch.optim as optim

from model.dc_sam import DC_SAM
from model.dc_sam_prior import DC_SAM_Prior
from common.logger_batch import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset
from SAM_plugin import SAM_plugin


def flatten_support_names(support_names):
    out = []
    for names in support_names:
        if isinstance(names, (list, tuple)):
            out.extend(names)
        else:
            out.append(names)
    return out


def train_one_epoch(args, epoch, model, sam_model, dataloader, optimizer, scheduler):
    utils.fix_randseed(args.seed + epoch)
    model.train_mode()
    average_meter = AverageMeter()

    for batch_idx, batch in enumerate(dataloader):
        batch = utils.to_cuda(batch)

        with torch.set_grad_enabled(True):
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

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            inter, union = Evaluator.classify_prediction(pred_mask.squeeze(1), batch)
            average_meter.update(inter, union, batch['class_name'], loss.detach())
            average_meter.write_process(batch_idx, len(dataloader), epoch, write_batch_idx=20)

    average_meter.write_result('Train', epoch)
    loss_avg = utils.mean(average_meter.loss_buf)
    miou, fb_iou, _ = average_meter.compute_iou()
    return loss_avg, miou, fb_iou


def main():
    parser = argparse.ArgumentParser(description='Image-to-image In-context Training (Batch Version)')

    # === Dataset ===
    parser.add_argument('--datapath', type=str, default='/path/to/dataset')
    parser.add_argument('--dataset-name', type=str, default='FSS')
    parser.add_argument('--img-size', type=int, default=512)

    # 保留 nshot；内部自动转 support_indices
    parser.add_argument('--nshot', type=int, default=1)
    parser.add_argument('--support-indices', type=str, default=None, help='Optional, e.g. "[0]" or "[0,1]"')

    # === Training ===
    parser.add_argument('--log-root', type=str, default='output/logs_image')
    parser.add_argument('--bsz', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-6)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--nworker', type=int, default=4)
    parser.add_argument('--seed', type=int, default=321)

    # === Model ===
    parser.add_argument('--sam_version', type=int, default=2, choices=[1, 2])
    parser.add_argument('--num_query', type=int, default=25)
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['vgg16', 'resnet50', 'resnet101', 'swinb', 'dinov2b'])
    parser.add_argument('--prior', action='store_true', help='Only use prior in the model')
    parser.add_argument('--sam1-ckpt', type=str, default='/datanas01/nas01/Student-home/2025D_ShiGuangze/Code/GPO_1/GPO/segmenter/checkpoint/sam_vit_h_4b8939.pth')
    parser.add_argument('--sam2-config', type=str, default='configs/sam2.1/sam2.1_hiera_l.yaml')
    parser.add_argument('--sam2-ckpt', type=str, default='/datanas01/nas01/Student-home/2025D_ShiGuangze/Code/IFP/Segment_Anything2/checkpoints/sam2.1_hiera_large.pt')

    args = parser.parse_args()

    if args.support_indices is None:
        args.support_indices = list(range(args.nshot))
    else:
        args.support_indices = eval(args.support_indices)

    # === Setup ===
    Logger.initialize(args, root=args.log_root)
    Evaluator.initialize()
    utils.fix_randseed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # === Model ===
    if args.prior:
        model = DC_SAM_Prior(args, args.backbone, False).to(device)
    else:
        model = DC_SAM(args, args.backbone, False).to(device)

    Logger.log_params(model)

    # === Keep SAM1 / SAM2 parameter ===
    sam_model = SAM_plugin(args).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )

    # === Dataset ===
    FSSDataset.initialize(img_size=args.img_size, datapath=args.datapath)
    dataloader_trn = FSSDataset.build_dataloader_fss(
        dataset_name=args.dataset_name,
        bsz=args.bsz,
        support_indices=args.support_indices,
        resize=(args.img_size, args.img_size),
        nworker=args.nworker
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * len(dataloader_trn)
    )

    # === Training loop ===
    best_trn_miou = float('-inf')
    for epoch in range(args.epochs):
        trn_loss, trn_miou, trn_fb_iou = train_one_epoch(
            args, epoch, model, sam_model, dataloader_trn, optimizer, scheduler
        )

        Logger.tbd_writer.add_scalars('loss', {'train': trn_loss}, epoch)
        Logger.tbd_writer.add_scalars('miou', {'train': trn_miou}, epoch)
        Logger.tbd_writer.add_scalars('fb_iou', {'train': trn_fb_iou}, epoch)
        Logger.tbd_writer.flush()

        # 按 train mIoU 保存当前最优
        if trn_miou > best_trn_miou:
            best_trn_miou = trn_miou
            Logger.save_model_miou(model, epoch, trn_miou)

    Logger.tbd_writer.close()
    Logger.info('==================== Finished Training ====================')


if __name__ == '__main__':
    main()