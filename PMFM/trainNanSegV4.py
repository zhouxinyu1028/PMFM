import os
import sys


if sys.platform != 'win32':
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import wandb
from medpy import metric
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.nanseg.train_dataset import TrainDataset
from datasets.nanseg.val_dataset import ValDataset
from loss import metrics
from loss.loss import BinaryFocalLoss, DiceLoss2D
from model.builder_model import build_model
from optimizer.builder_lr_scheduler import build_lr_scheduler
from optimizer.builder_optimizer import build_optimizer
from utils import weights_init
from utils.image_3d_to_2d_show import nanseg_show_single
from utils.load_yaml_file import read_yaml
from utils.logger_config import logger_config


dir_yaml = Path(__file__).resolve().parent / 'config' / 'nanseg.yaml'
yaml_data = read_yaml(dir_yaml)



def require_finite(value, context):
    """Stop at the first invalid tensor; never repair NaNs by hiding them."""
    if torch.is_tensor(value):
        if not torch.isfinite(value).all().item():
            raise FloatingPointError(f"Non-finite tensor at {context}; training stopped before checkpoint save.")
    elif isinstance(value, dict):
        for key, item in value.items():
            require_finite(item, f"{context}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            require_finite(item, f"{context}[{index}]")


def guarded_backward_step(loss, models, optimizers, context, max_norm=1.0):
    require_finite(loss, context + ".loss")
    # Enable forward/backward anomaly traces via PMFM_DETECT_ANOMALY=1.
    loss.backward()
    parameters = [p for model in models for p in model.parameters() if p.requires_grad]
    for index, parameter in enumerate(parameters):
        if parameter.grad is not None:
            require_finite(parameter.grad, f"{context}.gradient[{index}]")
    torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True)
    for optimizer in optimizers:
        optimizer.step()
    for index, model in enumerate(models):
        require_finite(model.state_dict(), f"{context}.updated_model[{index}]")


def as_bool(value):
    """yaml 里若写成字符串 "False"，bool("False") 会得到 True，这里统一解析。"""
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'y')
    return bool(value)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def update_image_dict(best_vis_dict, dice, ct, mr, label, ct_pred, mr_pred, fusion_pred, label_id):

    if (torch.max(label).item() != 1) or (torch.max(fusion_pred).item() != 1):
        return
    if best_vis_dict and dice <= best_vis_dict['dice']:
        return
    best_vis_dict.update({
        'dice': float(dice),
        'label_id': int(label_id),
        'ct_original': ct.detach().cpu(),
        'mr_original': mr.detach().cpu(),
        'label': label.detach().cpu(),
        'ct_pred': ct_pred.detach().cpu(),
        'mr_pred': mr_pred.detach().cpu(),
        'fusion_pred': fusion_pred.detach().cpu(),
    })


def _finite(value):

    return float(value) if np.isfinite(value) else float('nan')


def print_dice(evaluation, name='Train', is_BuildIn=False):

    labels = yaml_data['labels']
    result = OrderedDict()
    if not is_BuildIn:

        for i in range(len(labels)):
            result[name + '_dice_' + labels[i]] = float(evaluation.avg[i].item())
        result[name + '_dice_num_present'] = int((evaluation.count > 0).sum().item())
    else:
        for i in range(len(labels)):
            # dice
            result[name + '_dice(Build-In)_' + labels[i]] = _finite(evaluation.dice_avg[i].item())
            # jc
            result[name + '_jc(Build-In)_' + labels[i]] = _finite(evaluation.jc_avg[i].item())
            # hd95
            result[name + '_hd95(Build-In)_' + labels[i]] = _finite(evaluation.hd_avg[i].item())
            # asd
            result[name + '_asd(Build-In)_' + labels[i]] = _finite(evaluation.asd_avg[i].item())
    return result


def preprocess_pred(ct_pred, mr_pred, fusion_pred, target, label_id):
 
    index = int(label_id)
    return (ct_pred[:, index:index + 1], mr_pred[:, index:index + 1],
            fusion_pred[:, index:index + 1], target)


# 内置评价指标
def Built_in_evaluation_metrics(pred, gt, compute_hd_asd=True):
    one_hot_pred = (torch.sigmoid(pred) > 0.5).float()
    one_hot_pred = one_hot_pred.detach().cpu().numpy()
    gt = gt.detach().cpu().numpy()

    dice = metric.binary.dc(one_hot_pred, gt)
    jc = metric.binary.jc(one_hot_pred, gt)
    if not compute_hd_asd:
        # HD95/ASD 用的是 medpy 的 C 距离变换，占验证耗时的大头；跳过时返回 nan 让 wandb 忽略该曲线
        return dice, jc, float('nan'), float('nan')
    if np.any(one_hot_pred):
        hd = metric.binary.hd95(one_hot_pred, gt)
        asd = metric.binary.asd(one_hot_pred, gt)
    else:
        hd = float('inf')
        asd = float('inf')
    return dice, jc, hd, asd


def val(UNet_model, PFF_model, val_loader, loss_func, focal_loss, device, tqdm_description, alpha, experiment,
        compute_hd_asd=True, empty_cache_every=0):
    num_classes = int(yaml_data['model']['backbone']['num_classes'])  # 类别数
    is_print_image = as_bool(yaml_data['other']['is_print_image'])

    UNet_model.eval()
    PFF_model.eval()

    val_loss = metrics.LossAverage()
    val_ct_loss = metrics.LossAverage()
    val_mr_loss = metrics.LossAverage()
    val_fusion_loss = metrics.LossAverage()
    val_ct_dice = metrics.DiceAverage2D_single_label(num_classes)
    val_mr_dice = metrics.DiceAverage2D_single_label(num_classes)
    val_fusion_dice = metrics.DiceAverage2D_single_label(num_classes)

    # 内置评价指标
    val_build_in_evaluation = metrics.BuiltInEvaluationMetrics(num_classes)

    best_vis_dict = {}

    with torch.no_grad():
        with tqdm(total=len(val_loader), desc=tqdm_description, unit='batch') as pbar:
            for idx, (ct, mr, label, ori_ct, ori_mr, label_id) in enumerate(val_loader):
                ct, mr = ct.float().to(device), mr.float().to(device)
                label = label.float().to(device)
                if label.dim() == 3:  # 数据集给的是 (B, H, W) 时补通道维，统一成 (B, 1, H, W)
                    label = label.unsqueeze(1)

                # PFF 实际只用到 ct_pred / mr_pred，前四个参数目前未参与计算
                target_ct_pred, point_ct, ct_last, ct_pred = UNet_model(ct)
                target_mr_pred, point_mr, mr_last, mr_pred = UNet_model(mr)
                fusion_target = PFF_model(point_ct, point_mr, ct_last, mr_last, ct_pred, mr_pred)

                new_ct_pred, new_mr_pred, new_fusion_pred, new_label = preprocess_pred(
                    target_ct_pred, target_mr_pred, fusion_target, label, label_id)

                ct_prob = torch.sigmoid(new_ct_pred)
                mr_prob = torch.sigmoid(new_mr_pred)
                fusion_prob = torch.sigmoid(new_fusion_pred)

                ct_loss = loss_func(ct_prob, new_label) + focal_loss(ct_prob, new_label)
                mr_loss = loss_func(mr_prob, new_label) + focal_loss(mr_prob, new_label)
                fusion_loss = loss_func(fusion_prob, new_label) + focal_loss(fusion_prob, new_label)
                total_loss = ct_loss * alpha[0] + mr_loss * alpha[1] + fusion_loss * alpha[2]

                val_loss.update(total_loss.item(), ct.size(0))
                val_ct_loss.update(ct_loss.item(), ct.size(0))
                val_mr_loss.update(mr_loss.item(), ct.size(0))
                val_fusion_loss.update(fusion_loss.item(), ct.size(0))

                target_ct_pred_one_hot = (ct_prob > 0.5).float()
                target_mr_pred_one_hot = (mr_prob > 0.5).float()
                fusion_target_one_hot = (fusion_prob > 0.5).float()
                val_ct_dice.update(target_ct_pred_one_hot, new_label, label_id)
                val_mr_dice.update(target_mr_pred_one_hot, new_label, label_id)
                val_fusion_dice.update(fusion_target_one_hot, new_label, label_id)

                dice, jc, hd, asd = Built_in_evaluation_metrics(new_fusion_pred, new_label, compute_hd_asd)
                val_build_in_evaluation.update(dice, jc, hd, asd, label_id)

                # 用"当前样本自己的 dice"挑可视化帧，而不是跨器官的累计平均
                # update() 已经把本样本 dice 写进 .value，直接读可避免再做一次 GPU 计算 + 同步
                sample_dice = float(val_fusion_dice.value[int(label_id)])
                update_image_dict(best_vis_dict, sample_dice, ori_ct, ori_mr, label, target_ct_pred_one_hot,
                                  target_mr_pred_one_hot, fusion_target_one_hot, label_id)
                if empty_cache_every and (idx + 1) % empty_cache_every == 0:
                    torch.cuda.empty_cache()
                pbar.update()
                pbar.set_postfix(**{'loss': str(val_loss.avg)})

    if is_print_image and best_vis_dict:
        nanseg_show_single(best_vis_dict['ct_original'], best_vis_dict['mr_original'], best_vis_dict['label'],
                           best_vis_dict['ct_pred'], best_vis_dict['mr_pred'], best_vis_dict['fusion_pred'],
                           experiment, yaml_data['labels'], best_vis_dict['label_id'], name="Val")

    val_log = OrderedDict({
        'Val_avg_loss': float(val_loss.avg),
        'Val_ct_loss': float(val_ct_loss.avg),
        'Val_mr_loss': float(val_mr_loss.avg),
        'Val_fusion_loss': float(val_fusion_loss.avg),
        'Val_fusion_avg_dice': float(val_fusion_dice.total_avg),
        'Val_ct_avg_dice': float(val_ct_dice.total_avg),
        'Val_mr_avg_dice': float(val_mr_dice.total_avg),
        'Val_ct_mr_avg_dice': float((val_mr_dice.total_avg + val_ct_dice.total_avg) / 2),
    })
    val_log.update(print_dice(val_fusion_dice, name='Val'))
    val_log.update(print_dice(val_build_in_evaluation, name='Val', is_BuildIn=True))
    return val_log


def train(UNet_model, PFF_model, train_loader, optimizer_unet, optimizer_pff, loss_func, focal_loss, device,
          tqdm_description, alpha, experiment, empty_cache_every=0):
    num_classes = int(yaml_data['model']['backbone']['num_classes'])  # 类别数
    is_print_image = as_bool(yaml_data['other']['is_print_image'])

    UNet_model.train()  # 将模型调整为训练模式
    PFF_model.train()

    train_loss = metrics.LossAverage()
    train_ct_loss = metrics.LossAverage()
    train_mr_loss = metrics.LossAverage()
    train_fusion_loss = metrics.LossAverage()
    train_dice = metrics.DiceAverage2D_single_label(num_classes)

    best_vis_dict = {}

    with tqdm(total=len(train_loader), desc=tqdm_description, unit='batch') as pbar:
        for idx, (ct, mr, label, ori_ct, ori_mr, label_id) in enumerate(train_loader):
            ct, mr = ct.float().to(device), mr.float().to(device)
            label = label.float().to(device)
            if label.dim() == 3:  # 数据集给的是 (B, H, W) 时补通道维，统一成 (B, 1, H, W)
                label = label.unsqueeze(1)

            batch_context = f"{tqdm_description}, batch={idx}, label_id={label_id}"
            require_finite((ct, mr, label), batch_context + ".input")
            optimizer_unet.zero_grad()  # 优化器清零
            optimizer_pff.zero_grad()   # 优化器清零

            target_ct_pred, point_ct, ct_last, ct_pred = UNet_model(ct)
            target_mr_pred, point_mr, mr_last, mr_pred = UNet_model(mr)
            fusion_target = PFF_model(point_ct, point_mr, ct_last, mr_last, ct_pred, mr_pred)

            new_ct_pred, new_mr_pred, new_fusion_pred, new_label = preprocess_pred(
                target_ct_pred, target_mr_pred, fusion_target, label, label_id)

            require_finite((new_ct_pred, new_mr_pred, new_fusion_pred), batch_context + ".logits")
            ct_prob = torch.sigmoid(new_ct_pred)
            mr_prob = torch.sigmoid(new_mr_pred)
            fusion_prob = torch.sigmoid(new_fusion_pred)

            ct_loss = loss_func(ct_prob, new_label) + focal_loss(ct_prob, new_label)
            mr_loss = loss_func(mr_prob, new_label) + focal_loss(mr_prob, new_label)
            fusion_loss = loss_func(fusion_prob, new_label) + focal_loss(fusion_prob, new_label)
            total_loss = ct_loss * alpha[0] + mr_loss * alpha[1] + fusion_loss * alpha[2]

            require_finite((ct_loss, mr_loss, fusion_loss), batch_context + ".component_losses")
            guarded_backward_step(total_loss, (UNet_model, PFF_model),
                                  (optimizer_unet, optimizer_pff), batch_context,
                                  float(yaml_data.get('other', {}).get('max_grad_norm', 1.0)))

            train_loss.update(total_loss.item(), ct.size(0))
            train_ct_loss.update(ct_loss.item(), ct.size(0))
            train_mr_loss.update(mr_loss.item(), ct.size(0))
            train_fusion_loss.update(fusion_loss.item(), ct.size(0))

            target_ct_pred_one_hot = (ct_prob > 0.5).float()
            target_mr_pred_one_hot = (mr_prob > 0.5).float()
            fusion_target_one_hot = (fusion_prob > 0.5).float()
            train_dice.update(fusion_target_one_hot, new_label, label_id)

            sample_dice = float(train_dice.value[int(label_id)])
            update_image_dict(best_vis_dict, sample_dice, ori_ct, ori_mr, label, target_ct_pred_one_hot,
                              target_mr_pred_one_hot, fusion_target_one_hot, label_id)

            if empty_cache_every and (idx + 1) % empty_cache_every == 0:
                torch.cuda.empty_cache()
            pbar.update()
            pbar.set_postfix(**{'loss': str(train_loss.avg)})

    if is_print_image and best_vis_dict:
        nanseg_show_single(best_vis_dict['ct_original'], best_vis_dict['mr_original'], best_vis_dict['label'],
                           best_vis_dict['ct_pred'], best_vis_dict['mr_pred'], best_vis_dict['fusion_pred'],
                           experiment, yaml_data['labels'], best_vis_dict['label_id'], name='Train')

    train_log = OrderedDict({
        'Train_avg_loss': float(train_loss.avg),
        'Train_ct_loss': float(train_ct_loss.avg),
        'Train_mr_loss': float(train_mr_loss.avg),
        'Train_fusion_loss': float(train_fusion_loss.avg),
        'Train_all_avg_dice': float(train_dice.avg.mean().item()),
    })
    train_log.update(print_dice(train_dice, name='Train'))
    return train_log


def save_checkpoint(child_yaml_data, UNet_model, PFF_model, optimizer_unet, optimizer_pff,
                    lr_scheduler_unet, lr_scheduler_pff, epoch, logger, best_model, val_log):
    is_save_checkpoint = as_bool(child_yaml_data['is_save_checkpoint'])  # 是否保存
    save_dir = Path(child_yaml_data['save_dir'])  # 保存路径
    labels = yaml_data['labels']  # 所有标签

    if is_save_checkpoint:
        save_dir.mkdir(parents=True, exist_ok=True)
        model_state = {'unet': UNet_model.state_dict(), 'pff': PFF_model.state_dict()}
        require_finite(model_state, f"save epoch={epoch}.models")
        require_finite((optimizer_unet.state_dict(), optimizer_pff.state_dict()),
                       f"save epoch={epoch}.optimizers")
        require_finite(val_log, f"save epoch={epoch}.validation")

        sum_dice = 0.
        for i in range(len(labels)):
            name = 'Val_dice_' + labels[i]
            # 验证集里没有出现的器官在上一步已被记为 0.0，这里再兜一层，避免 KeyError
            dice = float(val_log.get(name, 0.))
            sum_dice += dice
            if dice > best_model[i][1]:
                torch.save(model_state, save_dir.joinpath(labels[i] + '_best_model.pth'))
                best_model[i][0] = epoch
                best_model[i][1] = dice

        last_index = len(best_model) - 1
        if sum_dice > best_model[last_index][1]:
            torch.save(model_state, save_dir.joinpath('all_sum_best_model.pth'))
            best_model[last_index][0] = epoch
            best_model[last_index][1] = sum_dice

        # 完整状态（含优化器/学习率调度器/epoch/best_model）用于断点续训
        state = dict(model_state)
        state.update({
            'optimizer_unet': optimizer_unet.state_dict(),
            'optimizer_pff': optimizer_pff.state_dict(),
            'lr_scheduler_unet': lr_scheduler_unet.state_dict(),
            'lr_scheduler_pff': lr_scheduler_pff.state_dict(),
            'epoch': epoch,
            'best_model': best_model,
        })
        torch.save(state, save_dir.joinpath('latest_model.pth'))
        logger.info(f'Checkpoint {epoch} saved!')


def load_checkpoint(child_yaml_data, device, UNet_model, PFF_model, optimizer_unet, optimizer_pff,
                    lr_scheduler_unet, lr_scheduler_pff, logger):
 
    is_load_checkpoint = as_bool(child_yaml_data['is_load_checkpoint'])
    if not is_load_checkpoint:
        return 1, None

    load_dir = Path(child_yaml_data['load_dir'])
    state_dict = torch.load(load_dir, map_location=device)
    require_finite({'unet': state_dict['unet'], 'pff': state_dict['pff']}, 'loaded checkpoint.models')
    if as_bool(child_yaml_data.get('is_load_optimizer', True)):
        require_finite((state_dict.get('optimizer_unet', {}), state_dict.get('optimizer_pff', {})),
                       'loaded checkpoint.optimizers')
    UNet_model.load_state_dict(state_dict['unet'])
    PFF_model.load_state_dict(state_dict['pff'])

    if as_bool(child_yaml_data.get('is_load_optimizer', True)):
        if 'optimizer_unet' in state_dict:
            optimizer_unet.load_state_dict(state_dict['optimizer_unet'])
            optimizer_pff.load_state_dict(state_dict['optimizer_pff'])
        if 'lr_scheduler_unet' in state_dict:
            lr_scheduler_unet.load_state_dict(state_dict['lr_scheduler_unet'])
            lr_scheduler_pff.load_state_dict(state_dict['lr_scheduler_pff'])
    else:
        logger.info('Optimizer/scheduler state ignored (checkpoint.load.is_load_optimizer=False)')

    start_epoch = int(state_dict.get('epoch', 0)) + 1
    logger.info(f'Model loaded from {load_dir}, resume from epoch {start_epoch}')
    return start_epoch, state_dict.get('best_model')


def main():
    logger = logger_config(format='%(levelname)s: %(message)s')  # 日志
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
    torch.autograd.set_detect_anomaly(os.environ.get('PMFM_DETECT_ANOMALY', '0') == '1')
    logger.info(f'Using device {device}')
    logger.info(f'Config: {dir_yaml}')

    set_seed(int(yaml_data['other'].get('seed', 42)))

    epochs = int(yaml_data["schedule"]["total_epochs"])  # 总训练次数
    batch_size = int(yaml_data["schedule"]["batch_size"])  # batch大小
    num_workers = int(yaml_data['data'].get('num_workers', 2))
    learning_rate_unet = float(yaml_data["schedule"]['optimizer_unet']["lr"])  # 优化器的学习率
    learning_rate_pff = float(yaml_data["schedule"]['optimizer_pff']["lr"])  # 优化器的学习率
    num_classes = int(yaml_data['model']['backbone']['num_classes'])
    val_every = max(1, int(yaml_data['other'].get('val_every_epochs', 1)))  # 每 N 个 epoch 验证一次
    compute_hd_asd = as_bool(yaml_data['other'].get('compute_hd_asd', True))  # 验证时是否算 HD95/ASD
    empty_cache_every = max(0, int(yaml_data['other'].get('empty_cache_every', 50)))  # 显存碎片回收间隔


    if batch_size != 1:
        raise ValueError('PFF 的点云融合目前只支持 batch_size=1，'
                         '请把 config/nanseg.yaml 的 schedule.batch_size 设为 1')
    if num_classes != len(yaml_data['labels']):
        raise ValueError(f'num_classes({num_classes}) 与 labels 数量({len(yaml_data["labels"])}) 不一致')

    # 注意：[[0, 0]] * n 会让所有元素指向同一个 list，best_model 的判定会全部失效
    best_model = [[0, 0] for _ in range(num_classes + 1)]  # 前num_classes个存单器官最优，最后一个存总和最优

    # -------------------------------数据--------------------------------------
    train_datasets = TrainDataset(yaml_data['data'])
    n_train = len(train_datasets)
    val_datasets = ValDataset(yaml_data['data'])
    n_val = len(val_datasets)
    train_loader = DataLoader(train_datasets, shuffle=True, batch_size=batch_size,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_datasets, shuffle=False, batch_size=1,
                            num_workers=num_workers, pin_memory=True)
    # -------------------------------------------------------------------------

    # -------------------------------模型--------------------------------------
    UNet_model = build_model(yaml_data['model']['backbone']).to(device)
    UNet_model.apply(weights_init.init_model)
    PFF_model = build_model(yaml_data['model']['fusion']).to(device)
    PFF_model.apply(weights_init.init_model)
    # -------------------------------------------------------------------------

    # -------------------------------优化器-------------------------------------
    optimizer_unet = build_optimizer(UNet_model.parameters(), yaml_data['schedule']['optimizer_unet'])
    optimizer_pff = build_optimizer(PFF_model.parameters(), yaml_data['schedule']['optimizer_pff'])
    lr_scheduler_unet = build_lr_scheduler(optimizer_unet, yaml_data['schedule']['lr_scheduler'])
    lr_scheduler_pff = build_lr_scheduler(optimizer_pff, yaml_data['schedule']['lr_scheduler'])
    # -------------------------------------------------------------------------

    # -------------------------------损失函数-----------------------------------
    loss_cfg = yaml_data.get('loss') or {}
    alpha = [float(x) for x in loss_cfg.get('alpha', [0.25, 0.25, 0.5])]  # CT、 MR、 Fusion各占比例
    focal_alpha = loss_cfg.get('focal_alpha', [0.75, 0.25])
    focal_gamma = float(loss_cfg.get('focal_gamma', 2))
    loss = DiceLoss2D()

    focal_loss = BinaryFocalLoss(alpha=float(focal_alpha[1]), gamma=focal_gamma)
    logger.info(f'''Loss function setting:
                Proportion of weight(CT/MR/Fusion) :    {alpha}
                Focal alpha/gamma :                     {focal_alpha}/{focal_gamma}
    ''')


    start_epoch, loaded_best_model = load_checkpoint(
        yaml_data['checkpoint']['load'], device, UNet_model, PFF_model,
        optimizer_unet, optimizer_pff, lr_scheduler_unet, lr_scheduler_pff, logger)
    if start_epoch > 1 and loaded_best_model and len(loaded_best_model) == len(best_model):
        best_model = loaded_best_model


    rescue_cfg = yaml_data.get('other_rescue') or {}
    rescue_lr = float(rescue_cfg.get('resume_lr', 0) or 0)
    rescue_span = int(rescue_cfg.get('resume_epochs', 0) or 0)
    if rescue_lr > 0 and rescue_span > 0:
        logger.info(f'Rescue schedule: epoch {start_epoch}..{start_epoch + rescue_span - 1} '
                    f'lr={rescue_lr}, afterwards back to {learning_rate_unet}/{learning_rate_pff}')


    wandb_mode = str(yaml_data['other'].get('wandb_mode', 'offline')).lower()
    wandb_kwargs = dict(project=yaml_data['other'].get('wandb_project', 'SI-NanSeg'),
                        name=yaml_data['other'].get('wandb_name', None),
                        resume='allow', mode=wandb_mode)
    if wandb_mode == 'online':
        wandb_kwargs['anonymous'] = 'must'
    experiment = wandb.init(**wandb_kwargs)
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate_unet=learning_rate_unet,
             learning_rate_pff=learning_rate_pff, alpha=alpha)
    )
    logger.info(f'''Starting training:
                Epochs:                 {epochs} (from {start_epoch})
                Batch size:             {batch_size}
                Learning rate (Unet):   {learning_rate_unet}
                Learning rate (PFF):    {learning_rate_pff}
                Training size:          {n_train}
                Validation size:        {n_val} (every {val_every} epoch(s), HD95/ASD: {compute_hd_asd})
                Checkpoints:            {yaml_data['checkpoint']['save']["save_dir"]}
                Device:                 {device.type}
            ''')

    # ##################### 开始训练 ########################### #
    for epoch in range(start_epoch, epochs + 1):
        if rescue_lr > 0 and rescue_span > 0:
            in_rescue = epoch < start_epoch + rescue_span
            for group in optimizer_unet.param_groups:
                group['lr'] = rescue_lr if in_rescue else learning_rate_unet
            for group in optimizer_pff.param_groups:
                group['lr'] = rescue_lr if in_rescue else learning_rate_pff
            if epoch == start_epoch + rescue_span:
                logger.info(f'Rescue finished, lr restored to {learning_rate_unet}/{learning_rate_pff}')

        train_log = train(UNet_model, PFF_model, train_loader, optimizer_unet, optimizer_pff,
                          loss, focal_loss, device, 'train {}/{}'.format(epoch, epochs), alpha, experiment,
                          empty_cache_every=empty_cache_every)


        epoch_log = OrderedDict(train_log)
        val_log = None
        if epoch % val_every == 0:
            val_log = val(UNet_model, PFF_model, val_loader, loss, focal_loss, device,
                          'val {}/{}'.format(epoch, epochs), alpha, experiment, compute_hd_asd=compute_hd_asd,
                          empty_cache_every=empty_cache_every)
            epoch_log.update(val_log)

            lr_scheduler_unet.step(val_log['Val_ct_mr_avg_dice'])  # 调整学习率
            lr_scheduler_pff.step(val_log['Val_fusion_avg_dice'])  # 调整学习率
        else:
            logger.info(f'Epoch {epoch}: validation skipped (val_every_epochs={val_every})')

  
        save_checkpoint(yaml_data['checkpoint']['save'], UNet_model, PFF_model, optimizer_unet, optimizer_pff,
                        lr_scheduler_unet, lr_scheduler_pff, epoch, logger, best_model, val_log or {})

        epoch_log['unet lr'] = optimizer_unet.param_groups[0]["lr"]
        epoch_log['pff lr'] = optimizer_pff.param_groups[0]["lr"]
        experiment.log(epoch_log)

    experiment.finish()


if __name__ == '__main__':
    main()
