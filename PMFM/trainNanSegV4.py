import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm


from utils.load_yaml_file import read_yaml
from datasets.nanseg.train_dataset import TrainDataset
from datasets.nanseg.val_dataset import ValDataset
from optimizer.builder_optimizer import build_optimizer
from optimizer.builder_lr_scheduler import build_lr_scheduler
from model.builder_model import build_model
from collections import OrderedDict
from loss import metrics
from loss.loss import FocalLoss, DiceLoss2D
from utils.logger_config import logger_config
import torch.nn.functional as F
from utils import weights_init
from medpy import metric

# yaml配置路径路径
dir_yaml = Path(__file__).resolve().parent / 'config' / 'nanseg.yaml'
yaml_data = read_yaml(dir_yaml)     # 加载配置文件


class NullConfig:
    def update(self, values):
        pass


class NullExperiment:
    def __init__(self):
        self.config = NullConfig()

    def log(self, values):
        pass


def init_experiment(*args, **kwargs):
    return NullExperiment()


def print_dice(evaluation, name='Train', is_BuildIn=False):
    labels = yaml_data['labels']
    result = {}
    if not is_BuildIn:
        # 是否是内置评价指标
        for i in range(len(labels)):
            if evaluation.avg[i].item():
                result.update({name + '_dice_' + labels[i]: evaluation.avg[i]})
    else:
        for i in range(len(labels)):
            # dice
            if evaluation.dice_avg[i].item():
                result.update({name + '_dice(Build-In)_' + labels[i]: evaluation.dice_avg[i]})
            # jc
            if evaluation.jc_avg[i].item():
                result.update({name + '_jc(Build-In)_' + labels[i]: evaluation.jc_avg[i]})
            # hd95
            if evaluation.hd_avg[i].item():
                result.update({name + '_hd95(Build-In)_' + labels[i]: evaluation.hd_avg[i]})
            # asd
            if evaluation.asd_avg[i].item():
                result.update({name + '_asd(Build-In)_' + labels[i]: evaluation.asd_avg[i]})
    return result


def preprocess_pred(ct_pred, mr_pred, fusion_pred, target, label_id):

    return ct_pred[:, label_id, :, :], mr_pred[:, label_id, :, :], fusion_pred[:, label_id, :, :], target


# 内置评价指标
def Built_in_evaluation_metrics(pred, gt):
    one_hot_pred = (F.sigmoid(pred) > 0.5).float()
    # print("min = ", torch.min(one_hot_pred).item(), "max = ", torch.max(one_hot_pred).item())
    one_hot_pred = one_hot_pred.detach().cpu().numpy()

    gt = gt.detach().cpu().numpy()
    dice = metric.binary.dc(one_hot_pred, gt)
    jc = metric.binary.jc(one_hot_pred, gt)
    if np.any(one_hot_pred):
        hd = metric.binary.hd95(one_hot_pred, gt)
        asd = metric.binary.asd(one_hot_pred, gt)
    else:
        hd = float('inf')
        asd = float('inf')
    return dice, jc, hd, asd


def val(child_yaml_data, UNet_model, PFF_model, val_loader, loss_func, focal_loss, device, experiment, tqdm_description, alpha):
    num_classes = int(child_yaml_data['num_classes'])  # 类别数
    UNet_model.eval()
    PFF_model.eval()
    val_loss = metrics.LossAverage()
    val_ct_dice = metrics.DiceAverage2D_single_label(num_classes)
    val_mr_dice = metrics.DiceAverage2D_single_label(num_classes)
    val_fusion_dice = metrics.DiceAverage2D_single_label(num_classes)

    # 内置评价指标
    val_build_in_evaluation = metrics.BuiltInEvaluationMetrics(num_classes)

    with torch.no_grad():
        with tqdm(total=len(val_loader), desc=tqdm_description, unit='batch') as pbar:
            for idx, (ct, mr, label, ori_ct, ori_mr, label_id) in enumerate(val_loader):
                ct, mr = ct.float(), mr.float()
                ct, mr, label = ct.to(device), mr.to(device), label.to(device)

                target_ct_pred, point_ct, ct_last, ct_pred = UNet_model(ct)
                target_mr_pred, point_mr, mr_last, mr_pred = UNet_model(mr)
                fusion_target = PFF_model(point_ct, point_mr, ct_last, mr_last, ct_pred, mr_pred)

                new_ct_pred, new_mr_pred, new_fusion_pred, new_label = preprocess_pred(target_ct_pred, target_mr_pred, fusion_target, label, label_id)

                ct_loss = loss_func(F.sigmoid(new_ct_pred), new_label.to(device))
                ct_loss += focal_loss(F.sigmoid(new_ct_pred).to(device), new_label.to(device).long(), True)
                mr_loss = loss_func(F.sigmoid(new_mr_pred), new_label.to(device))
                mr_loss += focal_loss(F.sigmoid(new_mr_pred).to(device), new_label.to(device).long(), True)
                fusion_loss = loss_func(F.sigmoid(new_fusion_pred), new_label.to(device))
                fusion_loss += focal_loss(F.sigmoid(new_fusion_pred).to(device), new_label.to(device).long(), True)

                total_loss = ct_loss * alpha[0] + mr_loss * alpha[1] + fusion_loss * alpha[2]
                val_loss.update(total_loss.item(), ct.size(0))

                target_ct_pred_one_hot = (F.sigmoid(new_ct_pred) > 0.5).float()
                target_mr_pred_one_hot = (F.sigmoid(new_mr_pred) > 0.5).float()
                fusion_target_one_hot = (F.sigmoid(new_fusion_pred) > 0.5).float()
                val_ct_dice.update(target_ct_pred_one_hot, new_label, label_id)
                val_mr_dice.update(target_mr_pred_one_hot, new_label, label_id)
                val_fusion_dice.update(fusion_target_one_hot, new_label, label_id)

                dice, jc, hd, asd = Built_in_evaluation_metrics(new_fusion_pred, new_label)
                val_build_in_evaluation.update(dice, jc, hd, asd, label_id)

                experiment.log({
                    'Val_ct_loss': ct_loss.item(),
                    'Val_mr_loss': mr_loss.item(),
                    'Val_fusion_loss': fusion_loss.item(),
                    'Val_total_loss': total_loss.item()
                })
                pbar.update()
                pbar.set_postfix(**{'loss': str(val_loss.avg)})

    val_log = OrderedDict({
        'Val_fusion_avg_dice': val_fusion_dice.total_avg,
        'Val_ct_avg_dice': val_ct_dice.total_avg,
        'Val_mr_avg_dice': val_mr_dice.total_avg,
        'Val_ct_mr_avg_dice': (val_mr_dice.total_avg + val_ct_dice.total_avg) / 2,
        'Val_avg_loss': val_loss.avg,
    })
    val_log.update(print_dice(val_fusion_dice, name='Val'))
    val_log.update(print_dice(val_build_in_evaluation, name='Val', is_BuildIn=True))
    experiment.log(val_log)
    return val_log


def train(child_yaml_data, UNet_model, PFF_model, train_loader, optimizer_unet, optimizer_pff, loss_func, focal_loss, device,
          experiment, tqdm_description, alpha):
    num_classes = int(child_yaml_data['num_classes'])  # 类别数
    UNet_model.train()  # 将模型调整为训练模式
    PFF_model.train()

    train_loss = metrics.LossAverage()
    train_dice = metrics.DiceAverage2D_single_label(num_classes)
    # 内置评价指标
    train_build_in_evaluation = metrics.BuiltInEvaluationMetrics(num_classes)

    with tqdm(total=len(train_loader), desc=tqdm_description, unit='batch') as pbar:
        for idx, (ct, mr, label, ori_ct, ori_mr, label_id) in enumerate(train_loader):
            ct, mr = ct.float(), mr.float()
            ct, mr, label = ct.to(device), mr.to(device), label.to(device)

            optimizer_unet.zero_grad()  # 优化器清零
            optimizer_pff.zero_grad()   # 优化器清零

            target_ct_pred, point_ct, ct_last, ct_pred = UNet_model(ct)
            target_mr_pred, point_mr, mr_last, mr_pred = UNet_model(mr)
            fusion_target = PFF_model(point_ct, point_mr, ct_last, mr_last, ct_pred, mr_pred)

            new_ct_pred, new_mr_pred, new_fusion_pred, new_label = preprocess_pred(target_ct_pred, target_mr_pred, fusion_target, label, label_id)

            ct_loss = loss_func(F.sigmoid(new_ct_pred), new_label.to(device))
            ct_loss += focal_loss(F.sigmoid(new_ct_pred).to(device), new_label.to(device).long(), True)
            mr_loss = loss_func(F.sigmoid(new_mr_pred), new_label.to(device))
            mr_loss += focal_loss(F.sigmoid(new_mr_pred).to(device), new_label.to(device).long(), True)
            fusion_loss = loss_func(F.sigmoid(new_fusion_pred), new_label.to(device))
            fusion_loss += focal_loss(F.sigmoid(new_fusion_pred).to(device), new_label.to(device).long(), True)

            total_loss = ct_loss * alpha[0] + mr_loss * alpha[1] + fusion_loss * alpha[2]
            total_loss.backward()  # 反向传播
            optimizer_unet.step()  # 更新模型参数
            optimizer_pff.step()  # 更新模型参数

            train_loss.update(total_loss.item(), ct.size(0))
            target_ct_pred_one_hot = (F.sigmoid(new_ct_pred) > 0.5).float()

            target_mr_pred_one_hot = (F.sigmoid(new_mr_pred) > 0.5).float()
            fusion_target_one_hot = (F.sigmoid(new_fusion_pred) > 0.5).float()
            train_dice.update(fusion_target_one_hot, new_label, label_id)

            dice, jc, hd, asd = Built_in_evaluation_metrics(new_fusion_pred, new_label)
            train_build_in_evaluation.update(dice, jc, hd, asd, label_id)

            experiment.log({
                'Train_ct_loss': ct_loss.item(),
                'Train_mr_loss': mr_loss.item(),
                'Train_fusion_loss': fusion_loss.item(),
                'Train_total_loss': total_loss.item()
            })
            pbar.update()
            pbar.set_postfix(**{'loss': str(train_loss.avg)})

    train_log = OrderedDict({
        'Train_avg_loss': train_loss.avg,
        'Train_all_avg_dice': train_dice.avg.mean().item(),
    })
    train_log.update(print_dice(train_dice, name='Train'))
    train_log.update(print_dice(train_build_in_evaluation, name='Train', is_BuildIn=True))
    experiment.log(train_log)
    return train_log


def save_checkpoint(child_yaml_data, UNet_model, PFF_model, optimizer_unet, optimizer_pff, epoch, logger, best_model,
                    val_log):
    is_save_checkpoint = bool(child_yaml_data['is_save_checkpoint'])  # 是否保存
    save_dir = Path(child_yaml_data['save_dir'])  # 保存路径
    labels = yaml_data['labels']  # 所有标签

    if is_save_checkpoint:
        save_dir.mkdir(parents=True, exist_ok=True)
        state = {'unet': UNet_model.state_dict(), 'pff': PFF_model.state_dict(),
                 'optimizer_unet': optimizer_unet.state_dict(), 'optimizer_pff': optimizer_pff.state_dict(),
                 'epoch': epoch}
        torch.save(state, save_dir.joinpath('latest_model.pth'))
        sum_dice = 0
        for i in range(len(labels)):
            name = 'Val_dice_' + labels[i]
            sum_dice += val_log[name]
            if val_log[name] > best_model[i][1]:
                torch.save(state, save_dir.joinpath(labels[i] + '_best_model.pth'))
                best_model[i][0] = epoch
                best_model[i][1] = val_log[name]
        last_index = len(best_model) - 1
        if sum_dice > best_model[last_index][1]:
            torch.save(state, save_dir.joinpath('all_sum_best_model.pth'))
            best_model[last_index][0] = epoch
            best_model[last_index][1] = sum_dice
        logger.info(f'Checkpoint {epoch} saved!')


def load_checkpoint(child_yaml_data, device, UNet_model, PFF_model, logger):
    is_load_checkpoint = bool(child_yaml_data['is_load_checkpoint'])
    load_dir = Path(child_yaml_data['load_dir'])
    if is_load_checkpoint:
        state_dict = torch.load(load_dir, map_location=device)
        UNet_model.load_state_dict(state_dict['unet'])
        PFF_model.load_state_dict(state_dict['pff'])
        logger.info(f'Model loaded from {load_dir}')


def main():
    logger = logger_config(format='%(levelname)s: %(message)s')  # 日志
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
    logger.info(f'Using device {device}')
    epochs = int(yaml_data["schedule"]["total_epochs"])  # 总训练次数
    batch_size = int(yaml_data["schedule"]["batch_size"])  # batch大小
    learning_rate_unet = float(yaml_data["schedule"]['optimizer_unet']["lr"])  # 优化器的学习率
    learning_rate_pff = float(yaml_data["schedule"]['optimizer_pff']["lr"])  # 优化器的学习率
    num_classes = yaml_data['model']['backbone']['num_classes']
    best_model = [[0, 0]] * (num_classes + 1)  # 0表示liver最优，1表示tumor最优模型，2表示两个的均值最优
    # -------------------------------数据--------------------------------------
    train_datasets = TrainDataset(yaml_data['data'])
    n_train = len(train_datasets)
    val_datasets = ValDataset(yaml_data['data'])
    n_val = len(val_datasets)
    train_loader = DataLoader(train_datasets, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_datasets, shuffle=False, batch_size=1)
    # -------------------------------------------------------------------------

    # -------------------------------模型--------------------------------------
    UNet_model = build_model(yaml_data['model']['backbone']).to(device)
    UNet_model.apply(weights_init.init_model)
    PFF_model = build_model(yaml_data['model']['fusion']).to(device)
    PFF_model.apply(weights_init.init_model)
    load_checkpoint(yaml_data['checkpoint']['load'], device, UNet_model, PFF_model, logger)
    # -------------------------------------------------------------------------

    # -------------------------------优化器-------------------------------------
    optimizer_unet = build_optimizer(UNet_model.parameters(), yaml_data['schedule']['optimizer_unet'])
    optimizer_pff = build_optimizer(PFF_model.parameters(), yaml_data['schedule']['optimizer_pff'])
    lr_scheduler_unet = build_lr_scheduler(optimizer_unet, yaml_data['schedule']['lr_scheduler'])
    lr_scheduler_pff = build_lr_scheduler(optimizer_pff, yaml_data['schedule']['lr_scheduler'])
    # -------------------------------损失函数-------------------------------------
    logger.info(f'Loading the weights...')
    # weights = torch.asarray(setting_weights(yaml_data)).to(device)   # 动态获取交叉熵损失权重
    alpha = [0.25, 0.25, 0.5]   # CT、 MR、 Fusion各占比例
    loss = DiceLoss2D()         # diceloss用于辅助损失
    focal_loss = FocalLoss(class_num=num_classes, alpha=torch.asarray([0.75, 0.25]))
    logger.info(f'''Loss function setting:
                Proportion of weight(CT/MR/Fusion) :    {alpha}
    ''')
    # -------------------------------------------------------------------------
    # (初始化日志)
    experiment = init_experiment(project='TestSize', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate_unet=learning_rate_unet, learning_rate_pff=learning_rate_pff)
    )
    logger.info(f'''Starting training:
                Epochs:                 {epochs}
                Batch size:             {batch_size}
                Learning rate (Unet):   {learning_rate_unet}
                Learning rate (PFF):    {learning_rate_pff}
                Training size:          {n_train}
                Validation size:        {n_val}
                Checkpoints:            {yaml_data['checkpoint']['save']["save_dir"]}
                Device:                 {device.type}
            ''')

    # ##################### 开始训练 ########################### #
    for epoch in range(1, epochs + 1):
        train_log = train(yaml_data['model']['backbone'], UNet_model, PFF_model, train_loader, optimizer_unet,
                          optimizer_pff, loss, focal_loss, device, experiment, 'train {}/{}'.format(epoch, epochs), alpha)
        val_log = val(yaml_data['model']['backbone'], UNet_model, PFF_model, val_loader, loss, focal_loss, device, experiment,
                      'val {}/{}'.format(epoch, epochs), alpha)

        lr_scheduler_unet.step(val_log['Val_ct_mr_avg_dice'])  # 调整学习率
        lr_scheduler_pff.step(val_log['Val_fusion_avg_dice'])  # 调整学习率

        save_checkpoint(yaml_data['checkpoint']['save'], UNet_model, PFF_model, optimizer_unet, optimizer_pff, epoch,
                        logger, best_model, val_log)

        experiment.log({
            'unet lr': optimizer_unet.param_groups[0]["lr"],
            'pff lr': optimizer_pff.param_groups[0]["lr"]
        })


if __name__ == '__main__':
    main()
