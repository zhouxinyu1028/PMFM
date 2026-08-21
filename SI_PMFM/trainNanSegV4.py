import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.nanseg.train_dataset import TrainDataset
from datasets.nanseg.val_dataset import ValDataset
from loss import metrics
from loss.loss import DiceLoss2D, FocalLoss
from model.builder_model import build_model
from utils import weights_init
from utils.load_yaml_file import read_yaml
from utils.logger_config import logger_config

CONFIG = Path(__file__).resolve().parent / 'config' / 'nanseg.yaml'
yaml_data = read_yaml(CONFIG)

def select_label(ct_pred, mr_pred, fusion_pred, label, label_id):
    idx = int(label_id.item()) if torch.is_tensor(label_id) else int(label_id)
    return ct_pred[:, idx:idx + 1], mr_pred[:, idx:idx + 1], fusion_pred[:, idx:idx + 1], label, idx

def make_optimizer(params, cfg):
    return torch.optim.AdamW(params, lr=float(cfg['lr']), weight_decay=float(cfg.get('weight_decay', 0.0)))

def run_epoch(split, unet_model, cfm_model, loader, optimizer_unet, optimizer_cfm, loss_func, focal_loss, device, alpha):
    training = split == 'train'
    unet_model.train(training)
    cfm_model.train(training)
    loss_meter = metrics.LossAverage()
    dice_meter = metrics.DiceAverage2D_single_label(int(yaml_data['model']['backbone']['num_classes']))
    iterator = tqdm(loader, desc=split, unit='batch')
    for ct, mr, label, _, _, label_id in iterator:
        ct, mr, label = ct.float().to(device), mr.float().to(device), label.float().to(device)
        if training:
            optimizer_unet.zero_grad()
            optimizer_cfm.zero_grad()
        with torch.set_grad_enabled(training):
            target_ct_pred, point_ct, ct_last, ct_pred = unet_model(ct)
            target_mr_pred, point_mr, mr_last, mr_pred = unet_model(mr)
            fusion_target = cfm_model(point_ct, point_mr, ct_last, mr_last, ct_pred, mr_pred)
            ct_out, mr_out, fusion_out, label_out, label_index = select_label(target_ct_pred, target_mr_pred, fusion_target, label, label_id)
            ct_loss = loss_func(torch.sigmoid(ct_out), label_out) + focal_loss(torch.sigmoid(ct_out), label_out.long(), True)
            mr_loss = loss_func(torch.sigmoid(mr_out), label_out) + focal_loss(torch.sigmoid(mr_out), label_out.long(), True)
            fusion_loss = loss_func(torch.sigmoid(fusion_out), label_out) + focal_loss(torch.sigmoid(fusion_out), label_out.long(), True)
            total_loss = ct_loss * alpha[0] + mr_loss * alpha[1] + fusion_loss * alpha[2]
            if training:
                total_loss.backward()
                optimizer_unet.step()
                optimizer_cfm.step()
        loss_meter.update(total_loss.item(), ct.size(0))
        pred = (torch.sigmoid(fusion_out) > 0.5).float()
        dice_meter.update(pred.detach(), label_out.detach(), label_index)
        iterator.set_postfix(loss=loss_meter.avg, dice=dice_meter.total_avg)
    return {split + '_loss': loss_meter.avg, split + '_fusion_dice': dice_meter.total_avg}

def save_checkpoint(cfg, unet_model, cfm_model, optimizer_unet, optimizer_cfm, epoch):
    if not bool(cfg['is_save_checkpoint']):
        return
    save_dir = Path(cfg['save_dir'])
    if not save_dir.is_absolute():
        save_dir = Path(__file__).resolve().parent / save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'unet': unet_model.state_dict(), 'cfm': cfm_model.state_dict(), 'optimizer_unet': optimizer_unet.state_dict(), 'optimizer_cfm': optimizer_cfm.state_dict(), 'epoch': epoch}, save_dir / 'latest_model.pth')

def load_checkpoint(cfg, device, unet_model, cfm_model):
    if not bool(cfg['is_load_checkpoint']):
        return
    load_dir = Path(cfg['load_dir'])
    if not load_dir.is_absolute():
        load_dir = Path(__file__).resolve().parent / load_dir
    state = torch.load(load_dir, map_location=device)
    unet_model.load_state_dict(state['unet'])
    cfm_model.load_state_dict(state['cfm'])

def main():
    logger = logger_config(format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Using device %s', device)
    batch_size = int(yaml_data['schedule']['batch_size'])
    train_loader = DataLoader(TrainDataset(yaml_data['data']), shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(ValDataset(yaml_data['data']), shuffle=False, batch_size=1)
    unet_model = build_model(yaml_data['model']['backbone']).to(device)
    cfm_model = build_model(yaml_data['model']['fusion']).to(device)
    unet_model.apply(weights_init.init_model)
    cfm_model.apply(weights_init.init_model)
    load_checkpoint(yaml_data['checkpoint']['load'], device, unet_model, cfm_model)
    optimizer_unet = make_optimizer(unet_model.parameters(), yaml_data['schedule']['optimizer_unet'])
    optimizer_cfm = make_optimizer(cfm_model.parameters(), yaml_data['schedule']['optimizer_cfm'])
    scheduler_unet = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_unet, mode=yaml_data['schedule']['lr_scheduler'].get('mode', 'max'), patience=int(yaml_data['schedule']['lr_scheduler'].get('patience', 5)))
    scheduler_cfm = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_cfm, mode=yaml_data['schedule']['lr_scheduler'].get('mode', 'max'), patience=int(yaml_data['schedule']['lr_scheduler'].get('patience', 5)))
    num_classes = int(yaml_data['model']['backbone']['num_classes'])
    loss_func = DiceLoss2D()
    focal_loss = FocalLoss(class_num=num_classes, alpha=torch.asarray([0.75, 0.25]))
    alpha = [0.25, 0.25, 0.5]
    for epoch in range(1, int(yaml_data['schedule']['total_epochs']) + 1):
        train_log = run_epoch('train', unet_model, cfm_model, train_loader, optimizer_unet, optimizer_cfm, loss_func, focal_loss, device, alpha)
        val_log = run_epoch('val', unet_model, cfm_model, val_loader, optimizer_unet, optimizer_cfm, loss_func, focal_loss, device, alpha)
        scheduler_unet.step(val_log['val_fusion_dice'])
        scheduler_cfm.step(val_log['val_fusion_dice'])
        save_checkpoint(yaml_data['checkpoint']['save'], unet_model, cfm_model, optimizer_unet, optimizer_cfm, epoch)
        logger.info('Epoch %s train=%s val=%s', epoch, train_log, val_log)

if __name__ == '__main__':
    main()
