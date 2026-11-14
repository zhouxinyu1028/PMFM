
import torch
import wandb
from PIL import Image
from torchvision import transforms
import numpy as np


def image_3d_to_2d_show(original_images, original_target, pred_target, experiment):
    # images:[batch_size, channel, s, w, h]
    # image:[channel, s, w, h]

    batch_size, channel, s, w, h = pred_target.shape
    for batch_id in range(batch_size):
        for channel_id in range(channel):
            image_pred = pred_target[batch_id].argmax(dim=0)  # [s, w, h]
            image_target = original_target[batch_id][channel_id]
            original_image = original_images[batch_id][0]
            for i in range(s):
                silce_pred = image_pred[i, :, :]
                silce_target = image_target[i, :, :]
                silce_original_image = original_image[i, :, :]
                experiment.log({
                    "original image": wandb.Image(silce_original_image.cpu()),
                    "target": {
                        'original': wandb.Image(silce_target.float().cpu()),
                        'pred': wandb.Image(silce_pred.float().cpu()),
                    }
                })


def tensor_to_numpy(Tensor):
    if torch.is_tensor(Tensor):
        Tensor = Tensor.numpy()
    return Tensor


def nanseg_show(ct_original, mr_original, original_target, ct_pred_target, mr_pred_target, fusion_pred_target, experiment, args, name='Train'):
  
    ct_original = ct_original.cpu()
    mr_original = mr_original.cpu()
    original_target = original_target.cpu()
    ct_pred_target = ct_pred_target.cpu()
    mr_pred_target = mr_pred_target.cpu()
    fusion_pred_target = fusion_pred_target.cpu()

    ct_original = tensor_to_numpy(ct_original)
    mr_original = tensor_to_numpy(mr_original)
    original_target = tensor_to_numpy(original_target)
    ct_pred_target = tensor_to_numpy(ct_pred_target)
    mr_pred_target = tensor_to_numpy(mr_pred_target)
    fusion_pred_target = tensor_to_numpy(fusion_pred_target)

    ct_original = ct_original.astype(np.uint8)
    mr_original = mr_original.astype(np.uint8)
    original_target = original_target.astype(np.uint8)
    ct_pred_target = ct_pred_target.astype(np.uint8)
    mr_pred_target = mr_pred_target.astype(np.uint8)
    fusion_pred_target = fusion_pred_target.astype(np.uint8)

    experiment.log({
        name + " " + 'ct image': wandb.Image(ct_original[0]),
        name + " " + 'mr image': wandb.Image(mr_original[0])
    })
    label_dict = {}
    num_classes = original_target.shape[1]
    step = 0
    for id in range(num_classes):
        if np.max(original_target[:, id, :, :]).item() == 1:
            label_dict.update({
                name + " " + args[id].replace('_', ' ') + " (original) ": wandb.Image(original_target[0][id]),
                name + " " + args[id].replace('_', ' ') + " (ct_pred) ": wandb.Image(ct_pred_target[0][step]),
                name + " " + args[id].replace('_', ' ') + " (mr_pred) ": wandb.Image(mr_pred_target[0][step]),
                name + " " + args[id].replace('_', ' ') + " (fusion_pred) ": wandb.Image(fusion_pred_target[0][step]),
            })
            step += 1

    experiment.log(label_dict)


def nanseg_show_single(ct_original, mr_original, original_target, ct_pred_target, mr_pred_target, fusion_pred_target, experiment, args, label_id, name):
 
    ct_original = ct_original.cpu()
    mr_original = mr_original.cpu()
    original_target = original_target.cpu()
    ct_pred_target = ct_pred_target.cpu()
    mr_pred_target = mr_pred_target.cpu()
    fusion_pred_target = fusion_pred_target.cpu()

    ct_original = tensor_to_numpy(ct_original)
    mr_original = tensor_to_numpy(mr_original)
    original_target = tensor_to_numpy(original_target)
    ct_pred_target = tensor_to_numpy(ct_pred_target)
    mr_pred_target = tensor_to_numpy(mr_pred_target)
    fusion_pred_target = tensor_to_numpy(fusion_pred_target)

    ct_original = ct_original.astype(np.uint8)
    mr_original = mr_original.astype(np.uint8)
    original_target = original_target.astype(np.uint8)
    ct_pred_target = ct_pred_target.astype(np.uint8)
    mr_pred_target = mr_pred_target.astype(np.uint8)
    fusion_pred_target = fusion_pred_target.astype(np.uint8)

    experiment.log({
        name + " " + 'ct image': wandb.Image(ct_original[0]),
        name + " " + 'mr image': wandb.Image(mr_original[0])
    })
    experiment.log({
        name + " " + args[label_id].replace('_', ' ') + " (original) ": wandb.Image(original_target[0][0]),
        name + " " + args[label_id].replace('_', ' ') + " (ct_pred) ": wandb.Image(ct_pred_target[0][0]),
        name + " " + args[label_id].replace('_', ' ') + " (mr_pred) ": wandb.Image(mr_pred_target[0][0]),
        name + " " + args[label_id].replace('_', ' ') + " (fusion_pred) ": wandb.Image(fusion_pred_target[0][0]),
    })
