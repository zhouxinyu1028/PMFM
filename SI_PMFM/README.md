# PMFM: Point-cloud Matrix Fusion Model

Accurate full segmentation of organs-at-risk in head and neck cancer based on multimodal point cloud fusion.

This repository contains a cleaned PMFM implementation aligned with the paper modules: PEM, PMM, CFM, and PMFM. The code keeps only the CT/MR multimodal segmentation training pipeline.

## Highlights

- Paper-aligned module names: PEM, PMM, CFM, PMFM.
- PEM implements Plain Filter and Focus Filter.
- Focus Filter performs CT/MR intersection followed by neighborhood expansion.
- PMM uses PointNet to infer the point-cloud probability matrix.
- CFM fuses CT prediction, MR prediction, and PMM probability maps.
- Unrelated GAN, 3D, prediction, notebook, test, wandb, and visualization code has been removed.

## Project Structure

SI_PMFM/
  config/
    nanseg.yaml
  datasets/
    nanseg/
      common.py
      train_dataset.py
      val_dataset.py
  loss/
    loss.py
    metrics.py
  model/
    PMFM.py
    CFM.py
    pointnet.py
    unet2d.py
    builder_model.py
  utils/
    PEM.py
    dice_score.py
    load_yaml_file.py
    logger_config.py
    weights_init.py
  requirements.txt
  trainNanSegV4.py

## Module Mapping

Paper module | Code file | Role
--- | --- | ---
PEM | utils/PEM.py | Converts 2D masks into point clouds and applies Focus Filter.
PMM | model/CFM.py | PointNet-based point-cloud matrix module.
CFM | model/CFM.py | Cross fusion of CT/MR predictions and point probability maps.
PMFM | model/PMFM.py | Full multimodal segmentation model.

## Dataset

The dataset path is configured in config/nanseg.yaml:

C:\Users\Administrator\Desktop\Datasets\_SI

Expected layout:

_SI/
  train/
    *.npz
  val/
    *.npz
  test/
    train/
    val/

Each npz sample is expected to contain images, labels, and label_id.

## Installation

conda create -n pmfm python=3.10 -y
conda activate pmfm
pip install -r requirements.txt

If you already use the existing environment:

conda activate dualpath

## Training

Edit config/nanseg.yaml if needed, then run:

python trainNanSegV4.py

Training flow:

CT/MR input -> UNet backbone -> PEM -> PMM -> CFM -> fusion prediction

## Configuration

Key options are in config/nanseg.yaml:

- model.backbone: 2D UNet configuration
- model.fusion: CFM configuration
- data.train.datas_dir: training npz directory
- data.val.datas_dir: validation npz directory
- schedule: optimizer, batch size, and epoch settings

## Notes

- This cleaned version keeps only the paper-related PMFM pipeline.
- Legacy names such as PFF and PFFUNet were removed to avoid mismatch with the paper.
- Visualization and point-cloud saving code were removed from the training path.
- If training cannot find data, first confirm that C:\Users\Administrator\Desktop\Datasets\_SI\train exists.

## Citation

If this code is used in a paper or report, cite the corresponding PMFM paper and describe this repository as a cleaned PMFM reproduction.
