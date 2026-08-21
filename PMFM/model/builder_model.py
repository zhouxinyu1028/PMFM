from model.PFF import PFF
from model.unet2d import UNet
from model.PFFUNet import PFFUNet


def build_model(args):
    if args['name'] == 'PFF':
        return PFF(num_classes=int(args['num_classes']), isGroupNorm=bool(args['isGroupNorm']), num_gropus=int(args['num_gropus']))
    elif args['name'] == 'UNet':
        return UNet(in_channels=int(args['in_channels']), num_classes=int(args['num_classes']), base_c=int(args.get('base_c', 64)), pointnet=bool(args['pointnet']), isGroupNorm=bool(args['isGroupNorm']), num_gropus=int(args['num_gropus']), is_point=bool(args['is_point']))
    elif args['name'] == 'PFFUNet':
        return PFFUNet(in_channels=int(args['in_channels']), num_classes=int(args['num_classes']), base_c=int(args.get('base_c', 64)), pointnet=bool(args['pointnet']), isGroupNorm=bool(args['isGroupNorm']), num_gropus=int(args['num_gropus']))
    else:
        raise ValueError(f"Unsupported PMFM model: {args['name']}")

