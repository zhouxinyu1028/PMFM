from model.unet2d import UNet
from model.CFM import CFM
from model.PMFM import PMFM

def build_model(args):
    name = args['name']
    if name == 'UNet':
        return UNet(int(args['in_channels']), int(args['num_classes']), pointnet=bool(args['pointnet']), isGroupNorm=bool(args['isGroupNorm']), num_gropus=int(args['num_gropus']), is_point=bool(args.get('is_point', True)))
    if name == 'CFM':
        return CFM(num_classes=int(args['num_classes']), isGroupNorm=bool(args['isGroupNorm']), num_gropus=int(args['num_gropus']))
    if name == 'PMFM':
        return PMFM(in_channels=int(args['in_channels']), num_classes=int(args['num_classes']), pointnet=bool(args['pointnet']), isGroupNorm=bool(args['isGroupNorm']), num_gropus=int(args['num_gropus']))
    raise ValueError('Unsupported model name: ' + str(name))
