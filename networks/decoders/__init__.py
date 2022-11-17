from .resnet_decoder import ResNet_D_Dec, ResShortCut_D_Decoder, BasicBlock
from .resnet_seg_decoder import ResNet_Seg_Dec,ResShortCut_Seg_Decoder, BasicBlockseg

__all__ = ['res_shortcut_decoder','res_seg_shortcut_decoder']

def res_shortcut_decoder(**kwargs):
    return ResShortCut_D_Decoder(BasicBlock, [2, 3, 3, 2], **kwargs)

def res_seg_shortcut_decoder(**kwargs):
    return ResShortCut_Seg_Decoder(BasicBlockseg, [2, 3, 3, 2], **kwargs)
