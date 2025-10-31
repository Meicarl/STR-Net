import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Support both "package" imports (from strnet.models ...) and direct execution
try:
    # when imported as package: strnet.models.r3d_swintrans
    from .r3d_swintransformer3D import SwinTransformer3D
    from ..blocks.r3d_block import Down, Uper, Out, MSFblock3D, EMA3D, Usp
except Exception:
    # fallback for direct execution (python r3d_swintrans.py)
    # add project root (parent of `strnet`) to sys.path so absolute imports work
    pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)
    from strnet.models.r3d_swintransformer3D import SwinTransformer3D
    from strnet.blocks.r3d_block import Down, Uper, Out, MSFblock3D, EMA3D, Usp

class STR_Net(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 embed_dim=36, 
                 depths=[2, 2, 6, 2], 
                 num_heads=[3, 6, 12, 24], 
                 patch_size = (2,2,2),
                 window_size=(2,2,4), 
                 mlp_ratio=4., 
                 drop_rate=0., 
                 attn_drop_rate=0., 
                 drop_path_rate=0.2, 
                 norm_layer=nn.LayerNorm, 
                 frozen_stages=-1, 
                 use_checkpoint=False):
        super().__init__()
        self.ST3d = SwinTransformer3D(
            in_chans=in_channels,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            patch_size=patch_size,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            frozen_stages=frozen_stages,
            use_checkpoint=use_checkpoint
        )
        self.encoder1 = Down(in_channels=in_channels ,out_channels=embed_dim ,kernel_size=3 ,stride=1 ,res_block=False)
        self.encoder2 = Down(in_channels=embed_dim ,out_channels=embed_dim ,kernel_size=3 ,stride=1 ,res_block=False)
        self.encoder3 = Down(in_channels=2*embed_dim ,out_channels=2*embed_dim ,kernel_size=3 ,stride=1 ,res_block=False)
        self.encoder4 = Down(in_channels=4*embed_dim ,out_channels=4*embed_dim ,kernel_size=3 ,stride=1 ,res_block=False)
        self.encoder5 = Down(in_channels=8*embed_dim ,out_channels=8*embed_dim ,kernel_size=3 ,stride=1 ,res_block=False)
        self.encoder6 = Down(in_channels=16*embed_dim ,out_channels=16*embed_dim ,kernel_size=3 ,stride=1 ,res_block=False)
        self.decoder5 = Uper(in_channels=16*embed_dim ,out_channels=8*embed_dim ,kernel_size=3 ,upsample_kernel_size=2 ,res_block=False)
        self.decoder4 = Uper(in_channels=8*embed_dim ,out_channels=4*embed_dim ,kernel_size=3 ,upsample_kernel_size=2 ,res_block=False)
        self.decoder3 = Uper(in_channels=4*embed_dim ,out_channels=2*embed_dim ,kernel_size=3 ,upsample_kernel_size=2 ,res_block=False)
        self.decoder2 = Uper(in_channels=2*embed_dim ,out_channels=embed_dim ,kernel_size=3 ,upsample_kernel_size=2 ,res_block=False)
        self.decoder1 = Uper(in_channels=embed_dim ,out_channels=embed_dim ,kernel_size=3 ,upsample_kernel_size=2 ,res_block=False)
        self.OUT = Out(in_channels=embed_dim, out_channels=out_channels)

        
        in_channels_list = [embed_dim*8, embed_dim*4, embed_dim*2, embed_dim, embed_dim]
        msfout_channels = embed_dim  #
        self.MSF = MSFblock3D(in_channels_list, msfout_channels)
        
        self.EMA = EMA3D(embed_dim)  


    def forward(self, x):
        hidden_out = self.ST3d(x)
        en1 = self.encoder1(x)
        en2 = self.encoder2(hidden_out[0])
        en3 = self.encoder3(hidden_out[1]) 
        en4 = self.encoder4(hidden_out[2])
        en5 = self.encoder5(hidden_out[3])
        en6 = self.encoder6(hidden_out[4])

        
        de5 = self.decoder5(en6,en5)
        de4 = self.decoder4(de5,en4)
        de3 = self.decoder3(de4,en3)
        de2 = self.decoder2(de3,en2)
        de1 = self.decoder1(de2,en1)

        features = [de5, de4, de3, de2, de1]
        fused_output = self.MSF(features)

        ema_output = self.EMA(fused_output)


        out = self.OUT(ema_output)

        return out
    
if __name__ == '__main__':
    model = STR_Net(in_channels=1, out_channels=1)
    x = torch.randn(1,1,25,25,49)
    out = model(x)
    print(out.size())
