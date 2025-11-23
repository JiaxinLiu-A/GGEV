import torch
import torch.nn as nn
import torch.nn.functional as F
from core_rt.update import BasicUpdateBlock, SpatialAttentionExtractor
from core_rt.extractor import Feature
from core_rt.geometry import Combined_Geo_Encoding_Volume
from core_rt.submodule import *
from core_rt.dynamicconv import DynamicConvBlock
from einops import rearrange
import torchvision
from depth_anything_v2.dpt import DepthAnythingV2

try:
    autocast = torch.cuda.amp.autocast
except:
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

def normalize_image(img):
    '''
    @img: (B,C,H,W) in range 0-255, RGB order
    '''
    tf = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
    return tf(img/255.0).contiguous()

class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()
        
        self.feature_att_4 = FeatureAtt(in_channels, 96)
        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))
                                    
        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))                             

        self.dynamicconv1 = DynamicConvBlock(volume_dim=in_channels*2, cont_dim=96)
        self.dynamicconv2 = DynamicConvBlock(volume_dim=in_channels*4, cont_dim=128)

        self.agg_1 = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1))
        
        self.conv1_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=False,
                                  relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))
        self.conv0_up = BasicConv(in_channels*2, in_channels, deconv=True, is_3d=True, bn=False,
                                  relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

    def forward(self, x, features):
        B, C, D, H, W = x.shape
        x = self.feature_att_4(x, features[0])
        x = self.conv1(x)
        x = rearrange(x, 'b c d h w -> (b d) c h w')
        x = self.dynamicconv1(x, features[1], D//2)
        conv1 = rearrange(x, '(b d) c h w -> b c d h w', d=D//2)
        
        x = self.conv2(conv1)
        x = rearrange(x, 'b c d h w -> (b d) c h w')
        x = self.dynamicconv2(x, features[2], D//4)
        x = rearrange(x, '(b d) c h w -> b c d h w', d=D//4)
        
        x = self.conv1_up(x)
        x = torch.cat((x, conv1), dim=1)
        x = self.agg_1(x)
        x = self.conv0_up(x)
        
        return x

class feat_fusion(nn.Module):
    def __init__(self, dim):
        super(feat_fusion, self).__init__()  
        self.conv4x = nn.Conv2d(in_channels=int(dim+96), out_channels=96, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv8x = nn.Conv2d(in_channels=int(dim+64), out_channels=96, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv16x = nn.Conv2d(in_channels=int(int(dim+192)), out_channels=128, kernel_size=1, stride=1, padding=0, bias=False)
        
    def forward(self, mono_features, ori_features):
        feat_4x = self.conv4x(torch.cat((mono_features[0], ori_features[0]), 1))
        feat_8x = self.conv8x(torch.cat((mono_features[1], ori_features[1]), 1))
        feat_16x = self.conv16x(torch.cat((mono_features[2], ori_features[2]), 1))
        return [feat_4x, feat_8x, feat_16x]

class GGEVStereo(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        mono_model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        mono_dim = mono_model_configs[self.args.encoder]['features']
        self.depth_anything = DepthAnythingV2(**mono_model_configs[args.encoder])
        state_dict_dpt = torch.load(f'../pretrained/depth_anything_v2_{args.encoder}.pth', map_location='cpu')
        self.depth_anything.load_state_dict(state_dict_dpt, strict=True)
        self.depth_anything.requires_grad_(False)
        
        self.feature = Feature()
        self.stem_2 = nn.Sequential(
            BasicConv_IN(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32), nn.ReLU()
            )
        self.stem_4 = nn.Sequential(
            BasicConv_IN(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(48), nn.ReLU()
            )
        
        self.feat_fusion = feat_fusion(mono_dim)
        self.upsample_feature = nn.Conv2d(32+32, 32, kernel_size=1, padding=0, stride=1)

        self.desc = nn.Sequential(BasicConv_IN(96, 96, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(96, 96, kernel_size=1, padding=0, stride=1, bias=False))
        self.cost_agg = hourglass(8)
        self.classifier = nn.Conv3d(8, 1, 3, 1, 1, bias=False)
        
        self.spx = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)
        self.spx_2 = Conv2x_IN(24, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv_IN(96, 24, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(24, 24, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(24), nn.ReLU()
            )

        self.spx_2_gru = Conv2x(32, 32, True)
        self.spx_gru = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)
        
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=args.hidden_dim)
        self.hnet = nn.Sequential(BasicConv(96, args.hidden_dim, kernel_size=3, stride=1, padding=1),
                                     nn.Conv2d(args.hidden_dim, args.hidden_dim, 3, 1, 1, bias=False))
        self.sam = SpatialAttentionExtractor()
    
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def upsample_disp(self, disp, mask_feat_4, stem_2x):

        with autocast(enabled=self.args.mixed_precision, dtype=getattr(torch, self.args.precision_dtype, torch.float16)):
            xspx = self.spx_2_gru(mask_feat_4, stem_2x)
            spx_pred = self.spx_gru(xspx)
            spx_pred = F.softmax(spx_pred, 1)
            up_disp = context_upsample(disp, spx_pred)
        return up_disp
    
    def infer_mono(self, image):
        height_ori, width_ori = image.shape[2:]
        image = F.interpolate(image, scale_factor=14 / 16, mode='bicubic', align_corners=False)
        self.depth_anything = self.depth_anything.eval()
        with torch.no_grad():
            depth, mono_features = self.depth_anything.forward_features(image)
            depth = F.interpolate(depth, size=(height_ori, width_ori), mode='bilinear', align_corners=False)
            
        return depth, mono_features
    
    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False):     
        """ Estimate disparity between pair of frames """
        image1 = normalize_image(image1)
        image2 = normalize_image(image2)
        
        with autocast(enabled=self.args.mixed_precision, dtype=getattr(torch, self.args.precision_dtype, torch.float16)):
            B, C, H, W = image1.shape
            depth, mono_features_ori = self.infer_mono(image1)
            mono_feature_2x = mono_features_ori[0]
            mono_features = mono_features_ori[1:]
            
            images = torch.cat([image1, image2], dim=0)
            features = self.feature(images)
            features_left = [f[:B] for f in features]
            features_right = [f[B:] for f in features]
            stem_2 = self.stem_2(images)
            stem_4 = self.stem_4(stem_2)
            stem_2x, stem_2y = stem_2[:B], stem_2[B:]
            stem_4x, stem_4y = stem_4[:B], stem_4[B:]
            features_left[0] = torch.cat((features_left[0], stem_4x), 1)
            features_right[0] = torch.cat((features_right[0], stem_4y), 1)
            
            upsample_feature = self.upsample_feature(torch.cat((stem_2x, mono_feature_2x), 1))
            stereo_features_left = self.feat_fusion(mono_features, features_left)

            match_left = self.desc(features_left[0])
            match_right = self.desc(features_right[0])
            gwc_volume = build_gwc_volume(match_left, match_right, self.args.max_disp//4, 8)
            
            geo_encoding_volume = self.cost_agg(gwc_volume, stereo_features_left)
            
            # Init disp from geometry encoding volume
            prob = F.softmax(self.classifier(geo_encoding_volume).squeeze(1), dim=1)
            init_disp = disparity_regression(prob, self.args.max_disp//4, 1)

            del prob, gwc_volume

            if not test_mode:
                xspx = self.spx_4(stereo_features_left[0])
                xspx = self.spx_2(xspx, upsample_feature)
                spx_pred = self.spx(xspx)
                spx_pred = F.softmax(spx_pred, 1)

            hidden = self.hnet(stereo_features_left[0])
            net = torch.tanh(hidden)
            att = self.sam(stereo_features_left[0])

        geo_block = Combined_Geo_Encoding_Volume
        geo_fn = geo_block(geo_encoding_volume.float(), radius=self.args.corr_radius, num_levels=self.args.corr_levels)

        disp = init_disp
        disp_preds = []

        # GRUs iterations to update disparity
        for itr in range(iters):
            disp = disp.detach()
            geo_feat = geo_fn(disp)
            with autocast(enabled=self.args.mixed_precision, dtype=getattr(torch, self.args.precision_dtype, torch.float16)):
                net, mask_feat_4, delta_disp = self.update_block(net, geo_feat, disp, att)
            disp = disp + delta_disp
            if test_mode and itr < iters-1:
                continue

            # upsample predictions
            disp_up = self.upsample_disp(disp*4., mask_feat_4, upsample_feature)
            disp_preds.append(disp_up)

        if test_mode:
            return disp_up

        init_disp = context_upsample(init_disp*4., spx_pred.float())
        return init_disp, disp_preds, depth
