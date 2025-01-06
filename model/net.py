import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.rotation_utils import Ortho6d2Mat
from model.seg_model import optimize_shape_ransac

from model.modules import ModifiedResnet, PointNet2MSG
from model.losses import SmoothL1Dis, ChamferDis, PoseDis, main_PoseDis

from model.pclfeats import PclFeats


class Net(nn.Module):
    def __init__(self, nclass=6, freeze_world_enhancer=False):
        super(Net, self).__init__()
        self.nclass = nclass
        self.freeze_world_enhancer=freeze_world_enhancer
        self.rgb_cam_extractor = ModifiedResnet()
        self.pts_cam_extractor = PointNet2MSG(radii_list=[[0.01, 0.02], [0.02, 0.04], [0.04, 0.08], [0.08, 0.16]])
        self.implicit_transform = ImplicitTransformation(nclass)
        self.main_estimator = HeavyEstimator(num_obj=10)
        self.cam_enhancer = LightEstimator()
        self.world_enhancer = WorldSpaceEnhancer(freeze=freeze_world_enhancer)
        self.generative_decoder = GenerativeDecoder()

        self.pcl_feat = PclFeats()

    def get_current_feature(self, shape_descriptors, labels):
        target_nlabel = 128
        bs = len(labels)
        current_centers = {}
        current_features = np.zeros((bs, target_nlabel))
        current_features = torch.from_numpy(current_features)
        for i in range(bs):
            label = labels[i]
            current_center = shape_descriptors[i, label, 1:]
            current_centers[i] = current_center
            current_feature = optimize_shape_ransac(label, current_center)
            linear = nn.Linear(in_features=current_feature.size(1), out_features=target_nlabel)
            linear = linear.to(current_feature.device)
            current_feature = linear(current_feature)

            current_features[i] = current_feature
        current_features = current_features.to(device)
        return current_features


    def forward(self, inputs, current_feature):
        end_points = {}
        bs = current_feature.size(0)

        # assert False
        rgb = inputs['rgb']
        pts = inputs['pts']
        choose = inputs['choose']
        obj_id = inputs['category_label']


        if self.training:
            pts_w_gt = inputs['qo']
        cls = inputs['category_label'].reshape(-1)

        center = torch.mean(pts, 1, keepdim=True)
        pts = pts - center

        b = pts.size(0)
        index = cls + torch.arange(b, dtype=torch.long).cuda() * self.nclass

        # rgb feat
        rgb_local = self.rgb_cam_extractor(rgb)

        d = rgb_local.size(1)
        rgb_local = rgb_local.view(b, d, -1)
        choose = choose.unsqueeze(1).repeat(1, d, 1)
        rgb_local = torch.gather(rgb_local, 2, choose).contiguous()

        if self.training:

            # pcl feature from pointnet
            pts_local = self.pts_cam_extractor(pts)
            # print('pts_local:', pts_local.size())

            r_aux_cam, t_aux_cam, s_aux_cam = self.cam_enhancer(pts, rgb_local, pts_local)
            pts_w, pts_w_local = self.implicit_transform(rgb_local, pts_local, pts, center, index, current_feature)
            r, t, s, c = self.main_estimator(pts, pts_w, rgb_local, pts_local, pts_w_local)
            r_aux_world, t_aux_world, s_aux_world, pts_w_local_gt = self.world_enhancer(pts, pts_w_gt, rgb_local, pts_local, obj_id)

            end_points["pred_qo"] = pts_w
            end_points["pts_w_local"] = pts_w_local
            end_points["pts_w_local_gt"] = pts_w_local_gt
            end_points['pred_rotation'] = r
            end_points['pred_translation'] = t + center
            end_points['pred_size'] = s
            end_points['pred_rotation_aux_cam'] = r_aux_cam
            end_points['pred_translation_aux_cam'] = t_aux_cam + center.squeeze(1)
            end_points['pred_size_aux_cam'] = s_aux_cam
            end_points['pred_confidence'] = c
            if not self.freeze_world_enhancer:
                end_points['pred_rotation_aux_world'] = r_aux_world
                end_points['pred_translation_aux_world'] = t_aux_world + center.squeeze(1)
                end_points['pred_size_aux_world'] = s_aux_world
        else:
            pts_local = self.pts_cam_extractor(pts)
            pts_w, pts_w_local = self.implicit_transform(rgb_local, pts_local, pts, center, index, current_feature)
            r, t, s, c = self.main_estimator(pts, pts_w, rgb_local, pts_local, pts_w_local)
            end_points["pred_qo"] = pts_w
            end_points['pred_rotation'] = r
            end_points['pred_translation'] = t + center
            end_points['pred_size'] = s
            end_points['pred_confidence'] = c

        return end_points

class SupervisedLoss(nn.Module):
    def __init__(self, cfg):
        super(SupervisedLoss, self).__init__()
        self.cfg=cfg.loss
        self.freeze_world_enhancer=cfg.freeze_world_enhancer

    def forward(self, end_points):
        qo = end_points['pred_qo']
        t = end_points['pred_translation']
        r = end_points['pred_rotation']
        s = end_points['pred_size']
        c = end_points['pred_confidence']
        loss = self._get_loss(r, t, s, c, qo, end_points)

        return loss
  
    def _get_loss(self, r, t, s, c, qo, end_points, w=0.015):
        pts_w_local = end_points["pts_w_local"]
        pts_w_local_gt = end_points["pts_w_local_gt"]
        t_aux_cam = end_points['pred_translation_aux_cam']
        r_aux_cam = end_points['pred_rotation_aux_cam']
        s_aux_cam = end_points['pred_size_aux_cam']
        loss_feat = nn.functional.mse_loss(pts_w_local, pts_w_local_gt)
        loss_qo = SmoothL1Dis(qo, end_points['qo'])
        loss_pose, dis = main_PoseDis(r, t, s, c, end_points['rotation_label'],end_points['translation_label'],end_points['size_label'])
        loss_confid = torch.mean((dis * c.squeeze(2) - w * torch.log(c.squeeze(2))))
        loss_pose_aux_cam = PoseDis(r_aux_cam, t_aux_cam, s_aux_cam, end_points['rotation_label'],end_points['translation_label'], end_points['size_label'])
        cfg = self.cfg
        loss = loss_pose + loss_pose_aux_cam + cfg.gamma1 * loss_qo + cfg.gamma2*loss_feat + cfg.gamma3*loss_confid
        if not self.freeze_world_enhancer:
            r_aux_world = end_points['pred_rotation_aux_world']
            t_aux_world = end_points['pred_translation_aux_world']
            s_aux_world = end_points['pred_size_aux_world']
            loss_pose_aux_world = PoseDis(r_aux_world, t_aux_world, s_aux_world, end_points['rotation_label'],end_points['translation_label'], end_points['size_label'])
            loss = loss + loss_pose_aux_world
        return loss

class ShapeLoss(nn.Module):
    def __init__(self):
        super(ShapeLoss, self).__init__()
        self.loss = torch.nn.MSELoss()

    def shape_loss(self, current_feature, target_feature, bs):
        loss = 0
        for i in range(bs):
            current = current_feature[i]
            target = target_feature[i]
            temp_loss = self.loss(current, target)
            loss = loss + temp_loss
        loss = loss / bs
        return loss

    def forward(self, current_features, target_features, feature):
        bs = len(current_features)
        loss = self.shape_loss(current_features, target_features, bs) + 0.0001*torch.sqrt(torch.sum(feature**2)+1e-8)
        return loss

class ImplicitTransformation(nn.Module):
    def __init__(self, nclass=6):
        super(ImplicitTransformation, self).__init__()
        self.nclass = nclass
        self.feature_refine = FeatureDeformer(nclass)

    def forward(self, rgb_local, pts_local, pts, center, index, feature):
        pts_local_w, pts_w = self.feature_refine(pts, rgb_local, pts_local, index, feature)
        return pts_w, pts_local_w


class FeatureDeformer(nn.Module):
    def __init__(self, nclass=6):
        super(FeatureDeformer, self).__init__()
        self.nclass = nclass

        self.pts_mlp1 = nn.Sequential(
            nn.Conv1d(3, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 1),
            nn.ReLU(),
        )

        self.deform_mlp1 = nn.Sequential(
            nn.Conv1d(320, 384, 1),
            nn.ReLU(),
            nn.Conv1d(384, 256, 1),
            nn.ReLU(),
        )

        self.deform_mlp2 = nn.Sequential(
            nn.Conv1d(640, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 384, 1),
            nn.ReLU(),
            nn.Conv1d(384, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.ReLU(),
        )

        self.deform_mlp3 = nn.Sequential(
            nn.Conv1d(128, 256, 1),
            nn.ReLU(),
            # nn.Conv1d(384, 256, 1),
            # nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.ReLU(),
        )


        self.pred_nocs = nn.Sequential(
            nn.Conv1d(128, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, nclass*3, 1),
        )

    def forward(self, pts, rgb_local, pts_local, index, feature):
        npoint = pts_local.size(2)
        pts_pose_feat = self.pts_mlp1(pts.transpose(1,2))

        # with pcl
        deform_feat = torch.cat([
            pts_pose_feat,
            pts_local,
            rgb_local,
        ], dim=1)

        pts_local_w = self.deform_mlp1(deform_feat)
        pts_global_w = torch.mean(pts_local_w, 2, keepdim=True)
        pred_local = self.deform_mlp3(feature.to(torch.float32).unsqueeze(-1).expand(-1, -1, 1024))
        pts_local_w = torch.cat([pts_local_w, pred_local, pts_global_w.expand_as(pts_local_w)], 1)
        pts_local_w = self.deform_mlp2(pts_local_w)

        pts_w = self.pred_nocs(pts_local_w)
        pts_w = pts_w.view(-1, 3, npoint).contiguous()
        pts_w = torch.index_select(pts_w, 0, index)   
        pts_w = pts_w.permute(0, 2, 1).contiguous()   

        return pts_local_w, pts_w
    
class WorldSpaceEnhancer(nn.Module):
    def __init__(self, freeze=False):
        super(WorldSpaceEnhancer, self).__init__()
        self.freeze=freeze
        self.extractor = PointNet2MSG(radii_list=[[0.05,0.10], [0.10,0.20], [0.20,0.30], [0.30,0.40]])
        self.pcl_feat = PclFeats()
        if not freeze:
            self.pose_estimator = WorldSpace_HeavyEstimator()
    
    def forward(self, pts, pts_w_gt, rgb_local, pts_local, obj_id):
        if not self.freeze:
            pts_w_local_gt = self.extractor(pts_w_gt)
            r_aux_world, t_aux_world, s_aux_world = self.pose_estimator(pts, pts_w_gt, rgb_local.detach(), pts_local.detach(), pts_w_local_gt)
            return r_aux_world, t_aux_world, s_aux_world, pts_w_local_gt
        else:
            pts_feat_gt = self.extractor(pts_w_gt)

            return None, None, None, pts_feat_gt

class LightEstimator(nn.Module):
    def __init__(self):
        super(LightEstimator, self).__init__()

        self.pts_mlp = nn.Sequential(
            nn.Conv1d(3, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 1),
            nn.ReLU(),
        )

        self.pose_mlp1 = nn.Sequential(
            nn.Conv1d(128+64+128, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 256, 1),
            nn.ReLU(),
        )

        self.pose_mlp2 = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.rotation_estimator = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 6),
        )
        self.translation_estimator = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )
        self.size_estimator = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )

    def forward(self, pts, rgb_local, pts_local):

        pts = self.pts_mlp(pts.transpose(1,2))
        pose_feat = torch.cat([rgb_local, pts, pts_local], dim=1) # 

        pose_feat = self.pose_mlp1(pose_feat)
        pose_global = torch.mean(pose_feat, 2, keepdim=True)
        pose_feat = torch.cat([pose_feat, pose_global.expand_as(pose_feat)], 1)
        pose_feat = self.pose_mlp2(pose_feat).squeeze(2)

        r = self.rotation_estimator(pose_feat)
        r = Ortho6d2Mat(r[:, :3].contiguous(), r[:, 3:].contiguous()).view(-1,3,3)
        t = self.translation_estimator(pose_feat)
        s = self.size_estimator(pose_feat)
        return r,t,s

class WorldSpace_HeavyEstimator(nn.Module):
    def __init__(self):
        super(WorldSpace_HeavyEstimator, self).__init__()

        self.pts_mlp1 = nn.Sequential(
            nn.Conv1d(3, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 1),
            nn.ReLU(),
        )
        self.pts_mlp2 = nn.Sequential(
            nn.Conv1d(3, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 1),
            nn.ReLU(),
        )
        self.pose_mlp1 = nn.Sequential(
            nn.Conv1d(64+64+384, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 256, 1),
            nn.ReLU(),
        )
        self.pose_mlp2 = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.rotation_estimator = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 6),
        )
        self.translation_estimator = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )
        self.size_estimator = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )

    def forward(self, pts, pts_w, rgb_local, pts_local, pts_w_local):
        pts = self.pts_mlp1(pts.transpose(1,2))
        pts_w = self.pts_mlp2(pts_w.transpose(1,2))

        pose_feat = torch.cat([rgb_local, pts, pts_local, pts_w, pts_w_local], dim=1)
        pose_feat = self.pose_mlp1(pose_feat)
        pose_global = torch.mean(pose_feat, 2, keepdim=True)
        pose_feat = torch.cat([pose_feat, pose_global.expand_as(pose_feat)], 1)
        pose_feat = self.pose_mlp2(pose_feat).squeeze(2)

        r = self.rotation_estimator(pose_feat)
        r = Ortho6d2Mat(r[:, :3].contiguous(), r[:, 3:].contiguous()).view(-1,3,3)
        t = self.translation_estimator(pose_feat)
        s = self.size_estimator(pose_feat)
        return r,t,s
    
class HeavyEstimator(nn.Module):
    def __init__(self, num_obj):
        super(HeavyEstimator, self).__init__()

        self.pts_mlp1 = nn.Sequential(
            nn.Conv1d(3, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 1),
            nn.ReLU(),
        )
        self.pts_mlp2 = nn.Sequential(
            nn.Conv1d(3, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 1),
            nn.ReLU(),
        )
        self.pose_mlp1 = nn.Sequential(
            nn.Conv1d(64+64+384, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 256, 1),
            nn.ReLU(),
        )
        self.pose_mlp2 = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.rotation_estimator = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 6),
        )
        self.translation_estimator = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )
        self.size_estimator = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )

        # estimator branch
        self.conv1_r = torch.nn.Conv1d(512, 256, 1)
        self.conv1_t = torch.nn.Conv1d(512, 256, 1)
        self.conv1_c = torch.nn.Conv1d(512, 256, 1)
        self.conv1_s = torch.nn.Conv1d(512, 256, 1)

        self.conv2_r = torch.nn.Conv1d(256, 128, 1)
        self.conv2_t = torch.nn.Conv1d(256, 128, 1)
        self.conv2_c = torch.nn.Conv1d(256, 128, 1)
        self.conv2_s = torch.nn.Conv1d(256, 128, 1)

        self.conv3_r = torch.nn.Conv1d(128, num_obj * 6, 1)
        self.conv3_t = torch.nn.Conv1d(128, num_obj * 3, 1)
        self.conv3_c = torch.nn.Conv1d(128, num_obj * 1, 1)
        self.conv3_s = torch.nn.Conv1d(128, num_obj * 3, 1)

        self.adaptive_r = nn.AdaptiveAvgPool1d(1)
        self.adaptive_t = nn.AdaptiveAvgPool1d(1)
        self.adaptive_c = nn.AdaptiveAvgPool1d(1)
        self.adaptive_s = nn.AdaptiveAvgPool1d(1)

        self.num_obj = num_obj

    def forward(self, pts, pts_w, rgb_local, pts_local, pts_w_local):
        bs = pts.size(0)
        pts = self.pts_mlp1(pts.transpose(1,2))
        pts_w = self.pts_mlp2(pts_w.transpose(1,2))

        pose_feat = torch.cat([rgb_local, pts, pts_local, pts_w, pts_w_local], dim=1)
        pose_feat = self.pose_mlp1(pose_feat)
        pose_global = torch.mean(pose_feat, 2, keepdim=True)
        pose_feat = torch.cat([pose_feat, pose_global.expand_as(pose_feat)], 1)

        rx = F.relu(self.conv1_r(pose_feat))
        tx = F.relu(self.conv1_t(pose_feat))
        cx = F.relu(self.conv1_c(pose_feat))
        sx = F.relu(self.conv1_s(pose_feat))

        rx = F.relu(self.conv2_r(rx))
        tx = F.relu(self.conv2_t(tx))
        cx = F.relu(self.conv2_c(cx))
        sx = F.relu(self.conv2_s(sx))

        rx = self.conv3_r(rx)
        tx = self.conv3_t(tx)
        cx = torch.sigmoid(self.conv3_c(cx))
        sx = self.conv3_s(sx)

        rx = self.adaptive_r(rx).view(bs, self.num_obj, 6)
        r = torch.zeros(bs, self.num_obj, 3, 3).to(rx.device)
        for i in range(self.num_obj):
            r[:, i] = Ortho6d2Mat(rx[:, i, :3].contiguous(), rx[:, i, 3:].contiguous()).view(-1, 3, 3)
        t = self.adaptive_r(tx).view(bs, self.num_obj, 3)
        c = self.adaptive_r(cx).view(bs, self.num_obj, 1)
        s = self.adaptive_r(sx).view(bs, self.num_obj, 3)
        return r, t, s, c

class GenerativeDecoder(nn.Module):
    def __init__(self):
        super(GenerativeDecoder, self).__init__()
        self.dropout = True
        dropout_prob = 0.2
        self.use_tanh = True
        in_ch = 128
        out_ch = 1024
        feat_ch = 512

        print("[DeepSDF MLP-9] Dropout: {}; Do_prob: {}; in_ch: {}; hidden_ch: {}".format(self.dropout, dropout_prob,
                                                                                          in_ch, feat_ch))
        if self.dropout is False:
            self.net1 = nn.Sequential(
                nn.utils.weight_norm(nn.Linear(in_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch - in_ch)),
                nn.ReLU(inplace=True)
            )

            self.net2 = nn.Sequential(
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.Linear(feat_ch, out_ch)
            )
        else:
            self.net1 = nn.Sequential(
                nn.utils.weight_norm(nn.Linear(in_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob, inplace=False),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob, inplace=False),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob, inplace=False),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch - in_ch)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob, inplace=False),
            )

            self.net2 = nn.Sequential(
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob, inplace=False),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob, inplace=False),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob, inplace=False),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob, inplace=False),
                nn.Linear(feat_ch, out_ch)
            )

        num_params = sum(p.numel() for p in self.parameters())
        print('[num parameters: {}]'.format(num_params))

    def forward(self, z):
        in1 = z.float()
        out1 = self.net1(in1.float())
        in2 = torch.cat([out1, in1], dim=-1)
        out2 = self.net2(in2)
        if self.use_tanh:
            out2 = torch.tanh(out2)
        return out2
