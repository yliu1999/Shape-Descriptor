# follow FS-Net
import torch.nn as nn
import model.gcn3d as gcn3d
import torch
import torch.nn.functional as F
from absl import app
import absl.flags as flags

FLAGS = flags.FLAGS

class PclFeats(nn.Module):
    def __init__(self):
        super(PclFeats, self).__init__()
        self.neighbor_num = 10
        self.support_num = 7

        # 3D convolution for point cloud
        self.conv_0 = gcn3d.HSlayer_surface(kernel_num=128, support_num=self.support_num)
        self.conv_1 = gcn3d.HS_layer(128, 128, support_num=self.support_num)
        self.pool_1 = gcn3d.Pool_layer(pooling_rate=4, neighbor_num=4)
        self.conv_2 = gcn3d.HS_layer(128, 256, support_num=self.support_num)
        self.conv_3 = gcn3d.HS_layer(256, 256, support_num=self.support_num)
        self.pool_2 = gcn3d.Pool_layer(pooling_rate=4, neighbor_num=4)
        self.conv_4 = gcn3d.HS_layer(256, 512, support_num=self.support_num)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)

        self.pts_mlp = nn.Sequential(
            nn.Conv1d(1286, 1032, 1),
            nn.ReLU(),
            nn.Conv1d(1032, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 128, 1),
            nn.ReLU(),
        )

    def forward(self,
                vertices: "tensor (bs, vetice_num, 3)",
                cat_id: "tensor (bs, 1)",
                ):
        """
        Return: (bs, vertice_num, class_num)
        """
        bs, vertice_num, _ = vertices.size()
        if cat_id.shape[0] == 1:
            obj_idh = cat_id.view(-1, 1).repeat(cat_id.shape[0], 1)
        else:
            obj_idh = cat_id.view(-1, 1)

        one_hot = torch.zeros(bs, 6).to(cat_id.device).scatter_(1, obj_idh.long(), 1)

        fm_0 = F.relu(self.conv_0(vertices, self.neighbor_num), inplace=True)
        fm_1 = F.relu(self.bn1(self.conv_1(vertices, fm_0, self.neighbor_num).transpose(1, 2)).transpose(1, 2),
                      inplace=True)
        v_pool_1, fm_pool_1 = self.pool_1(vertices, fm_1)
        fm_2 = F.relu(self.bn2(self.conv_2(v_pool_1, fm_pool_1,
                                           min(self.neighbor_num, v_pool_1.shape[1] // 8)).transpose(1, 2)).transpose(1,
                                                                                                                      2),
                      inplace=True)
        fm_3 = F.relu(self.bn3(self.conv_3(v_pool_1, fm_2,
                                           min(self.neighbor_num, v_pool_1.shape[1] // 8)).transpose(1, 2)).transpose(1,
                                                                                                                      2),
                      inplace=True)
        v_pool_2, fm_pool_2 = self.pool_2(v_pool_1, fm_3)
        fm_4 = self.conv_4(v_pool_2, fm_pool_2, min(self.neighbor_num, v_pool_2.shape[1] // 8))
        f_global = fm_4.max(1)[0]  # (bs, f)

        nearest_pool_1 = gcn3d.get_nearest_index(vertices, v_pool_1)
        nearest_pool_2 = gcn3d.get_nearest_index(vertices, v_pool_2)
        fm_2 = gcn3d.indexing_neighbor_new(fm_2, nearest_pool_1).squeeze(2)
        fm_3 = gcn3d.indexing_neighbor_new(fm_3, nearest_pool_1).squeeze(2)
        fm_4 = gcn3d.indexing_neighbor_new(fm_4, nearest_pool_2).squeeze(2)
        one_hot = one_hot.unsqueeze(1).repeat(1, vertice_num, 1)  # (bs, vertice_num, cat_one_hot)

        feat = torch.cat([fm_0, fm_1, fm_2, fm_3, fm_4, one_hot], dim=2)
        final_feat = self.pts_mlp(feat.transpose(1,2))
        return final_feat