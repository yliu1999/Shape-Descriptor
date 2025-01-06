import os
from unicodedata import category
import numpy as np
# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
import itertools
import open3d

from trainers.base_trainer import BaseTrainer
import toolbox.lr_scheduler

class Trainer(BaseTrainer):
    def __init__(self, cfg, args, device):
        super(BaseTrainer, self).__init__()
        self.cfg = cfg
        self.args = args
        self.device = device

        ae_lib = importlib.import_module(cfg.models.ae.type)  # models.gcn3d.gcn3d
        self.ae = ae_lib.get_model(self.cfg.models.ae)
        self.loss_label = ae_lib.get_loss(self.cfg.trainer.loss_label)
        # print(self.device)
        self.ae.to(self.device)
        # print("ae:")
        # print(self.ae)

        self.optim_ae, self.lrscheduler_ae = self._get_optim(self.ae.parameters(), self.cfg.trainer.optim_ae)

        self.additional_log_info = {}
    
    # Init training-specific contexts
    def prep_train(self):
        self.train()
        
    def _get_optim(self, parameters, cfg):
        if cfg.type.lower() == "adam":
            optim = torch.optim.Adam(parameters, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, weight_decay=cfg.weight_decay, amsgrad=False)
        elif cfg.type.lower() == "sgd":
            optim = torch.optim.SGD(parameters, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
        else:
            raise NotImplementedError("Unknow optimizer: {}".format(cfg.type))
        
        scheduler = None
        if hasattr(cfg, 'lr_scheduler'):
            scheduler = getattr(toolbox.lr_scheduler, cfg.lr_scheduler.type)(cfg.lr_scheduler)
        return optim, scheduler
    
    def _step_lr(self, epoch):
        lr_ae = self.lrscheduler_ae(epoch)
        for g in self.optim_ae.param_groups:
            g['lr'] = lr_ae
        self.additional_log_info['epoch'] = epoch
        self.additional_log_info['lr'] = lr_ae
        # print('Step LR: ae'.format(lr_ae))
    
    def _forward_ae(self, p, onehot):
        # p = p.permute(0,2,1)
        pred = self.ae(p, onehot)
        return pred  # pred:[bs, 1024, 257]
        
    def epoch_start(self, epoch):
        # Setting LR
        self.train()
        self._step_lr(epoch)
    
    def pc_normalize(self, pc):
        return pc
        # centroid = torch.mean(pc, dim=-2, keepdims=True)
        # print('center', centroid.shape)
        # pc = pc - centroid
        # return pc
        # l = torch.sqrt(torch.sum(pc**2, dim=-1, keepdims=True))
        # m,_ = torch.max(l, dim=-2,keepdims=True)
        # pc = pc / m
        # # print(m)
        # return pc

    def step(self, data):
        input_pcd = data['points'].to(self.device, non_blocking=True).float()
        gt_label = data['points_label'].to(self.device, non_blocking=True).long()  # [bs, 6, 1024]
        onehot = data['onehot'].to(self.device, non_blocking=True).long()
        cate_sym = data['cate_sym'].to(self.device, non_blocking=True).long()
        category = data['category'].to(self.device, non_blocking=True).long()

        self.optim_ae.zero_grad()

        # #  # test --------------------------------------------------------------------------
        # pcd = input_pcd[0, :, :]
        # import numpy as np
        # import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D
        # point_cloud_tensor = pcd.reshape(1024, 3)
        # point_cloud_np = point_cloud_tensor.cpu().numpy()
        # # 获取每个坐标轴上的数据
        # x = point_cloud_np[:, 0]
        # y = point_cloud_np[:, 1]
        # z = point_cloud_np[:, 2]
        #
        # # 创建一个3D图形对象
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        #
        # # 绘制点云
        # ax.scatter(x, y, z, s=1)  # s表示点的大小
        #
        # # 设置坐标轴标签
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        #
        # # 显示图形
        # plt.show()
        # ---------------------------------------------------------

        pred_label = self._forward_ae(input_pcd, onehot)  # input_pcd:[bs, 1024, 3] onehot(category label embedding):[bs, 6]  pred_label:[bs, 1024, 257]

        losses = self.loss_label(pred_label.contiguous(), gt_label.contiguous(), cate_sym, category)
        loss = losses['loss']
        loss.backward()
        # print(loss.item())
        self.optim_ae.step()
        
        log_info = {}
        for k,v in losses.items():
            log_info[k] = v.item()
        log_info.update(self.additional_log_info)
        return log_info
    
    def epoch_end(self, epoch, **kwargs):
        return

    def classification(self, data, onehot):
        input_pcd = data['points'].to(self.device, non_blocking=True).float()
        input_pcd = self.pc_normalize(input_pcd)
        onehot = torch.from_numpy(onehot).cuda().unsqueeze(0)
        
        with torch.no_grad():
            pred_label = self._forward_ae(input_pcd, onehot)

        return pred_label, 0  # [bs, 1024, 257]

    def save(self, epoch, step):
        save_name = "epoch_{}_iters_{}.pth".format(epoch, step)
        path = os.path.join(self.cfg.save_dir, save_name)
        torch.save({
                'trainer_state_dict': self.state_dict(),
                'optim_ae_state_dict': self.optim_ae.state_dict(),
                'epoch': epoch,
                'step': step,
            }, path)
    
    def resume(self, ckpt_path):
        # print('Resuming {}...'.format(ckpt_path))
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.load_state_dict(ckpt['trainer_state_dict'], strict=False)
        return ckpt # ckpt['epoch']
        
