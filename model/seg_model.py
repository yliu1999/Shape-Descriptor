import torch
import yaml
import numpy as np
import torch.nn as nn

import argparse
from trainers.base_trainer import BaseTrainer
import importlib
import toolbox.lr_scheduler
from options import get_parser
args = get_parser()

class Trainer(BaseTrainer):
    def __init__(self, cfg, args, device):
        super(BaseTrainer, self).__init__()
        self.cfg = cfg
        self.args = args
        self.device = device

        ae_lib = importlib.import_module(cfg.models.ae.type)
        self.ae = ae_lib.get_model(self.cfg.models.ae)
        self.loss_label = ae_lib.get_loss(self.cfg.trainer.loss_label)
        self.ae.to(self.device)

        self.optim_ae, self.lrscheduler_ae = self._get_optim(self.ae.parameters(), self.cfg.trainer.optim_ae)

        self.additional_log_info = {}

    def prep_train(self):
        self.train()

    def _get_optim(self, parameters, cfg):
        if cfg.type.lower() == "adam":
            optim = torch.optim.Adam(parameters, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, weight_decay=cfg.weight_decay,
                                     amsgrad=False)
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

    def _forward_ae(self, p, onehot):
        pred = self.ae(p, onehot)
        return pred

    def epoch_start(self, epoch):
        self.train()
        self._step_lr(epoch)

    def pc_normalize(self, pc):
        return pc

    def step(self, data):
        input_pcd = data['points'].to(self.device, non_blocking=True).float()
        gt_label = data['points_label'].to(self.device, non_blocking=True).long()
        onehot = data['onehot'].to(self.device, non_blocking=True).long()
        cate_sym = data['cate_sym'].to(self.device, non_blocking=True).long()
        category = data['category'].to(self.device, non_blocking=True).long()

        self.optim_ae.zero_grad()

        pred_label = self._forward_ae(input_pcd,
                                      onehot)

        losses = self.loss_label(pred_label.contiguous(), gt_label.contiguous(), cate_sym, category)
        loss = losses['loss']
        loss.backward()
        self.optim_ae.step()

        log_info = {}
        for k, v in losses.items():
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

        return pred_label, 0

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
        print('Resuming {}...'.format(ckpt_path))
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.load_state_dict(ckpt['trainer_state_dict'], strict=False)
        return ckpt

def get_args():
    def dict2namespace(config):
        namespace = argparse.Namespace()
        for key, value in config.items():
            if isinstance(value, dict):
                new_value = dict2namespace(value)
            else:
                new_value = value
            setattr(namespace, key, new_value)
        return namespace

    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    return config

def load_model(args, cfg):
    trainer_lib = importlib.import_module(cfg.trainer.type)
    trainer = trainer_lib.Trainer(cfg, args, device)

    if args.pretrained is not None:
        start_epoch = trainer.resume(args.pretrained)

    trainer.eval()

    return trainer

def optimize_shape_ransac(label, center):
    def generate_all_pairs(n):
        x, y = np.meshgrid(range(n), range(n))
        x = x.reshape(-1)
        y = y.reshape(-1)
        valid = (x != y)
        all_pairs = np.stack((x[valid], y[valid]), axis=1)
        return all_pairs

    def compute_invariant_feature(points, pairs):
        vectors = points[:, pairs[:, 1]] - points[:, pairs[:, 0]]
        vectors = vectors.view(vectors.shape[0], -1, vectors.shape[-1])
        vector_norms = torch.norm(vectors, dim=-1)
        vector_norms = vector_norms.unsqueeze(1) * vector_norms.unsqueeze(2)
        vector_dots = vectors @ vectors.permute(0, 2, 1)
        feature = vector_dots / (vector_norms + 1e-8)

        return feature.view(feature.shape[0], -1)
    nlabel = label.shape[0]
    all_pairs = generate_all_pairs(nlabel)
    all_feature = compute_invariant_feature(center.unsqueeze(0), all_pairs)
    return all_feature

def seg_model(data, obj_id):
    points = data['points']
    bs = points.size(0)
    num_segs = 256
    cfg = get_args()
    model = load_model(args, cfg)

    pred,_ = model.classification(data, obj_id)
    pred = torch.exp(pred)

    score, pred_labels = torch.topk(pred, 3, dim=2)
    pred_labels = pred_labels.cpu().numpy()
    pred_label = pred_labels[:, :, 0]
    labels = {}
    for i in range(bs):
        labels[i] = np.unique(pred_label[i])
    target_nlabel = 128
    center_points = {}
    target_features = {}
    all_feature = np.zeros((bs, target_nlabel))
    all_feature = torch.from_numpy(all_feature)
    for i in range(bs):
        label = labels[i]
        nlabel = len(label)
        center_point = np.zeros((nlabel, 3))
        point = points[i].cpu().numpy()
        pred_label_now = pred_label[i]
        for j, label_now in enumerate(label):
            center_point[j] = np.mean(point[pred_label_now == label_now], axis=0)
        center_point = torch.from_numpy(center_point).type(torch.FloatTensor).cuda()

        feature = optimize_shape_ransac(label, center_point)
        if feature.size(1) == 0:
            feature = torch.zeros(1, 128)
        target_features[i] = feature
        linear = nn.Linear(in_features=feature.size(1), out_features=target_nlabel)
        linear = linear.to(feature.device)
        feature = linear(feature)
        all_feature[i] = feature
        center_points[i] = center_point
    all_feature = all_feature.to(device)

    return pred, target_features, all_feature, center_points, labels

