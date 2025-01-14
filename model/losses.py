import torch

def SmoothL1Dis(p1, p2, threshold=0.1):
    '''
    p1: b*n*3
    p2: b*n*3
    '''
    
    diff = torch.abs(p1 - p2)
    less = torch.pow(diff, 2) / (2.0 * threshold)
    higher = diff - threshold / 2.0
    dis = torch.where(diff > threshold, higher, less)
    a = torch.sum(dis, dim=1)
    dis = torch.mean(torch.sum(dis, dim=2 if len(p1.shape)==3 else 1))
    return dis


def ChamferDis(p1, p2):
    '''
    p1: b*n1*3
    p2: b*n2*3
    '''
    dis = torch.norm(p1.unsqueeze(2) - p2.unsqueeze(1), dim=3)
    dis1 = torch.min(dis, 2)[0]
    dis2 = torch.min(dis, 1)[0]
    dis = 0.5*dis1.mean(1) + 0.5*dis2.mean(1)
    return dis.mean()


def PoseDis(r1, t1, s1, r2, t2, s2):
    '''
    r1, r2: b*3*3
    t1, t2: b*3
    s1, s2: b*3
    '''
    dis_r = torch.mean(torch.norm(r1 - r2, dim=1))
    dis_t = torch.mean(torch.norm(t1 - t2, dim=1))
    dis_s = torch.mean(torch.norm(s1 - s2, dim=1))
    return dis_r + dis_t + dis_s


def main_PoseDis(r, t, s, c, r2, t2, s2):
    '''
    r1, r2: b*3*3
    t1, t2: b*3
    s1, s2: b*3
    '''
    num_obj = r.size(1)
    bs = r.size(0)
    dis = torch.zeros(bs, num_obj).to(r.device)
    for i in range(num_obj):
        r1 = r[:, i]
        t1 = t[:, i]
        s1 = s[:, i]
        dis_r = torch.mean(torch.norm(r1 - r2, dim=1), dim=1)
        dis_t = torch.norm(t1 - t2, dim=1)
        dis_s = torch.norm(s1 - s2, dim=1)

        dis[:, i] = dis_r + dis_t + dis_s
    return torch.mean(dis), dis
