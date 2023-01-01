import torch
import torch.nn.functional as F
from torch import nn

from fcos_core.layers import smooth_l1_loss

class FCOS_KTNet(nn.Module):
    def __init__(self, num_convs=2, in_channels=256, num_classes=0.0):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOS_KTNet, self).__init__()

        dis_tower = []
        for i in range(num_convs):
            dis_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            dis_tower.append(nn.GroupNorm(32, in_channels))
            dis_tower.append(nn.ReLU())

        self.add_module('dis_tower', nn.Sequential(*dis_tower))

        self.cls_logits = nn.Linear(in_channels, 1)
        nn.init.normal_(self.cls_logits.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_logits.bias, 0)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # initialization
        for modules in [self.dis_tower]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.loss_fn_no_reduce = nn.BCEWithLogitsLoss(reduction='none')
        self.num_classes = num_classes-1


    def forward(self, feature, target, score_map=None, domain='source', source_feature=None, source_prototype=None):
        assert target == 0 or target == 1 or target == 0.1 or target == 0.9
        assert domain == 'source' or domain == 'target'

        box_cls_map = score_map["box_cls"].clone().sigmoid()
        centerness_map = score_map["centerness"].clone().sigmoid()

        # Compute loss
        dis_feature = self.dis_tower(feature)
        pos_x = centerness_map * dis_feature
        pos_x = self.avgpool(pos_x)
        pos_x = pos_x.view(pos_x.size(0), -1)

        neg_x = (1 - centerness_map) * dis_feature
        neg_x = self.avgpool(neg_x)
        neg_x = neg_x.view(neg_x.size(0), -1)

        x = torch.cat((pos_x, neg_x))
        x = self.cls_logits(x)
        x = x.view(-1)

        target_pos = torch.full((pos_x.shape[0],), 1, dtype=torch.float, device=pos_x.device)
        target_neg = torch.full((neg_x.shape[0],), 0, dtype=torch.float, device=neg_x.device)
        target = torch.cat((target_pos, target_neg))
        loss = self.loss_fn(x, target)

        cls_feature_smooth = []
        for i in range(box_cls_map.shape[1]):
            cls_weight = box_cls_map[:,i:i+1,:].reshape(box_cls_map.size(0),1,-1).sum(dim=2)
            cls_feature_rebu = (dis_feature * box_cls_map[:,i:i+1,:]).reshape(dis_feature.size(0), dis_feature.size(1), -1).sum(dim=2)
            cls_feature_rebu = cls_feature_rebu / cls_weight.repeat(1, cls_feature_rebu.size(-1))
            cls_feature_smooth.append(torch.mean(cls_feature_rebu, dim=0, keepdim=True))
            
        if self.num_classes == 1:
            cls_feature_smooth.extend([torch.mean(pos_x, dim=0, keepdim=True), torch.mean(neg_x, dim=0, keepdim=True)])
        cls_feature_smooth = torch.cat(cls_feature_smooth)
        
        if source_feature is not None:
            cls_feature_smooth = F.normalize(cls_feature_smooth); source_feature = F.normalize(source_feature)
            cross_matrix = torch.matmul(cls_feature_smooth, source_feature.t())
            source_matrix = torch.matmul(source_feature, source_feature.t())
            target_matrix = torch.matmul(cls_feature_smooth, cls_feature_smooth.t())
            
            loss_reg =  smooth_l1_loss(cross_matrix, target_matrix, beta=1, size_average=False) + \
                        smooth_l1_loss(cross_matrix, source_matrix, beta=1, size_average=False) + \
                        smooth_l1_loss(source_matrix, target_matrix, beta=1, size_average=False)
            loss_reg += smooth_l1_loss(cls_feature_smooth, source_feature, beta=1, size_average=False)
            return loss, loss_reg
        return loss, cls_feature_smooth
