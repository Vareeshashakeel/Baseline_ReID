import torch.nn.functional as F
from loss.softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from loss.triplet_loss import TripletLoss
from loss.center_loss import CenterLoss


def make_loss(num_classes):
    feat_dim = 768
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)
    center_criterion2 = CenterLoss(num_classes=num_classes, feat_dim=3072, use_gpu=True)

    triplet = TripletLoss()
    xent = CrossEntropyLabelSmooth(num_classes=num_classes)

    def loss_func(score, feat, target, target_cam=None):
        if isinstance(score, list):
            id_loss = [xent(scor, target) for scor in score[1:]]
            id_loss = sum(id_loss) / len(id_loss)
            id_loss = 0.25 * id_loss + 0.75 * xent(score[0], target)
        else:
            id_loss = xent(score, target)

        if isinstance(feat, list):
            tri_loss = [triplet(feats, target)[0] for feats in feat[1:]]
            tri_loss = sum(tri_loss) / len(tri_loss)
            tri_loss = 0.25 * tri_loss + 0.75 * triplet(feat[0], target)[0]

            center = center_criterion(feat[0], target)
            centr2 = [center_criterion2(feats, target) for feats in feat[1:]]
            centr2 = sum(centr2) / len(centr2)
            center = 0.25 * centr2 + 0.75 * center
        else:
            tri_loss = triplet(feat, target)[0]
            center = center_criterion(feat, target)

        return id_loss + tri_loss, center

    return loss_func, center_criterion
