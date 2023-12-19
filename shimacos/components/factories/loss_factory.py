import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.autograd import Variable


def ohem_loss(rate, cls_pred, cls_target):
    batch_size = cls_pred.size(0)
    ohem_cls_loss = F.cross_entropy(
        cls_pred, cls_target, reduction="none", ignore_index=-1
    )

    sorted_ohem_loss, idx = torch.sort(ohem_cls_loss, descending=True)
    keep_num = min(sorted_ohem_loss.size()[0], int(batch_size * rate))
    if keep_num < sorted_ohem_loss.size()[0]:
        keep_idx_cuda = idx[:keep_num]
        ohem_cls_loss = ohem_cls_loss[keep_idx_cuda]
    cls_loss = ohem_cls_loss.sum() / keep_num
    return cls_loss


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta + m)
    """

    def __init__(
        self, in_features=2048, num_classes=9514, s=10.0, m=0.5, easy_margin=False
    ):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = num_classes
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(self.out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input, label=None):

        cosine = F.linear(input, F.normalize(self.weight)).float()
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        # phi = cos(theta + theta_m)
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        if label is not None:
            # --------------------------- convert label to one-hot ---------------------------
            # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
            one_hot = torch.zeros(cosine.size(), device="cuda")
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
            output = (one_hot * phi) + (
                (1.0 - one_hot) * cosine
            )  # you can use torch.where if your torch.__version__ is 0.4
            output *= self.s
            # print(output)
            loss = self.loss(output, label)
            return loss, cosine
        else:
            return cosine


class AdaCos(nn.Module):
    def __init__(self, num_features=2048, num_classes=2, m=0.50):
        super(AdaCos, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = math.sqrt(2) * math.log(2)
        self.m = m
        self.W = Parameter(torch.FloatTensor(num_classes, num_features))
        nn.init.xavier_uniform_(self.W)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input, label=None):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)
        if label is None:
            return logits
        # feature re-scale
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        with torch.no_grad():
            B_avg = torch.where(
                one_hot < 1,
                torch.exp(self.s * logits.float()),
                torch.zeros_like(logits).float(),
            )
            B_avg = torch.sum(B_avg) / input.size(0)
            # print(B_avg)
            theta_med = torch.median(theta[one_hot == 1])
            self.s = torch.log(B_avg) / torch.cos(
                torch.min(math.pi / 4 * torch.ones_like(theta_med), theta_med)
            )
        print(self.s)
        output = self.s * logits
        print(output)
        loss = self.loss(output, label.long())

        return loss, output


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class NormalizedMAE(nn.Module):
    def __init__(self, weights=[0.3, 0.175, 0.175, 0.175, 0.175]):
        super(NormalizedMAE, self).__init__()
        self.weights = torch.tensor(weights, requires_grad=False)
        # 訓練データから算出した平均値
        # self.norm = torch.tensor(
        #     [
        #         50.034068314838336,
        #         51.47469215992403,
        #         59.244131909485276,
        #         47.32512986153187,
        #         51.90565840967048,
        #     ],
        #     requires_grad=False,
        # )

    def forward(self, inputs, targets):
        inputs = inputs.double()
        is_nan = torch.isnan(targets)
        inputs[is_nan] = 0
        targets[is_nan] = 0
        diff = torch.abs(inputs - targets).sum(0)
        # norm = (targets != 0).sum(0) * self.norm.to(inputs.device)

        norm = targets.sum(0)
        loss = (diff / norm) * self.weights.to(inputs.device)
        return loss.sum() / self.weights.to(inputs.device).sum()


class RSNALoss(nn.Module):
    def __init__(self):
        super(RSNALoss, self).__init__()
        """
        Labelの順番：
            "pe_present_on_image",
            "negative_exam_for_pe",
            "indeterminate",
            "chronic_pe",
            "acute_and_chronic_pe",
            "central_pe",
            "leftsided_pe",
            "rightsided_pe",
            "rv_lv_ratio_gte_1",
            "rv_lv_ratio_lt_1",
        """
        self.exam_weights_dict = {
            0: 0.0736196319,
            1: 0.09202453988,
            2: 0.1042944785,
            3: 0.1042944785,
            4: 0.1877300613,
            5: 0.06257668712,
            6: 0.06257668712,
            7: 0.2346625767,
            8: 0.0782208589,
        }
        self.exam_weights_sum = sum(self.exam_weights_dict.values())
        self.image_weights = 0.0736196319
        self.bce = nn.BCELoss(reduction="none")
        # self.bce_ = nn.BCELoss(reduction="none")

    def forward(self, pred, label, seq_lens):
        """
        :input param
          pred: shape[B, L, C]
          label: shape[B, L, C]
        """
        seq_lens = list(seq_lens.detach().cpu().numpy())
        # Exam level
        exam_pred = pred[..., 1:]
        exam_label = label[..., 1:]
        total_weights = 0
        exam_loss = 0
        for i in range(exam_pred.shape[-1]):
            for j in range(len(seq_lens)):
                exam_loss += (
                    self.bce(
                        exam_pred[j, : seq_lens[j], i].mean(),
                        exam_label[j, : seq_lens[j], i].mean(),
                    )
                    * self.exam_weights_dict[i]
                ).sum()
                total_weights += self.exam_weights_dict[i]

        image_loss = 0
        image_exam_weights = []
        for j in range(len(seq_lens)):
            image_exam_weight = self.image_weights * label[j, : seq_lens[j], 0].mean()
            image_exam_weights.append(image_exam_weight)
            image_loss += (
                self.bce(pred[j, : seq_lens[j], 0], label[j, : seq_lens[j], 0])
                * image_exam_weight
            ).sum()
            total_weights += image_exam_weight * seq_lens[j]
        main_loss = exam_loss + image_loss
        main_loss /= total_weights
        return main_loss


class RSNASplitLoss(nn.Module):
    def __init__(self):
        super(RSNASplitLoss, self).__init__()
        """
        Labelの順番：
            "pe_present_on_image",
            "negative_exam_for_pe",
            "indeterminate",
            "chronic_pe",
            "acute_and_chronic_pe",
            "central_pe",
            "leftsided_pe",
            "rightsided_pe",
            "rv_lv_ratio_gte_1",
            "rv_lv_ratio_lt_1",
        """
        self.weight = nn.Parameter(
            torch.tensor(
                [
                    0.0736196319,
                    0.09202453988,
                    0.1042944785,
                    0.1042944785,
                    0.1877300613,
                    0.06257668712,
                    0.06257668712,
                    0.2346625767,
                    0.0782208589,
                ]
            ),
            requires_grad=False,
        )
        self.image_weights = 0.0736196319
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        # self.bce_ = nn.BCELoss(reduction="none")

    def forward(self, pred, batch):
        """
        :input param
          pred: shape[B, L, C]
          label: shape[B, L, C]
        """
        seq_lens = list(batch["seq_len"].detach().cpu().numpy())
        label = batch["label"].float()
        image_weight = batch["image_weight"].float() + self.image_weights
        # Exam level
        image_pred, exam_pred = pred
        exam_label, _ = label[..., 1:].max(1)
        exam_loss = (self.bce(exam_pred, exam_label) * self.weight).sum()
        total_weights = self.weight.sum() * exam_pred.shape[0]
        image_loss = 0
        for j in range(len(seq_lens)):
            image_exam_weight = (
                self.image_weights * image_weight[j, : seq_lens[j]].mean()
            )
            image_loss += (
                self.bce(image_pred[j, : seq_lens[j], 0], label[j, : seq_lens[j], 0])
                * image_exam_weight
            ).sum()
            total_weights += image_exam_weight * seq_lens[j]
        main_loss = exam_loss + image_loss
        main_loss /= total_weights
        return main_loss


class RSNASplitClassLoss(nn.Module):
    def __init__(self):
        super(RSNASplitClassLoss, self).__init__()
        """
        Labelの順番：
            "pe_present_on_image",
            "pe",
            "location",
            "rv_lv",
            "pe_type",
        """
        self.pe_weight = nn.Parameter(
            torch.tensor([0.0736196319, 0.0736196319, 0.09202453988]),
            requires_grad=False,
        )
        self.rv_lv_weight = nn.Parameter(
            torch.tensor([0.2346625767, 0.0782208589, 0.0736196319]),
            requires_grad=False,
        )
        self.pe_type_weight = nn.Parameter(
            torch.tensor([0.1042944785, 0.1877300613, 0.1042944785, 0.0736196319]),
            requires_grad=False,
        )
        self.location_weight = nn.Parameter(
            torch.tensor([0.1877300613, 0.06257668712, 0.06257668712]),
            requires_grad=False,
        )

        self.image_weights = 0.0736196319
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.pe_ce = nn.CrossEntropyLoss(reduction="none", weight=self.pe_weight)
        self.rv_lv_ce = nn.CrossEntropyLoss(reduction="none", weight=self.rv_lv_weight)
        self.pe_type_ce = nn.CrossEntropyLoss(
            reduction="none", weight=self.pe_type_weight
        )

    def forward(self, pred, batch):
        """
        :input param
          pred: shape[B, L, 14]
          label: shape[B, L, 6]
        """
        seq_lens = list(batch["seq_len"].detach().cpu().numpy())
        label = batch["convert_label"].long()
        image_weight = batch["image_weight"].float() + self.image_weight
        # Exam level
        image_pred, exam_pred = pred
        exam_label, _ = label[..., 1:].max(1)
        pe_pred = exam_pred[:, :3]
        location_pred = exam_pred[:, 3:6]
        rv_lv_pred = exam_pred[:, 6:9]
        pe_type_label = exam_pred[:, 9:13]
        # Loss
        exam_loss = (self.pe_ce(pe_pred, exam_label[:, 1])).sum()
        exam_loss += (
            self.bce(location_pred, exam_label[:, 1:4].float()) * self.location_weight
        ).sum()
        exam_loss += self.rv_lv_ce(rv_lv_pred, exam_label[:, 4]).sum()
        exam_loss += self.pe_type_ce(pe_type_label, exam_label[:, 5]).sum()

        total_weights = (
            self.pe_weight.sum()
            + self.rv_lv_weight.sum()
            + self.pe_type_weight.sum()
            + self.location_weight.sum()
        ) * exam_pred.shape[0]
        image_loss = 0
        for j in range(len(seq_lens)):
            image_exam_weight = (
                self.image_weights * image_weight[j, : seq_lens[j]].mean()
            )
            image_loss += (
                self.bce(
                    image_pred[j, : seq_lens[j], 0], label[j, : seq_lens[j], 0].float()
                )
                * image_exam_weight
            ).sum()
            total_weights += image_exam_weight * seq_lens[j]
        main_loss = exam_loss + image_loss
        main_loss /= total_weights
        return main_loss


class NormalRSNALoss(nn.Module):
    def __init__(self):
        super(NormalRSNALoss, self).__init__()
        """
        Labelの順番：
            "pe_present_on_image",
            "negative_exam_for_pe",
            "indeterminate",
            "chronic_pe",
            "acute_and_chronic_pe",
            "central_pe",
            "leftsided_pe",
            "rightsided_pe",
            "rv_lv_ratio_gte_1",
            "rv_lv_ratio_lt_1",
        """
        self.weights = torch.tensor(
            [
                0.0736196319,
                0.0736196319,
                0.09202453988,
                0.1042944785,
                0.1042944785,
                0.1877300613,
                0.06257668712,
                0.06257668712,
                0.2346625767,
                0.0782208589,
            ]
        )
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, pred, batch):
        """
        :input param
          pred: shape[B, L, C]
          label: shape[B, L, C]
        """
        seq_lens = list(batch["seq_len"].detach().cpu().numpy())
        label = batch["label"].float()
        # self.weights = self.weights.cuda()
        # Exam level
        loss = 0
        image_pred = pred[:, :3]
        image_label = label[:, :3].clone()
        image_label[torch.where(image_label.sum(1) == 0)[0]] = (
            torch.tensor([0, 1, 0]).float().to(label.device)
        )
        image_label = torch.where(image_label == 1)[1]
        loss += self.cross_entropy(image_pred, image_label)
        loss += self.bce(pred[:, 3:], label[:, 3:])

        return loss


class ClassificationLoss(nn.Module):
    def __init__(self):
        super(ClassificationLoss, self).__init__()
        self.bce = nn.BCELoss(reduction="mean")
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")
        self.image_weights = 0.0736196319

    def forward(self, pred, label, seq_lens):
        """
        :input param:
          pred: shape[B, L, C]
          label: shape[B, L, C]
        """
        if len(pred.shape) == 3:
            seq_lens = list(seq_lens.detach().cpu().numpy())
            # Exam level
            image_loss = 0
            total_weights = 0
            for j in range(len(seq_lens)):
                image_exam_weight = self.image_weights * label[j, : seq_lens[j]].mean()
                image_loss += (
                    self.bce(pred[j, : seq_lens[j]], label[j, : seq_lens[j]])
                ).sum()
                total_weights += image_exam_weight * seq_lens[j]
            image_loss /= len(seq_lens)
        else:
            label = torch.where(label == 1)[1]
            image_loss = self.cross_entropy(pred, label)
        return image_loss


def get_weighted_binary_cross_entropy():
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([0.2]))
    return loss_fn


def get_focalloss():
    loss_fn = FocalLoss()
    return loss_fn


def get_adacos():
    loss_fn = AdaCos()
    return loss_fn


def get_arcface(in_features, num_classes):
    loss_fn = ArcMarginProduct(in_features=in_features, num_classes=num_classes)
    return loss_fn


def get_binary_cross_entropy():
    loss_fn = nn.BCEWithLogitsLoss()
    return loss_fn


def get_any_binary_cross_entropy():
    loss_fn = nn.BCEWithLogitsLoss()
    return loss_fn


def get_weighted_cross_entropy():
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    return loss_fn


def get_classification_loss():
    loss_fn = ClassificationLoss()
    return loss_fn


def get_cross_entropy():
    loss_fn = nn.CrossEntropyLoss()
    return loss_fn


def get_rsna_loss():
    loss_fn = NormalRSNALoss()
    return loss_fn


def get_rsna_split_loss():
    loss_fn = RSNASplitLoss()
    return loss_fn


def get_rsna_split_class_loss():
    loss_fn = RSNASplitClassLoss()
    return loss_fn


def get_mse(weights=None):
    loss_fn = nn.MSELoss()
    return loss_fn


def get_loss(loss_class, **params):
    print("loss class:", loss_class)
    if "." in loss_class:
        obj = eval(loss_class.split(".")[0])
        attr = loss_class.split(".")[1]
        f = getattr(obj, attr)
    else:
        f = globals().get(loss_class)
    return f(**params)
