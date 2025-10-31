import torch
import torch.nn.functional as F

class DiceLoss(torch.nn.Module):
    def __init__(self, 
                 include_background=True, 
                 to_onehot_y=False, 
                 sigmoid=False, softmax=False, 
                 other_act=None, 
                 squared_pred=False, 
                 jaccard=False, 
                 reduction='mean', 
                 smooth_nr=1e-5, smooth_dr=1e-5, 
                 batch=False, weight=None):
        super(DiceLoss, self).__init__()
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.squared_pred = squared_pred
        self.jaccard = jaccard
        self.reduction = reduction
        self.smooth_nr = smooth_nr
        self.smooth_dr = smooth_dr
        self.batch = batch
        self.weight = weight
        if weight is not None:
            self.weight = torch.as_tensor(weight)

    def forward(self, input, target):
        if self.sigmoid:
            input = torch.sigmoid(input)

        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                pass
            else:
                input = torch.softmax(input, dim=1)

        if self.other_act is not None:
            input = self.other_act(input)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                pass
            else:
                target = F.one_hot(target.long(), num_classes=n_pred_ch)
                target = target.permute(0, 3, 1, 2).float()

        if not self.include_background:
            if n_pred_ch == 1:
                pass
            else:
                target = target[:, 1:]
                input = input[:, 1:]

        if target.shape != input.shape:
            raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")

        reduce_axis = list(range(2, len(input.shape)))
        if self.batch:
            reduce_axis = [0] + reduce_axis

        intersection = torch.sum(target * input, dim=reduce_axis)

        if self.squared_pred:
            ground_o = torch.sum(target**2, dim=reduce_axis)
            pred_o = torch.sum(input**2, dim=reduce_axis)
        else:
            ground_o = torch.sum(target, dim=reduce_axis)
            pred_o = torch.sum(input, dim=reduce_axis)

        denominator = ground_o + pred_o

        if self.jaccard:
            denominator = 2.0 * (denominator - intersection)

        f = 1.0 - (2.0 * intersection + self.smooth_nr) / (denominator + self.smooth_dr)

        if self.weight is not None:
            num_of_classes = target.shape[1]
            if self.weight.dim() == 0:
                self.weight = self.weight.expand(num_of_classes)
            elif self.weight.dim() == 1:
                if self.weight.shape[0] != num_of_classes:
                    raise ValueError("the length of the weight sequence should be the same as the number of classes")
            f = f * self.weight.view(-1, 1)

        if self.reduction == 'mean':
            f = torch.mean(f)
        elif self.reduction == 'sum':
            f = torch.sum(f)
        elif self.reduction == 'none':
            pass
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        return f
