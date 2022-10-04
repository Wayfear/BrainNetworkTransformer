from typing import List

import torch


def accuracy(output: torch.Tensor, target: torch.Tensor, top_k=(1,)) -> List[float]:
    """Computes the precision@k for the specified values of k"""
    max_k = max(top_k)
    batch_size = target.size(0)

    _, predict = output.topk(max_k, 1, True, True)
    predict = predict.t()
    correct = predict.eq(target.view(1, -1).expand_as(predict))

    res = []
    for k in top_k:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res


def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False