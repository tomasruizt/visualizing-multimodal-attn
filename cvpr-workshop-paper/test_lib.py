import torch
from getAttentionLib import maxabs_reduction


def test_maxabs_pooling():
    xs = torch.tensor([[1, 1, 3], [-3, -2, -1]])
    assert maxabs_reduction(xs, dim=0).shape == xs.sum(dim=0).shape
    assert maxabs_reduction(xs, dim=0).tolist() == [-3, -2, 3]
    assert maxabs_reduction(xs, dim=1).tolist() == [3, -3]
