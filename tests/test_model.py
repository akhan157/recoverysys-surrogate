import torch
from surrogate.model import SurrogateNet, INPUT_FEATURES, OUTPUT_FEATURES


def test_forward_shape():
    model = SurrogateNet()
    x = torch.randn(32, len(INPUT_FEATURES))
    y = model(x)
    assert y.shape == (32, len(OUTPUT_FEATURES))


def test_single_sample():
    model = SurrogateNet()
    model.eval()
    x = torch.randn(1, len(INPUT_FEATURES))
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, len(OUTPUT_FEATURES))
    assert not torch.isnan(y).any()
