import torch


@torch.no_grad()
def predict_multicrop(images, predict):
    """Average final BiMC probabilities, preserving one prediction per image.

    Process one view at a time so the model batch size stays unchanged.
    Single-crop input follows the original inference path.
    """
    if images.ndim == 4:
        return predict(images)
    if images.ndim != 5 or images.shape[1] != 10:
        raise ValueError('Expected [B, C, H, W] or [B, 10, C, H, W]')
    total = None
    for view in images.unbind(dim=1):
        probabilities = predict(view).float()
        total = probabilities if total is None else total + probabilities
    return total / images.shape[1]
