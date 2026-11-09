import torch


def make_pair(repetitions, sigma_map):
    """Average random, non-empty, disjoint groups from [N, C, H, W] observations."""
    if repetitions.ndim != 4 or repetitions.shape[0] < 2:
        raise ValueError("Expected at least two repetitions shaped [N, C, H, W]")
    count = repetitions.shape[0]
    while True:
        labels = torch.randint(0, 3, (count,))
        input_ids = torch.where(labels == 0)[0].to(repetitions.device)
        target_ids = torch.where(labels == 1)[0].to(repetitions.device)
        if input_ids.numel() and target_ids.numel():
            break

    input_image = repetitions[input_ids].mean(dim=0, keepdim=True)
    target_image = repetitions[target_ids].mean(dim=0, keepdim=True)
    if sigma_map.shape[0] == count:
        input_sigma = sigma_map[input_ids].square().sum(dim=0, keepdim=True).sqrt() / input_ids.numel()
    else:
        input_sigma = sigma_map / input_ids.numel() ** 0.5
    return input_image, target_image, input_sigma
