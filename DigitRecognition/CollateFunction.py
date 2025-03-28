import torch

def collate_fn(batch):
    batch_filtered = {}
    #if not isinstance(batch, list):
    #batch = list(filter (lambda x:x is not None, batch))
    if len(batch[0]) == 2:
        batch_filtered['features'] = torch.stack([f for f, t in batch if f is not None])
        batch_filtered['targets'] = torch.stack([t for f, t in batch if f is not None])
    else:
        batch_filtered['features'] = torch.stack([f for f in batch if f is not None])
    return batch_filtered
