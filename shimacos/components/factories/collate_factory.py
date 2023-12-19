import torch


def text_collate(batch):
    output = {}
    max_length = max([sample["raw_length"] for sample in batch])
    output["id"] = [sample["id"] for sample in batch]
    output["input_ids"] = torch.cat(
        [torch.as_tensor(sample["input_ids"][:, :max_length]) for sample in batch],
        dim=0,
    )
    output["attention_mask"] = torch.cat(
        [torch.as_tensor(sample["attention_mask"][:, :max_length]) for sample in batch],
        dim=0,
    )
    output["extension"] = torch.cat(
        [torch.as_tensor(sample["extension"]) for sample in batch],
        dim=0,
    )
    if "label" in batch[0]:
        output["label"] = torch.cat(
            [torch.as_tensor(sample["label"]) for sample in batch],
            dim=0,
        )
    return output
