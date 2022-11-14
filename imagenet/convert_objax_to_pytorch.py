import torch

def load_resnet18_from_objax(model, objax_model_weight, finetune_cut_last_layer=True):
    named_weights = dict(model.named_parameters())

    layer_id = 0
    for name, param in model.named_parameters():
        weight = torch.from_numpy(objax_model_weight[str(layer_id)]).cuda()
        if "conv" in name or "downsample" in name:
            weight = weight.permute((3, 2, 0, 1))
        if "gn" in name:
            weight = weight.view(weight.shape[1])
        if "fc" in name:
            if finetune_cut_last_layer:
                layer_id += 1
                continue
        param.data = weight.contiguous()
        layer_id += 1

    return model