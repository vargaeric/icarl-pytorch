import torch
from torch.utils.data import TensorDataset, DataLoader


# TODO: rewrite this function!!!!
def extract_features_from_layer(model, layer_name, x):
    activation = {}

    def get_activation(name):
        def hook(model_hook, x_hook, out_hook):
            activation[name] = out_hook.detach().cpu()

        return hook

    model.eval()

    with torch.no_grad():
        with getattr(model, layer_name).register_forward_hook(get_activation(layer_name)):
            output = model(x)

    return output, activation[layer_name]


def train(model, device, criterion, optimizer, feature, target):
    model.train()
    model.zero_grad()

    feature = feature.to(device)
    target = target.to(device)
    output = model(feature)
    loss = criterion(output, target)

    loss.backward()
    optimizer.step()

    # TODO verifiy this if it can not be simplified
    return loss.detach().cpu().item()


# TODO: rewrite!
# TODO: rename feature_extraction_layer
def test(model, device, criterion, feature_extraction_layer, feature, target):
    model.eval()

    with torch.no_grad():
        feature = feature.to(device)
        target = target.to(device)
        output, output_features = extract_features_from_layer(model, feature_extraction_layer, feature)
        loss = criterion(output, target)

    return loss.detach().cpu().item(), output, output_features


# TODO: rewrite!
def inference(model, device, feature):
    model.eval()

    with torch.no_grad():
        feature = feature.to(device)
        output = model(feature)

    return output


# TODO: rewrite!!!!!!!!!!!
def get_feature_extraction_layer(model, device, feature_extraction_layer, target, **kwargs):
    output_features = []

    x_dataset = TensorDataset(target)
    x_dataset_loader = DataLoader(x_dataset, **kwargs)

    model.eval()
    with torch.no_grad():
        for (patterns,) in x_dataset_loader:
            if device is not None:
                patterns = patterns.to(device)
            output_features.append(extract_features_from_layer(model, feature_extraction_layer, patterns)[1])

    return torch.cat(output_features)


# TODO: rewrite it and move it from here!!!:
def make_batch_one_hot(input_tensor, n_classes, dtype=torch.float):
    targets = torch.zeros(input_tensor.shape[0], n_classes, dtype=dtype)
    targets[range(len(input_tensor)), input_tensor.long()] = 1
    return targets