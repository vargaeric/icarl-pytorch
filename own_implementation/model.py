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
def get_feature_extraction_layer(model, device, feature_extraction_layer, x, **kwargs):
    output_features = []

    x_dataset = TensorDataset(x)
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


def get_accuracy(model, device, criterion, test_data_loader):
    model.eval()
    matches = 0
    # TODO: make it more efficient
    total = 0
    loss = 0

    with torch.no_grad():
        for feature, target in test_data_loader:
            model.zero_grad()

            original_target = target
            target = make_batch_one_hot(target, 100)

            feature = feature.to(device)
            original_target = original_target.to(device)
            target = target.to(device)

            output = model(feature)
            predicted_target = torch.argmax(output, dim=1)

            matches += (predicted_target == original_target).sum().item()
            total += len(feature)
            loss += criterion(output, target).detach().cpu().item()

    return (matches / total), loss


# def get_accuracy_2(model, device, criterion, test_loader, feature_extraction_layer, class_means):
#     with torch.no_grad():
#         for feature, labels in test_loader:
#             target = make_batch_one_hot(labels, 100)
#
#             feature = feature.to(device)
#             target = target.to(device)
#
#             _, pred, pred_inter = test(model, device, criterion, feature_extraction_layer, feature, target)
#
#             pred_inter = (pred_inter.T / torch.norm(pred_inter.T, dim=0)).T
#
#             # Lines 191-195: Compute score for iCaRL
#             sqd = torch.cdist(class_means[:, :, 0].T, pred_inter)
#             score_icarl = (-sqd).T
#             # Compute score for NCM
#             sqd = torch.cdist(class_means[:, :, 1].T, pred_inter)
#             score_ncm = (-sqd).T
#
#     return score_icarl, score_ncm


def get_accuracy_2(model, device, criterion, test_loader, feature_extraction_layer, class_means):
    stat_ncm = []

    with torch.no_grad():
        for feature, labels in test_loader:
            target = make_batch_one_hot(labels, 100)

            feature = feature.to(device)
            target = target.to(device)

            _, pred, pred_inter = test(model, device, criterion, feature_extraction_layer, feature, target)

            pred_inter = (pred_inter.T / torch.norm(pred_inter.T, dim=0)).T

            tensor = torch.zeros(64, 100)

            for key, value in class_means.items():
                tensor[:, key] = torch.tensor(value)

            sqd = torch.cdist(tensor.T, pred_inter)
            score_ncm = (-sqd).T

            stat_ncm += (
                [ll in best for ll, best in zip(labels, torch.argsort(score_ncm, dim=1)[:, -1:])])

        stat_ncm_numerical = torch.as_tensor([float(int(val_t)) for val_t in stat_ncm])

    return torch.mean(stat_ncm_numerical)
