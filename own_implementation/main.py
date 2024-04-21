import time

import torch
from torch.utils.data import DataLoader
from torchvision import models

from data import get_data_per_tasks
from icarl_net import make_icarl_net, initialize_icarl_net
from torch.nn import BCELoss
from torch.optim.lr_scheduler import MultiStepLR

from model import make_batch_one_hot, train, get_accuracy, inference, get_feature_extraction_layer, get_accuracy_2

import numpy as np
from sklearn.cluster import KMeans

# TODO: all with uppercase because they are constants
batch_size = 128
tasks_nr = 10
exemplars_per_class = 20  # Number of prototypes per class at the end: total protoset memory/ total number of classes
epochs = 70
lr_start = 2.
lr_milestones = [49, 63]
lr_factor = 5.
# lr_gamma = 1.0/5.
weight_decay = 0.00001
momentum = 0.9
seed = 1993
class_order = [87, 0, 52, 58, 44, 91, 68, 97, 51, 15,
               94, 92, 10, 72, 49, 78, 61, 14, 8, 86,
               84, 96, 18, 24, 32, 45, 88, 11, 4, 67,
               69, 66, 77, 47, 79, 93, 29, 50, 57, 83,
               17, 81, 41, 12, 37, 59, 25, 20, 80, 73,
               1, 28, 6, 46, 62, 82, 53, 9, 31, 75,
               38, 63, 33, 74, 27, 22, 36, 3, 16, 21,
               60, 19, 70, 90, 89, 43, 5, 42, 65, 76,
               40, 30, 23, 85, 2, 95, 56, 48, 71, 64,
               98, 13, 99, 7, 34, 55, 54, 26, 35, 39]

torch.manual_seed(seed)

train_data_per_tasks = None
test_data_per_tasks = None


class ModelWrapper(torch.nn.Module):
    def __init__(self, device, model):
        super(ModelWrapper, self).__init__()

        self.features = torch.nn.Sequential(*list(model.children())[:-1])

        num_features = model.fc.in_features

        self.fc = torch.nn.Linear(num_features, 100)
        self.model = model

        self.model.to(device)
        self.fc.to(device)

    def forward(self, x, extract_features=False):
        x = self.features(x)
        x = torch.flatten(x, 1)

        if extract_features:
            return x
        else:
            return self.fc(x)


training_data_grouped_by_classes = {}
class_means = {}


def group_training_data_by_classes(current_original_train_data):
    # TODO: maybe add only the features
    global training_data_grouped_by_classes

    for current_original_train_data in current_original_train_data:
        feature, target = current_original_train_data

        # print(type(feature))
        # print(type(target))

        if target in training_data_grouped_by_classes:
            training_data_grouped_by_classes[target].append([feature, target])
        else:
            training_data_grouped_by_classes[target] = [[feature, target]]


def get_current_task_classes(task_nr):
    return class_order[task_nr * tasks_nr:(task_nr + 1) * tasks_nr]


def l2_normalization(vector):
    return torch.nn.functional.normalize(vector, p=2, dim=0)


def select_exemplars_with_kmeans(model, device, dataset, n_clusters=exemplars_per_class):
    global exemplars_means

    new_memory_dataset = []
    exemplars_per_label = {}

    # Step 1: Extract features for the dataset
    model.eval()

    with torch.no_grad():
        for data, label in dataset:
            if label not in exemplars_per_label:
                exemplars_per_label[label] = [data]
            else:
                exemplars_per_label[label].append(data)

    # Step 2 and 3: Apply K-Means on normalized features
    for label, exemplars in exemplars_per_label.items():
        exemplars_tensor = torch.stack(exemplars).to(device)
        # features = model(exemplars_tensor, extract_features=True)

        features = get_feature_extraction_layer(
            model,
            device,
            'feature_extractor',
            exemplars_tensor,
        )

        normalized_features = [l2_normalization(feature) for feature in features]
        features_np = torch.stack(normalized_features).cpu().detach().numpy()

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(features_np)
        centers = kmeans.cluster_centers_

        # Step 4: Select nearest exemplars to cluster centers
        selected_exemplars = []

        for center in centers:
            distances = np.linalg.norm(features_np - center, axis=1)
            nearest_index = np.argmin(distances)
            selected_exemplars.append(exemplars[nearest_index])

        # Normalization and mean calculation for the selected exemplars
        # selected_features = model(torch.stack(selected_exemplars).to(device), extract_features=True)
        selected_features = get_feature_extraction_layer(
            model,
            device,
            'feature_extractor',
            torch.stack(selected_exemplars).to(device),
        )

        normalized_selected_features = [l2_normalization(feature) for feature in selected_features]
        class_means[label] = l2_normalization(torch.stack(normalized_selected_features).mean(dim=0))

        # Update memory dataset
        for data in selected_exemplars:
            new_memory_dataset.append((data, label))

    return new_memory_dataset


def define_class_means(model, device, task_nr, current_original_train_data, herding=True):
    global training_data_grouped_by_classes

    current_task_classes = get_current_task_classes(task_nr)

    selected_exemplars = []

    for current_task_class in current_task_classes:
        print(f"Selection exemplars for {current_task_class}...")

        original_data = training_data_grouped_by_classes[current_task_class]
        features = [item[0] for item in training_data_grouped_by_classes[current_task_class]]
        features_tensor = torch.stack(features)

        extracted_features_from_last_layer = get_feature_extraction_layer(
            model,
            device,
            'feature_extractor',
            features_tensor,
        )
        D = extracted_features_from_last_layer.T
        D = D / torch.norm(D, dim=0)

        mu = torch.mean(D, dim=1)

        # Storing or using the index
        class_means[current_task_class] = mu  # Store the mean

        if herding:
            D_transposed = D.T  # Correct the orientation
            similarity = torch.mv(D_transposed, mu)  # Corrected Matrix-vector multiplication

            top_k_values, top_k_indices = torch.topk(similarity, exemplars_per_class)

            selected_exemplars_for_class = [original_data[idx] for idx in top_k_indices.tolist()]

            selected_exemplars += selected_exemplars_for_class

    if not herding:
        selected_exemplars += select_exemplars_with_kmeans(model, device, current_original_train_data)

    return selected_exemplars


def main():
    global train_data_per_tasks, test_data_per_tasks

    train_data_per_tasks, test_data_per_tasks = get_data_per_tasks()

    model = make_icarl_net(num_classes=100)
    model.apply(initialize_icarl_net)
    old_model = None

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    # model = ModelWrapper(device, models.resnet18(weights=None))
    model = model.to(device)

    # TODO: rename criterion everywhere loss_function (or loss_fn)
    criterion = BCELoss()

    # TODO: why not just use lr_start?
    sh_lr = lr_start

    current_test_data = []
    exemplars = []

    for task_nr in range(tasks_nr):
        print(f"Task {task_nr}:")

        current_original_train_data = train_data_per_tasks[task_nr]
        current_train_data = train_data_per_tasks[task_nr]

        if task_nr != 0:
            # print(type(current_train_data))
            # print(type(exemplars))
            # print(type(current_train_data[0]))
            # print(type(exemplars[0]))
            # print(type(current_train_data[0][0]))
            # print(type(exemplars[0][0]))
            # print(type(current_train_data[0][1]))
            # print(type(exemplars[0][1]))

            current_train_data += exemplars

        print(f"Current training data length: {len(current_train_data)}")

        current_test_data += test_data_per_tasks[task_nr]

        train_data_loader = DataLoader(current_train_data, batch_size=batch_size, shuffle=True)
        # TODO: wtf????
        test_loader = DataLoader(current_test_data, batch_size=batch_size, shuffle=True)

        # TODO: Maybe moves these 3 lines out of task_nr
        optimizer = torch.optim.SGD(model.parameters(), lr=sh_lr, weight_decay=weight_decay, momentum=momentum)
        # TODO: simplify gamma
        scheduler = MultiStepLR(optimizer, lr_milestones, gamma=1.0 / lr_factor)

        # TODO: rewrite this!!!

        model.train()

        for epoch in range(epochs):
            total_loss = 0
            start_time = time.time()

            print(f"Epoch {epoch}: ", end='')

            for feature, target in train_data_loader:
                target = make_batch_one_hot(target, 100)

                # TODO: Maybe we dont need this
                feature = feature.to(device)
                target = target.to(device)

                if task_nr != 0:
                    predicted_target = inference(old_model, device, feature)
                    previous_classes = class_order[:(task_nr * tasks_nr)]
                    target[:, previous_classes] = predicted_target[:, previous_classes]

                total_loss += train(model, device, criterion, optimizer, feature, target)

            accuracy, loss = get_accuracy(model, device, criterion, test_loader)

            print(f'train loss - {total_loss} | val loss - {loss} | accuracy - {accuracy}')

            scheduler.step()

        # TODO: Rewrite!!!!
        if task_nr == 0:
            old_model = make_icarl_net(100)
            old_model = old_model.to(device)

        old_model.load_state_dict(model.state_dict())

        group_training_data_by_classes(current_original_train_data)

        new_exemplars = define_class_means(model, device, task_nr, current_original_train_data, False)

        print(f"New exemplars nr:{len(new_exemplars)}")

        ncm = get_accuracy_2(model, device, criterion, test_loader, 'feature_extractor', class_means)

        print('NCM: ', ncm)

        exemplars += new_exemplars

    print('You will succeed!!!!! yuhhuuu')


if __name__ == '__main__':
    main()
