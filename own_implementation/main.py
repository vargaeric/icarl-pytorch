import time

import torch
from torch.utils.data import DataLoader

from data import get_data_per_tasks
from icarl_net import make_icarl_net, initialize_icarl_net
from torch.nn import BCELoss
from torch.optim.lr_scheduler import MultiStepLR

from model import make_batch_one_hot, train

# TODO: all with uppercase because they are constants
batch_size = 128
tasks_nr = 10  # Classes per group
exemplars_per_class = 20  # Number of prototypes per class at the end: total protoset memory/ total number of classes
epochs = 70
lr_start = 2.
lr_milestones = [49, 63]
lr_factor = 5.
# lr_gamma = 1.0/5.
weight_decay = 0.00001
momentum = 0.9
seed = 1993

torch.manual_seed(seed)

train_data_per_tasks = None
test_data_per_tasks = None


def main():
    global train_data_per_tasks, test_data_per_tasks

    train_data_per_tasks, test_data_per_tasks = get_data_per_tasks()

    model = make_icarl_net(num_classes=100)
    model.apply(initialize_icarl_net)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)

    # TODO: rename criterion everywhere loss_function (or loss_fn)
    criterion = BCELoss()

    # TODO: why not just use lr_start?
    sh_lr = lr_start

    current_test_data = []
    exemplars = []

    for task_nr in range(tasks_nr):
        print(f"Task {task_nr}:")

        current_train_data = train_data_per_tasks[task_nr]

        if task_nr != 0:
            current_train_data += exemplars

        current_test_data += test_data_per_tasks[task_nr]
        train_data_loader = DataLoader(current_train_data, batch_size=batch_size, shuffle=True)

        # TODO: Maybe moves these 3 lines out of task_nr
        optimizer = torch.optim.SGD(model.parameters(), lr=sh_lr, weight_decay=weight_decay, momentum=momentum)
        # TODO: simplify gamma
        scheduler = MultiStepLR(optimizer, lr_milestones, gamma=1.0/lr_factor)

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

                if task_nr == 0:
                    total_loss += train(model, device, criterion, optimizer, feature, target)
                else:
                    print('To be continued!!!')


            print(f'total loss - {total_loss}')




    print('You will succeed!!!!! yuhhuuu')


if __name__ == '__main__':
    main()
