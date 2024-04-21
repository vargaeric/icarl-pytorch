from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms


def get_data_per_tasks():
    tasks_nr = 10
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

    all_train_data = CIFAR100(
        root='../data/cifar100',
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    all_test_data = CIFAR100(
        root='../data/cifar100',
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    train_data_per_tasks = [[] for _ in range(tasks_nr)]  # task numbers
    test_data_per_tasks = [[] for _ in range(tasks_nr)]  # task numbers

    print('Grouping training data per tasks...')

    # TODO: maybe they shouldn't be tuples

    for (feature, target) in all_train_data:
        target_index_in_class_order = class_order.index(target)
        task_nr = int(target_index_in_class_order / 10)  # 10 is the tasks_nr number?
        train_data_per_tasks[task_nr].append((feature, target))

    print('Grouping testing data per tasks...')

    for (feature, target) in all_test_data:
        target_index_in_class_order = class_order.index(target)
        task_nr = int(target_index_in_class_order / 10)  # 10 is the tasks_nr number?
        test_data_per_tasks[task_nr].append((feature, target))

    # selected_task_nr = 3
    #
    # print('\nTraining data:')
    # print(len(train_data_per_tasks[selected_task_nr]))
    #
    # for i in range(70):
    #     feature, target = train_data_per_tasks[selected_task_nr][i]
    #
    #     print(target, end=' ')
    #
    # print('\n\nTesting data:')
    # print(len(test_data_per_tasks[selected_task_nr]))
    #
    # for i in range(70):
    #     feature, target = test_data_per_tasks[selected_task_nr][i]
    #
    #     print(target, end=' ')

    return train_data_per_tasks, test_data_per_tasks
