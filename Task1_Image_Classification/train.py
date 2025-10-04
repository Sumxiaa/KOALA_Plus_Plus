import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, transforms
from training_funcs import train, validate
from models import _resnet18, _resnet50, _wide_resnet_50_2, ResNet50
from models import swin_t
import argparse


from koala import *
from adafisher import *
from adan import *

import random
import numpy as np
import os



def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def Trainer(model, optim_name, dataset_name='cifar10', max_epochs=100, random_seed=0, sigma=0.1, q=0.1, lr=1.0):
    """
    Train and evaluate the model on CIFAR-10 dataset.

    Args:
        model: Model to train
        optim_name: Name of the optimizer to use
        max_epochs: Number of epochs

    Returns:
        train_loss_history: List of training loss per epoch
        val_loss_history: List of validation loss per epoch
        train_top1_history: List of training Top-1 error per epoch
        val_top1_history: List of validation Top-1 error per epoch
        train_top5_history: List of training Top-5 error per epoch
        val_top5_history: List of validation Top-5 error per epoch
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    set_random_seed(random_seed)

    # Configure optimizer
    if optim_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
    elif optim_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=0.0003, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0005)
    elif optim_name == 'KOALA-V':
        optimizer = VanillaKOALA(
            params=model.parameters(),
            sigma=0.1, q=1, r=None, alpha_r=0.9,
            weight_decay=0.0005, lr=0.1
        )
    elif optim_name == 'KOALA-M':
        optimizer = MomentumKOALA(
            params=model.parameters(),
            sw=0.1, sc=0.0, sv=0.1, a=0.9,
            qw=0.01, qv=0.01, r=None, alpha_r=0.9,
            weight_decay=0.0005, lr=1.0
        )
    elif optim_name == 'KOALA-P':
        optimizer = KOALAPlusPlus(
            params=model.parameters(),
            sigma=sigma, q=q, r=None, alpha_r=0.9,
            weight_decay=0.0005, lr=lr
        )
    elif optim_name == 'AdaFisher':
        optimizer = AdaFisher(model, lr=1e-3, beta=0.9, gamma=0.8, Lambda=1e-3, weight_decay=5e-4)
    elif optim_name == 'Adan':
        optimizer = Adan(model.parameters(),lr = 1e-3, betas = (0.02, 0.08, 0.01), weight_decay = 0.02)
    else:
        raise ValueError(f"Unknown optimizer name: {optim_name}")

    gamma = 0.1 if dataset_name == 'cifar10' else 0.2
    # Learning rate scheduler
    if max_epochs == 100:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=gamma)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    elif max_epochs == 200:
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=gamma)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-4)
        # scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_epochs=10, total_epochs=200, eta_min=1e-4)
    else:
        raise ValueError(f"Unknown max_epoch: {max_epochs}")

    # scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-4)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Configure dataset
    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
    )
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    if dataset_name == 'cifar10':
      train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train),
        batch_size=128, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
      val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root="./data", train=False, transform=transform_test),
        batch_size=128, shuffle=False, num_workers=4, pin_memory=True
      )
    elif dataset_name == 'cifar100':
        train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train),
        batch_size=128, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root="./data", train=False, transform=transform_test),
        batch_size=128, shuffle=False, num_workers=4, pin_memory=True
        )
    elif dataset_name == 'imagenet':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225]),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225]),
        ])
        trainset = torchvision.datasets.ImageFolder(
            root=str(root / 'train'),
            transform=transform_train)
        train_sampler = \
            torch.utils.data.distributed.DistributedSampler(
                trainset) if dist else None
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=mini_batch_size,
            shuffle=(train_sampler is None),
            num_workers=num_workers,
            pin_memory=True, sampler=train_sampler)

        testset = torchvision.datasets.ImageFolder(
            root=str(root / 'val'),
            transform=transform_test)

        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=mini_batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)
    else:
      raise ValueError(f"Unknown dataset name: {dataset_name}")
    # Initialize history storage
    train_loss_history, val_loss_history = [], []
    train_top1_history, val_top1_history = [], []
    train_top5_history, val_top5_history = [], []

    # Training Loop
    for epoch in range(max_epochs):
        print(f"\nEpoch {epoch + 1}/{max_epochs}")

        # Train for one epoch
        train_loss, train_top1, train_top5 = train(
            train_loader, model, criterion, optimizer, scheduler, epoch, max_epochs,
            is_koala=("KOALA" in optim_name),
            train_loss_history=train_loss_history,
            train_top1_history=train_top1_history,
            train_top5_history=train_top5_history
        )

        # Validate
        val_loss, val_top1, val_top5 = validate(
            val_loader, model, criterion, epoch, max_epochs,
            val_loss_history=val_loss_history,
            val_top1_history=val_top1_history,
            val_top5_history=val_top5_history
        )

    # ------------------------ #
    # Visualization           #
    # ------------------------ #
    epochs = range(1, max_epochs + 1)
    """
    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss_history, label="Train Loss", marker="o")
    plt.plot(epochs, val_loss_history, label="Validation Loss", marker="s")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Training and Validation Loss ({optim_name})")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Top-1 Error
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_top1_history, label="Train Top-1 Error", marker="o")
    plt.plot(epochs, val_top1_history, label="Validation Top-1 Error", marker="s")
    plt.xlabel("Epochs")
    plt.ylabel("Error (%)")
    plt.title(f"Training and Validation Top-1 Error ({optim_name})")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Top-5 Error
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_top5_history, label="Train Top-5 Error", marker="o")
    plt.plot(epochs, val_top5_history, label="Validation Top-5 Error", marker="s")
    plt.xlabel("Epochs")
    plt.ylabel("Error (%)")
    plt.title(f"Training and Validation Top-5 Error ({optim_name})")
    plt.legend()
    plt.grid(True)
    plt.show()
    """


    print(f"Finished training with {optim_name}.")
    final_val_top1 = val_top1_history[-1]
    # get the best val top-1 error and its corresponding top-5 error
    best_val_top1 = min(val_top1_history)
    best_val_top5 = val_top5_history[val_top1_history.index(best_val_top1)]
    print(f"Best val top1 error: {best_val_top1}")
    print(f"Final val top1 error: {final_val_top1}")
    final_val_top5 = val_top5_history[-1]
    print(f"Best val top5 error: {best_val_top5}")
    print(f"Final val top5 error: {final_val_top5}")
    # val_loss history
    # 确保 logs 文件夹存在
    os.makedirs("logs", exist_ok=True)

    # 构造日志路径
    txt_filename = f"training_log_{optim_name}_dataset{dataset_name}_epochs{max_epochs}_seed{random_seed}_sigma{sigma}_q{q}_lr{lr}.txt"
    txt_path = os.path.join("logs", txt_filename)

    # 写入 txt 文件
    with open(txt_path, "w") as f:
        f.write(f"{'Epoch':<6} {'TrainLoss':<12} {'ValLoss':<12} "
            f"{'TrainTop1':<12} {'ValTop1':<12} "
            f"{'TrainTop5':<12} {'ValTop5':<12}\n")
        for epoch in range(max_epochs):
                f.write(f"{epoch+1:<6} "
                f"{train_loss_history[epoch]:<12.4f} "
                f"{val_loss_history[epoch]:<12.4f} "
                f"{train_top1_history[epoch]:<12.4f} "
                f"{val_top1_history[epoch]:<12.4f} "
                f"{train_top5_history[epoch]:<12.4f} "
                f"{val_top5_history[epoch]:<12.4f}\n")
        f.write("\n")
        f.write(f"Best val top1 error:  {best_val_top1:.4f}\n")
        f.write(f"Final val top1 error: {final_val_top1:.4f}\n")
        f.write(f"Best val top5 error:  {best_val_top5:.4f}\n")
        f.write(f"Final val top5 error: {final_val_top5:.4f}\n")

    print(f"Training log saved to {txt_path}")
    # return train_loss_history, val_loss_history, train_top1_history, val_top1_history, train_top5_history, val_top5_history



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="ResNet18", choices=["ResNet18", "ResNet50", "WideRes", "vit"])
    parser.add_argument('--optimizer', type=str, default="KOALA-P", choices=["SGD", "Adam", "KOALA-V", "KOALA-M", "KOALA-P", "AdaFisher", "Adan"])
    parser.add_argument('--dataset', type=str, default="cifar10", choices=["cifar10", "cifar100", "imagenet"])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sigma', type=float, default=0.1)
    parser.add_argument('--q', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1.0)

    args = parser.parse_args()
    if args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_classes = 100
    elif args.dataset == 'imagenet':
        num_classes = 1000
    # num_classes = 10 if args.dataset == 'cifar10' else 100
    # 模型选择
    if args.model == 'ResNet18':
        model = _resnet18(num_classes=num_classes, is_cifar=True)
    elif args.model == 'ResNet50':
        # model = _resnet50(num_classes=num_classes, is_cifar=True)
        model = ResNet50(num_classes=num_classes)
    elif args.model == 'WideRes':
        model = _wide_resnet_50_2(num_classes=num_classes, is_cifar=True)
    elif args.model == 'vit':
        model = swin_t(num_classes=num_classes)
    

    # 启动训练
    Trainer(
        model=model,
        optim_name=args.optimizer,
        dataset_name=args.dataset,
        max_epochs=args.epochs,
        random_seed=args.seed,
        sigma=args.sigma,
        q=args.q,
        lr=args.lr
    )
