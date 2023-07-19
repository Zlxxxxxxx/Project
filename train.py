import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score, recall_score, accuracy_score
from torchvision import transforms

from dataset import FinCrimesDataset
from model import MLP

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def save_model(model, optimizer, args, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")

def model_eval(dataloader, model, device):
    model.eval()  =
    total_samples = 0
    correct_predictions = 0

    with torch.no_grad():
        for batch in dataloader:
            features, labels = batch
            features = features.type(torch.float)
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            _, predicted_labels = torch.max(outputs, dim=1)

            total_samples += labels.size(0)
            correct_predictions += (predicted_labels == labels).sum().item()

    accuracy = correct_predictions / total_samples
    return accuracy


def train(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

    train_path = 'Data/train.csv'
    val_path = 'Data/validation.csv'

    train_dataset = FinCrimesDataset(train_path)
    val_dataset = FinCrimesDataset(val_path)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

    input_size = 4
    output_size = 3
    model = MLP(input_size, output_size)
    model = model.to(device)
   
    lr = args.lr
    optimizer = optim.AdamW(model.parameters(), lr, weight_decay=0.1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    best_dev_acc = 1

    for epoch in range(args.epochs):

        model.train()
        train_loss = 0
        num_batches = 0

        for batch in train_dataloader:

            features, label = batch
            features = features.type(torch.float)
            features = features.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            # outputs = torch.argmax(outputs, dim=1)
            # outputs = outputs.unsqueeze(1)
            # print(outputs)
            # print(label)

            # loss_fn = nn.CrossEntropyLoss()
            loss_fn = nn.MultiLabelSoftMarginLoss()
            loss = loss_fn(outputs, label)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        scheduler.step()

        train_acc = model_eval(train_dataloader, model, device)
        dev_acc = model_eval(val_dataloader, model, device)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, args.filepath)

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")


def test(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

    saved = torch.load(args.filepath)
    input_size = 4
    output_size = 3
    model = MLP(input_size, output_size)
    model.load_state_dict(saved['model'])
    model = model.to(device)
    print(f"load model from {args.filepath}")

    test_path = 'Data/test.csv'

    test_dataset = FinCrimesDataset(test_path)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

    total_samples = 0
    correct_predictions = 0

    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            features, label = batch
            features = features.to(device)
            label = label.to(device)

            outputs = model(features)
            _, predicted_labels = torch.max(outputs, dim=1)


            total_samples += label.size(0)
            correct_predictions += (predicted_labels == label).sum().item()

    accuracy = correct_predictions / total_samples

    print(f"Test loss: {accuracy:.3f}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--use_gpu", action='store_true', default=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
  
    args.filepath = f'{args.epochs}-{args.lr}.pt'

 
    config = SimpleNamespace(
        filepath=args.filepath,
        lr=args.lr,
        use_gpu=args.use_gpu,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    print('Training on dataset...')
    train(config)

    print('Testing on dataset...')
    test(config)

# nohup python -u train.py > train.out 2>&1 &
