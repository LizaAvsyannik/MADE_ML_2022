import matplotlib.pyplot as plt
import modules
import numpy as np
import torch
from tqdm import tqdm


def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y.astype(int)] = 1.
    return y_one_hot


def get_batches(dataset, batch_size):
    X, Y = dataset
    n_samples = X.shape[0]
        
    # Shuffle at the start of epoch
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        
        batch_idx = indices[start:end]
    
        yield X[batch_idx], Y[batch_idx]


def build_model(activation_function, n_in, n_out, with_norm=False):
    net = modules.Sequential()
    net.add(modules.Flatten())
    net.add(modules.Linear(n_in, 128))
    net.add(activation_function)
    if with_norm:
        net.add(modules.BatchNormalization(alpha=0.1))
        net.add(modules.ChannelwiseScaling(128))
    net.add(modules.Linear(128, n_out))
    net.add(modules.LogSoftMax())
    return net


def train_loop(model, X, y, batch_size, n_epoch, criterion, optimizer, optimizer_config, X_val, y_val):
    train_loss_history = []
    val_loss_history = []
    optimizer_state = {}

    for _ in tqdm(range(n_epoch)):
        model.train()
        for x_batch, y_batch in get_batches((X, y), batch_size):
            
            model.zeroGradParameters()
            
            # Forward
            predictions = model.forward(x_batch)
            loss = criterion.forward(predictions, y_batch)

            # Backward
            dp = criterion.backward(predictions, y_batch)
            model.backward(x_batch, dp)
            
            # Update weights
            optimizer(model.getParameters(), 
                        model.getGradParameters(), 
                        optimizer_config,
                        optimizer_state)      
            
            train_loss_history.append(loss)
        
        model.evaluate()
        predictions = model.forward(X_val)
        val_loss = criterion.forward(predictions, y_val)
        val_loss_history.append(val_loss)
    
    return train_loss_history, val_loss_history


def plot_losses(train_losses, val_losses, labels):
    for loss, label in zip(train_losses, labels):
        plt.plot(loss, label=label)
    plt.title('Train Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.show()

    for loss, label in zip(train_losses, labels):
        print(f'Train Loss {label}: {loss[-1]}')

    for loss, label in zip(val_losses, labels):
        plt.plot(loss, label=label)
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.show()

    for loss, label in zip(val_losses, labels):
        print(f'Validation Loss {label}: {loss[-1]}')


def train_loop_torch(model, X, y, batch_size, n_epoch, criterion, optimizer, scheduler, X_val, y_val, device):
    train_loss_history = []
    val_loss_history = []

    for _ in tqdm(range(n_epoch)):
        for x_batch, y_batch in get_batches((X, y), batch_size):
            model.train()
            # Forward
            predictions = model.forward(torch.FloatTensor(x_batch).unsqueeze(1).to(device))
            loss = criterion(predictions, torch.LongTensor(y_batch).to(device))

            # Backward
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()    
            
            train_loss_history.append(loss.item())
        
            model.eval()
            with torch.no_grad():
                predictions = model.forward(torch.FloatTensor(X_val).unsqueeze(1).to(device))
                val_loss = criterion(predictions, torch.LongTensor(y_val).to(device))
                val_loss_history.append(val_loss.item())
        scheduler.step()

    return train_loss_history, val_loss_history
