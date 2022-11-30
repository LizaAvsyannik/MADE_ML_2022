import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm


def train_loop(model, loss_func, opt, n_epoch, train_loader, val_loader, device):
    train_loss_history = []
    train_accuracy_history = []
    val_loss_history = []
    val_accuracy_history = []

    for _ in tqdm(range(n_epoch)):
        epoch_train_loss = []
        epoch_train_accuracy = []
        epoch_val_loss = []
        epoch_val_accuracy = []
        model.train()
        for x_batch, y_batch in train_loader:
            predictions = model.forward(x_batch.to(device))
            loss = loss_func(predictions, y_batch.to(device))

            loss.backward()
                
            opt.step()
            opt.zero_grad()    
            
            epoch_train_loss.append(loss.item())
            predictions = predictions.argmax(dim=-1).cpu().numpy()
            epoch_train_accuracy.append(accuracy_score(predictions, y_batch.numpy()))

        train_loss_history.append(sum(epoch_train_loss) / len(epoch_train_loss))
        train_accuracy_history.append(sum(epoch_train_accuracy) / len(epoch_train_accuracy))

        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                predictions = model.forward(x_batch.to(device))
                loss = loss_func(predictions, y_batch.to(device))
                epoch_val_loss.append(loss.item())
                predictions = predictions.argmax(dim=-1).cpu().numpy()
                epoch_val_accuracy.append(accuracy_score(predictions, y_batch.numpy()))

        val_loss_history.append(sum(epoch_val_loss) / len(epoch_val_loss))
        val_accuracy_history.append(sum(epoch_val_accuracy) / len(epoch_val_accuracy))
    
    return train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history


def plot_results(train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history, title=''):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title)
    ax[0].plot(np.array(train_loss_history), label='Train Loss')
    ax[0].plot(np.array(val_loss_history), label='Validation Loss')
    ax[0].set_title('Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].set_yscale('log')
    ax[0].legend()

    ax[1].plot(np.array(train_accuracy_history), label='Train Accuracy')
    ax[1].plot(np.array(val_accuracy_history), label='Validation Accuracy')
    ax[1].set_title('Accuracy')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_ylim(0.8, 1.0)
    plt.show()


def calculate_test_accuracy(model, test_loader, device):
    model.eval()
    accuracy_scores = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            probas = model.forward(x_batch.to(device))
            predictions = probas.argmax(dim=-1).cpu().numpy()
            accuracy_scores.append(accuracy_score(predictions, y_batch.numpy()))

    print(f'Accuracy : {sum(accuracy_scores) / len(accuracy_scores)}')
