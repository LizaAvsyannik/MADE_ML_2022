import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import clear_output
from sklearn.metrics import accuracy_score
from tqdm import tqdm


def train_loop(model, loss_func, opt, n_epoch, train_loader, device, val_loader=None):
    history = []
    val_accuracy = []
    for i in tqdm(range(n_epoch)):
        epoch_loss = []
        for x_batch, y_batch in train_loader:
            output = model(x_batch.to(device))
            
            loss = loss_func(output, y_batch.to(device))

            loss.backward()
            opt.step()
            opt.zero_grad()
            
            epoch_loss.append(loss.item())

        history.append(sum(epoch_loss) / len(epoch_loss))

        if val_loader:
            epoch_accuracy = []
            model.eval()
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    output = model(torch.Tensor(x_val).to(device))
                    predictions = output.argmax(-1).detach().cpu().numpy()
                    epoch_accuracy.append(accuracy_score(predictions, y_val))
            val_accuracy.append(sum(epoch_accuracy) / len(epoch_accuracy))

        if (i + 1) % (n_epoch / 10) == 0:
            clear_output(True)
            plt.plot(history, label='Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()
            
            if val_loader:
                plt.plot(val_accuracy, label='Validation accuracy')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.show()
        
    print(f'Final Loss: {history[-1]}')


from PIL import Image

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        x = Image.open(image_path)
        if self.transform is not None:
            x = self.transform(x)
        return x
    
    def __len__(self):
        return len(self.image_paths)
