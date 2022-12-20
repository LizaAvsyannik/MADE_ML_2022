import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output


def train_loop(model, loss_func, opt, n_epoch, train_loader, device):
    history = []
    for i in range(n_epoch):
        epoch_loss = []
        for x_batch, y_batch in train_loader:
            output = model(x_batch.to(device))
            
            loss = loss_func(output, y_batch.to(device))

            loss.backward()
            opt.step()
            opt.zero_grad()
            
            epoch_loss.append(loss.item())

        history.append(sum(epoch_loss) / len(epoch_loss))
        if (i + 1) % 100 == 0:
            clear_output(True)
            plt.plot(history, label='Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

    assert np.mean(history[:10]) > np.mean(history[-10:]), "RNN didn't converge."
    print(f'Final Loss: {history[-1]}')
