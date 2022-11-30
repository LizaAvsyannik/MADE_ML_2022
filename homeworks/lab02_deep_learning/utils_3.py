import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import clear_output


def check_if_roman_numeral(s):
    validRomanNumerals = ["M", "D", "C", "L", "X", "V", "I", "(", ")"]
    valid = True
    for letter in s:
        if letter not in validRomanNumerals:
            valid = False
            break
    return valid


def to_matrix(token_to_idx, sequences, max_len=None, dtype='int32', PAD='_'):    
    max_len = max_len or max(map(len, sequences))
    sequences_ix = np.zeros([len(sequences), max_len], dtype)

    for i in range(len(sequences)):
        line_ix = [token_to_idx[c] for c in sequences[i]]
        sequences_ix[i, :len(sequences[i])] = line_ix
        sequences_ix[i, len(sequences[i]):] = token_to_idx[PAD]

    return sequences_ix


def train_loop(model, loss_func, opt, n_epoch, num_tokens, train_loader, device):
    history = []
    for i in range(n_epoch):
        epoch_loss = []
        for batch_ix in train_loader:
            batch_ix = torch.tensor(batch_ix, dtype=torch.long).to(device)
            
            initial_state = model.initial_state(batch_size=batch_ix.shape[0])
            logits_seq, _ = model(batch_ix, initial_state)
            predictions_logits = logits_seq[:, :-1]
            actual_next_tokens = batch_ix[:, 1:]

            loss = loss_func(
            predictions_logits.reshape((-1, num_tokens)),
            actual_next_tokens.reshape(-1)
            )

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
