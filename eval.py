
from data.util import SpectrumDataLoader, SpectrumDataset, ExperimentalDataset
from transformer import SpectraTransformer
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import selfies as sf
import matplotlib.pyplot as plt

seed = 4892312

dataset=SpectrumDataset() #Choose Dataset wrapper

batch_size = 1024+512

####### Model params ######## Ensure these agree with main.py
max_spectra_len = 81
vocab_size = len(dataset.idx_to_symbol)
max_len = dataset.pad_to_len
d_model = 320
num_layers = 6 
num_heads = 5 
drop_out=0.2
pad_idx = dataset.symbol_to_idx['[nop]']  
save_every = 5

####### Eval params #########
val_split = .15
test_split= .15
print_every = 100000000000



loader = SpectrumDataLoader(dataset=dataset, 
                            batch_size=batch_size, 
                            val_split=val_split,
                            test_split=test_split,
                            seed = 4892312)
test_loader = loader.test_dataloader()
print("Dataset Loaded")

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SpectraTransformer(
    max_spectra_len=max_spectra_len, 
    vocab_size=vocab_size, 
    max_len=max_len,
    d_model=d_model, 
    num_layers=num_layers, 
    num_heads=num_heads, 
    pad_idx=pad_idx
).to(device)

directory = "./checkpoints"
num_checkpoints = len(os.listdir(directory))

train_epochs = list(range(1,num_checkpoints*save_every+1))
train_loss = []
with open("loss.txt") as f:
    for line in f.readlines():
        train_loss.append(float(line.strip()))
val_epochs = list(range(save_every, num_checkpoints*save_every+1, save_every))
val_losses = []
accuracies = []

for filename in os.listdir(directory):
    checkpoint_path = os.path.join(directory, filename)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"Val Loss: {checkpoint['val_loss']}")

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    #print("Model Loaded")
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # === Evaluation loop ===
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    global_index = 0
    num_correct = 0

    with torch.no_grad():
        for spectra, decoder_input, targets in test_loader:

            spectra = spectra.to(device)
            decoder_input = decoder_input.to(device)
            targets = targets.to(device)

            output = model(spectra, decoder_input)  # (B, T, vocab_size)
            output = output.permute(0, 2, 1)        # (B, vocab_size, T)

            loss = criterion(output, targets)
            total_loss += loss.item()

            # Accuracy
            pred = output.argmax(dim=1)             # (B, T)
            mask = (targets != pad_idx)
            total_correct += ((pred == targets) & mask).sum().item()
            total_tokens += mask.sum().item()
            
            
            B = spectra.size(0)
            for i in range(B):
                pred_seq = pred[i].tolist()
                targ_seq = targets[i].tolist()
                pred_selfies = sf.encoding_to_selfies(pred_seq, vocab_itos=dataset.idx_to_symbol, enc_type='label').split('[nop]')[0]
                targ_selfies = sf.encoding_to_selfies(targ_seq, vocab_itos=dataset.idx_to_symbol, enc_type='label').split('[nop]')[0]
                if pred_selfies == targ_selfies and pred_selfies is not None and targ_selfies is not None:
                    num_correct += 1
                if global_index % print_every == 0:
                    print(f"Prediction SELFIES: {pred_selfies}")
                    print(f"Target     SELFIES: {targ_selfies}\n")
                    pass
                
                global_index += 1

    avg_loss = total_loss / len(test_loader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    
    val_losses.append(avg_loss)
    accuracies.append( num_correct/(test_split*len(dataset)) )
    #print(f"Evaluation complete.")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Token Accuracy: {accuracy * 100:.2f}%")
    print(f'String Accuracy: {100*num_correct/(len(dataset)*test_split)}')


#normalize val loss
max_val_loss = max(val_losses)
val_losses = [val / max_val_loss for val in val_losses]


print(val_losses)
plt.figure(figsize=(10, 5))
plt.plot(train_epochs, train_loss, color='green',label="Training Loss")
plt.plot(val_epochs, val_losses,color='maroon', label="Validation Loss", marker='o')
plt.plot(val_epochs, accuracies, color= 'blue', label="Accuracy", marker='x')
plt.xlabel("Epoch")
plt.ylabel("Metric Value")
plt.title("Validation Metrics Over Epochs")
plt.legend()
plt.grid(True)
plt.show()
