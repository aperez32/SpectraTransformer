
from transformer import SpectraTransformer
from data.util import ExperimentalDataset, SpectrumDataset, SpectrumDataLoader
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
import os
import time
from torch.amp import autocast, GradScaler

#Deterministic
seed = 4892312
torch.manual_seed(seed)
torch.cuda.manual_seed(seed) 

def main():

    ####### Dataset processing params #####
    dataset=SpectrumDataset() #Choose Dataset wrapper
    batch_size = 1024
    num_workers = 4


    ####### Model Params ########
    max_spectra_len = 81
    vocab_size = len(dataset.idx_to_symbol)
    max_len = dataset.pad_to_len
    d_model = 320
    num_layers = 6 
    num_heads = 5 
    drop_out= 0.2    
    pad_idx = dataset.symbol_to_idx['[nop]']  # i.e., pad_idx = 31

    ####### Training Params #####
    save_every = 5
    num_epochs = 300
    val_split= 0.15
    test_split = 0.15
    label_smoothing=0.1
    learning_rate = 1e-4

    save_path = "./checkpoints/"
    os.makedirs(save_path, exist_ok=True)


    loader = SpectrumDataLoader(dataset=dataset, 
                                batch_size=batch_size, 
                                val_split=val_split, 
                                test_split=test_split,
                                seed=seed, 
                                num_workers=num_workers)
    train_loader = loader.train_dataloader()
    val_loader = loader.val_dataloader()
    print("Dataset Loaded")


    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpectraTransformer(
        max_spectra_len=max_spectra_len, 
        vocab_size=vocab_size, 
        max_len=max_len,
        d_model=d_model, 
        num_layers=num_layers, 
        num_heads=num_heads, 
        dropout=drop_out,
        pad_idx=pad_idx
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=label_smoothing)
    scaler = GradScaler()

    print("Model Loaded. Starting Training.")

    with open('loss.txt', 'w') as f:
        pass

    for epoch in range(1, num_epochs + 1):
        start = time.time()
        model.train()
        total_loss = 0
        for spectra, decoder_input, targets in train_loader:
            spectra = spectra.to(device)
            decoder_input = decoder_input.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            with autocast(device_type='cuda'):
                logits = model(spectra, decoder_input)
                loss = loss_fn(
                    logits.view(-1, logits.size(-1)),  # (B*T, V)
                    targets.view(-1)                   # (B*T,)
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        avg_loss = total_loss/len(train_loader)
        print(f"Epoch {epoch} Loss: {avg_loss:.4f} Time: {time.time() - start:.2f}s")
        with open("loss.txt", 'a') as f:
            f.writelines(f'{avg_loss:.4f}\n')
        # validate and save checkpoint every {save_every} epochs
        if epoch % save_every == 0 or epoch == num_epochs:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for spectra, decoder_input, targets in val_loader:
                    spectra = spectra.to(device)
                    decoder_input = decoder_input.to(device)
                    targets = targets.to(device)

                    output = model(spectra, decoder_input)
                    output = output.permute(0, 2, 1)
                    loss = loss_fn(output, targets)
                    val_loss += loss.item()    
                val_loss /= len(val_loader)
                ckpt = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }
                torch.save(ckpt, os.path.join(save_path, f'epoch{epoch}.pt'))
                print(f"Checkpoint saved at epoch {epoch} Val Loss: {val_loss:.4f}")


if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()