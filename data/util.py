import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import random_split, DataLoader
import json
import torch
import selfies as sf


class ExperimentalDataset(Dataset):
    def __init__(self, path='./data/data_improved_final.tsv'):
        super().__init__()

        # Load Dataset
        self.path = path
        self.data = []
        with open(self.path, 'r') as f:
            lines = f.readlines()[1:]
            data_original= [(i.split("\t")[2],i.split("\t")[5:-1]) for i in lines]
        data_parsed = []
        for selfies, ir in data_original:
            ir = [float(i) for i in ir]
            data_parsed.append((selfies, ir.reverse()))
        for selfies, ir in data_parsed:
            to_np = np.zeros((81, 2))
            j = 0
            for i in range(0, len(ir)//2):
                to_np[i][0] = ir[j]
                to_np[i][1] = ir[j+1]
                j+=2
            self.data.append(
                {'decoderin':"[START]"+selfies,'decoderout':selfies+"[STOP]", 'spectra':to_np}
                )
            

        # Symbol to integer label maps
        self.symbol_to_idx = json.load(open('./data/symbol_to_idx.json'))
        self.idx_to_symbol = {int(k):v for k, v in json.load(open('./data/idx_to_symbol.json')).items()}

        # IR spectra
        self.spectra = np.array([sample['spectra'] for sample in self.data])
        self.decoder_inputs = [sample['decoderin'] for sample in self.data]
        self.decoder_outputs = [sample['decoderout'] for sample in self.data]
        self.pad_to_len = 24

        #Filter out incorrect symbols 
        valid_data = []
        for sample in self.data:
            try:
                # Decode to ensure it's syntactically valid SELFIES
                _ = sf.selfies_to_encoding(sample['decoderin'], self.symbol_to_idx, self.pad_to_len, enc_type='label')
                if sf.len_selfies(sample['decoderin']) > 24:
                    continue
                valid_data.append(sample)
            except Exception as e:
                #print(f"Skipping invalid SELFIES: {sample['decoderin']} | Error: {e}")
                continue

        self.data = valid_data
        self.spectra = np.array([sample['spectra'] for sample in self.data])
        self.decoder_inputs = [sample['decoderin'] for sample in self.data]
        self.decoder_outputs = [sample['decoderout'] for sample in self.data]
        # Maximum SELFIES string length


        # Calculate min and max for normalization
        self.spectra_min = 0.0
        self.spectra_max = 4070.68

    def __len__(self):
        # Get number of samples
        return self.spectra.shape[0]

    def __getitem__(self, idx):
        # Transform spectra to pytorch float tensors

        # Normalize the spectrum
        spectrum = self.spectra[idx].copy()
        spectrum[:, 0] = (spectrum[:, 0] - self.spectra_min) / (self.spectra_max - self.spectra_min) #Normalize wavenumbers
        spectrum[:, 1] = spectrum[:, 1] / 100.0 # Normalize transmittance

        spectrum = torch.from_numpy(spectrum).float()

        # Transform decoder inputs to integer label vectors and to pytorch tensors
        decoder_input = self.decoder_inputs[idx]
        decoder_input = torch.as_tensor(sf.selfies_to_encoding(decoder_input, self.symbol_to_idx, self.pad_to_len, enc_type='label'))

        # Transform decoder outputs to integer label vectors and to pytorch tensors
        decoder_output = self.decoder_outputs[idx]
        decoder_output = torch.as_tensor(sf.selfies_to_encoding(decoder_output, self.symbol_to_idx, self.pad_to_len, enc_type='label'))

        return spectrum, decoder_input, decoder_output
    
class SpectrumDataset(Dataset):

    # PyTorch Dataset of IR Spectra and correspond selfies

    
    def __init__(self, path='./data/fixed_spectra.npz'):
        super().__init__()

        # Dataset
        self.path = path
        data = np.load(self.path)
        self.data = {k: data[k] for k in data.files}

        # Symbol to integer label maps
        with open("./data/symbol_to_idx.json") as f:
            self.symbol_to_idx = json.load(f)
        with open("./data/idx_to_symbol.json") as h:
            self.idx_to_symbol = {int(k):v for k, v in json.load(h).items()}

        # IR spectra
        self.spectra = self.data['spectra']

        # Decoder inputs (SELFIES strings [START]...)
        self.decoder_inputs = self.data['decoderin']

        # Decoder outputs (SELFIES strins ...[STOP])
        self.decoder_outputs = self.data['decoderout']

        # Maximum SELFIES string length
        self.pad_to_len = max(sf.len_selfies(s) for s in self.decoder_inputs)

        # Calculate min and max for normalization
        self.spectra_min = np.min(self.spectra)
        self.spectra_max = np.max(self.spectra)

    def __len__(self):
        # Get number of samples
        return self.spectra.shape[0]

    def __getitem__(self, idx):
        # Transform spectra to pytorch float tensors

        # Normalize the spectrum
        spectrum = self.spectra[idx].copy()
        spectrum[:, 0] = (spectrum[:, 0] - self.spectra_min) / (self.spectra_max - self.spectra_min) #Normalize wavenumbers
        spectrum[:, 1] = spectrum[:, 1] / 100.0 # Normalize transmittance

        spectrum = torch.from_numpy(spectrum).float()

        # Transform decoder inputs to integer label vectors and to pytorch tensors
        decoder_input = self.decoder_inputs[idx]
        decoder_input = torch.as_tensor(sf.selfies_to_encoding(decoder_input, self.symbol_to_idx, self.pad_to_len, enc_type='label'))

        # Transform decoder outputs to integer label vectors and to pytorch tensors
        decoder_output = self.decoder_outputs[idx]
        decoder_output = torch.as_tensor(sf.selfies_to_encoding(decoder_output, self.symbol_to_idx, self.pad_to_len, enc_type='label'))

        return spectrum, decoder_input, decoder_output
    
class SpectrumDataLoader():
    def __init__(self, dataset, batch_size=64, val_split=0.1, test_split=0.1, shuffle=False, seed=12, num_workers=0):
        torch.manual_seed(seed)                # CPU
        torch.cuda.manual_seed(seed)           # Current GPU
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed=seed
        self.num_workers = num_workers

        total_len = len(dataset)
        val_len = int(total_len * val_split)
        test_len = int(total_len*test_split)
        train_len = total_len - (val_len+test_len)
        
        generator = torch.Generator().manual_seed(self.seed)
        self.train_set, self.val_set, self.test_set = random_split(dataset, [train_len, val_len, test_len], generator=generator)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=self.shuffle, generator=torch.Generator().manual_seed(self.seed), num_workers=self.num_workers)
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=self.shuffle, generator=torch.Generator().manual_seed(self.seed), num_workers=self.num_workers)
    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=self.shuffle, generator=torch.Generator().manual_seed(self.seed), num_workers=self.num_workers)
