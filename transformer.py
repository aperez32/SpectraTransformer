import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectraTransformer(nn.Module):
    def __init__(self, max_spectra_len, vocab_size, max_len, d_model=256, num_layers=3, num_heads=4, dropout=0.1, pad_idx=0):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.pad_idx = pad_idx
        # Cache causal masking
        self.register_buffer("cached_tgt_mask", self._generate_causal_mask(max_len))
        
        # Projection matrix encodes wn,tr pairs to vectors of dimension d_model
        self.encoder_input_proj = nn.Linear(2 , d_model)
        self.encoder_pos_embedding = nn.Embedding(max_spectra_len, d_model) 

        # Decoder token embedding + positional embedding
        self.decoder_token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.decoder_pos_embedding = nn.Embedding(max_len, d_model)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )

        # Output projection
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, spectra, decoder_input):
       
        #Encoder
        #Project spectra into model space and mask padding
        B, L, _ = spectra.shape
        src = self.encoder_input_proj(spectra)    #(B, L (81), 2) -> (B, L, d_model)
        src_key_padding_mask = self.create_src_key_padding_mask(spectra)  # shape (B, L)
        # positional encoding of peaks -> temporary comment
        pos_ids = torch.arange(L, device=src.device).unsqueeze(0).expand(B, L)
        pos_emb = self.encoder_pos_embedding(pos_ids)  
        src = src + pos_emb


        #Decoder
        #Batch size, tensor dim
        B, T = decoder_input.size()
        # Embed decoder input and add positions
        tok_emb = self.decoder_token_embedding(decoder_input)                # converts token tensors to (B, T, d_model)
        pos_ids = torch.arange(T, device=decoder_input.device).unsqueeze(0).expand(B, T)
        pos_emb = self.decoder_pos_embedding(pos_ids)                        # positional embedding (B, T, d_model)
        tgt = tok_emb + pos_emb                                      # joins both, encoding tokens and positions (B, T, d_model)
        # Mask padding tokens
        tgt_key_padding_mask = (decoder_input == self.pad_idx).float()  
        # Causal mask so decoder can't peek ahead
        tgt_mask = self.cached_tgt_mask[:T, :T].to(tgt.device, dtype=tgt.dtype)
        

        out = self.transformer(src=src, 
                               tgt=tgt, 
                               src_key_padding_mask=src_key_padding_mask, 
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               tgt_mask=tgt_mask)

        return self.output_layer(out)                                # (B, T, vocab_size)

    def _generate_causal_mask(self, size):
        return torch.triu(torch.full((size, size), float('-inf')), diagonal=1)
    def create_src_key_padding_mask(self, spectra):
        # spectra shape: (B, L, 2)
        # Padding positions where both channels are zero
        return (spectra.abs().sum(dim=-1) == 0)  

