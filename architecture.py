import torch
import torch.nn as nn
import torch. nn.functional as F
from dataclasses import dataclass
from typing import Optional
import math

@dataclass
class Modelargs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1 # This will be set later in the build method
    multiple_of: int =256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    #Arguments needed for the KV cache
    max_batch_size : int =32
    max_seq_len:int = 2048

    device:str = None

class Transformer(nn.Module):
    def __init__(self, args: Modelargs):
        super().__init__()

        assert args.vocab_size != 1, "vocab must have been set by now by the build stage"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size , args.dim)

        self.layers = nn.ModuleList()
        for layers_id in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias = False)

        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads , self.args.max_seq_len *2, device = self.args.device)

    
    def forward(self, tokens: torch.Tensor, start_pos : int):
        #(B, Seq_Len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1 , "only one token can be processed here"

        #(B, Seq_Len) --> (B,Seq_Len, dim)
        h = self.tok_embeddings(tokens)
        
        # Retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos:start_pos+seq_len]

        #consecutively apply all the encoder layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()
        return output
