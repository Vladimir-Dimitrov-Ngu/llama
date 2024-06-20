import math
from contextlib import nullcontext
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from base_llama import LlamaConfig, LlamaPreTrainedModel
from rope import apply_rotary_emb


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Compute the root mean square normalization. Use Equation 4 under
        Section 4 of https://arxiv.org/abs/1910.07467 as a reference. Add
        the given epsilon value (self.eps) to the tensor's norm (i.e. inside
        the square root in Equation 4) before normalizing the tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.
        """
        x_squared = x.pow(2)  # Element-wise square
        
        # Calculate the mean of squared elements across dim=1
        mean = torch.mean(x_squared, dim=1, keepdim=True)  # Shape: (n, 1)
        
        # Compute denominator: sqrt(mean + eps)
        denominator = torch.sqrt(mean + self.eps)  # Shape: (n, 1)
        
        # Normalize x by the denominator
        normalized = x / denominator
        
        return normalized

    def forward(self, x):
        """
        Apply the root mean square normalizer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Attention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.n_kv_heads = (
            config.n_heads if config.n_kv_heads is None else config.n_kv_heads
        )
        assert config.n_heads % self.n_kv_heads == 0
        model_parallel_size = 1
        self.n_local_heads = config.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.max_seq_len = config.max_seq_len

        # linear layers for computing query, key, value, and output tensors
        self.compute_query = nn.Linear(
            config.dim, config.n_heads * self.head_dim, bias=False
        )
        self.compute_key = nn.Linear(
            config.dim, self.n_kv_heads * self.head_dim, bias=False
        )
        self.compute_value = nn.Linear(
            config.dim, self.n_kv_heads * self.head_dim, bias=False
        )
        self.compute_output = nn.Linear(
            config.n_heads * self.head_dim, config.dim, bias=False
        )

        # dropout layers for attention and residual connection
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

    def compute_query_key_value_scores(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """
        Jointly compute Scaled Dot Product Attention (see Section 3.2.1 in
        https://arxiv.org/abs/1706.03762 for details). Or see torch.nn.functional.scaled_dot_product_attention.
        An optimal implemention will jointly computing attention for multiple
        heads (n_local_heads of them) at once using matrix/tensor operations.

        Args:
            query (torch.Tensor): Query tensor with shape (bs, n_local_heads, seqlen, head_dim).
            key (torch.Tensor): Key tensor with shape (bs, n_local_kv_heads, seqlen, head_dim).
            value (torch.Tensor): Value tensor with shape (bs, n_local_kv_heads, seqlen, head_dim).

        Returns:
            torch.Tensor: Output tensor after applying Scaled Dot Product Attention. softmax(QK^T)V, shape (bs, n_local_heads, seqlen, head_dim).
        """
        # Compute the scaled dot product attention scores
        # (Q @ K.T) / sqrt(d_k)
        scores = torch.einsum("bnqd,bnkd->bnqk", query, key) / (self.head_dim**0.5)

        # Apply the softmax function to the attention scores along the last dimension to normalize the scores so they sum to one
        attention_probs = F.softmax(scores, dim=-1)

        # Apply dropout to the attention probabilities
        if self.dropout > 0.0:
            attention_probs = self.attn_dropout(attention_probs)

        # Get the weighted sum of the values based on their attention probabilities: attention_probs @ V
        output = torch.einsum("bnqk,bnkd->bnqd", attention_probs, value)

        # Apply residual dropout to the output tensor
        if self.dropout > 0.0:
            output = self.resid_dropout(output)

        # Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
        return output  # (bs, n_local_heads, seqlen, head_dim)

    def forward(self, x: torch.Tensor):
        """
        Llama2 uses Grouped-Query Attention. The details of GQA are actually
        not critical to solving this assignment; you are simply asked to
        compute Scaled Dot Product Attention (see above for details). GQA is
        a memory optimization to compute multi-head attention efficiently.
        See
        Section 2.2 in https://arxiv.org/abs/2305.13245 or
        https://ai.plainenglish.io/understanding-llama2-kv-cache-grouped-query-attention-rotary-embedding-and-more-c17e5f49a6d7
        for details.
        """
        batch_size, seqlen, _ = x.shape

        query = self.compute_query(x)
        key = self.compute_key(x)
        value = self.compute_value(x)
        query = query.view(batch_size, seqlen, self.n_local_heads, self.head_dim)
        key = key.view(batch_size, seqlen, self.n_local_kv_heads, self.head_dim)
        value = value.view(batch_size, seqlen, self.n_local_kv_heads, self.head_dim)

        # RoPE relative positional embeddings
        query, key = apply_rotary_emb(query, key, self.head_dim, self.max_seq_len)

        # Grouped multiquery attention: expand out keys and values.
        # Convert both to:
        # (bs, seqlen, n_local_heads, head_dim)
        key = torch.repeat_interleave(key, dim=2, repeats=self.n_rep)
        value = torch.repeat_interleave(value, dim=2, repeats=self.n_rep)

        # make heads into a batch dimension
        query = query.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        output = self.compute_query_key_value_scores(query, key, value)

        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(batch_size, seqlen, -1)

        # final projection into the residual stream
        output = self.resid_dropout(self.compute_output(output))
        return output


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def SwiGLU(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the SwiGLU activation function (see Section 2 in
        https://arxiv.org/abs/2204.02311 for details).

        """
        return F.silu(self.w1(x)) * self.w3(x)

    def forward(self, x):
        return self.dropout(self.w2(self.SwiGLU(x)))


class LlamaLayer(nn.Module):
    def __init__(self, layer_id: int, config: LlamaConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        self.attention = Attention(config)
        self.feed_forward = FeedForward(
            dim=config.dim,
            hidden_dim=config.hidden_dim,
            multiple_of=config.multiple_of,
            dropout=config.dropout,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(config.dim, eps=config.layer_norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.layer_norm_eps)

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the basic transformer building block. This is a
        modernized version of the block shown on the left of Figure 1 on
        https://arxiv.org/pdf/1706.03762.pdf.

        The transformer block should consist of:
        1) layer normalization of the input (via Root Mean Square layer normalization)
        2) self-attention on the layer-normalized input
        3) a residual connection (i.e., add the input to the output of the self-attention)
        3) layer normalization on the output of the self-attention
        4) a feed-forward network on the layer-normalized output of the self-attention
        5) add a residual connection from the unnormalized self-attention output to the output of the feed-forward network

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # layer normalization of the input
        x_norm = self.attention_norm(x)
        # self-attention on the layer-normalized input
        attn_output = self.attention(x_norm)
        # residual connection
        x = x + attn_output

        # layer normalization on the output of the self-attention
        x_norm = self.ffn_norm(x)
        # feed-forward network on the layer-normalized output of the self-attention
        ffn_output = self.feed_forward(x_norm)
        # add a residual connection from the unnormalized self-attention output to the output of the feed-forward network
        x = x + ffn_output
        return x


class Llama(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        """Initialize the Llama model using the given configuration.

        Args:
            config (LlamaConfig): Configuration class for Llama model.

        Attributes:
            params (LlamaConfig): Stores the configuration passed to the model.
            vocab_size (int): Vocabulary size.
            n_layers (int): Number of transformer layers.

            tok_embeddings (nn.Embedding): Embedding layer for token inputs.
            dropout (nn.Dropout): Dropout layer to prevent overfitting.
            layers (nn.ModuleList): List of transformer layers (LlamaLayer).
            norm (RMSNorm): Normalization layer applied at the end of the model.
            output (nn.Linear): Linear layer that maps the final output vectors to a vocabulary-sized space.
        """
        super().__init__(config)
        self.params = config
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layers):
            self.layers.append(LlamaLayer(layer_id, config))
        self.norm = RMSNorm(config.dim, eps=config.layer_norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # share the unembedding parameters with the embedding parameters
        # which can lead to more efficient learning and reduced model size
        # https://paperswithcode.com/method/weight-tying
        self.tok_embeddings.weight = self.output.weight

        # some useful precompute for the RoPE relative positional embeddings

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("w3.weight") or pn.endswith("compute_output.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Defines the forward pass of the Llama model.

        Args:
            tokens (torch.Tensor): Input token indices.
            targets (Optional[torch.Tensor]): Target token indices for training.

        Returns:
            Tuple containing:
            - logits: Logits over the vocabulary.
            - h: Hidden states of the last layer.
        """
        _batch_size, _ = tokens.shape  # batch size, sequence length

        # Embed tokens and apply dropout.
        h = self.dropout(self.tok_embeddings(tokens))

        # Pass through each transformer layer.
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.output(h)
        else:
            logits = self.output(
                h[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim

        return logits, h

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        We perform this generation using basic temperature sampling. Note that we are not using
        nucleus sampling (i.e. limiting ourselves to sampling from the top-k most probable tokens
        at each timestep), though this is often used in conjunction with temperature sampling,
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        Also note this is a super inefficient version of sampling with no key/value cache.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx
                if idx.size(1) <= self.params.max_seq_len
                else idx[:, -self.params.max_seq_len :]
            )
            # forward the model to get the logits for the last index in the sequence
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # get the logits at the final time step

            # Generate the next token index using temperature sampling
            # If temperature is close to 0, select the single most likely index
            if abs(temperature) < 1e-6:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)

            # else, sample from the logits to introduce randomness for the next token
            else:
                # scale (divide) the logits/probabilities by the given temperature
                idx_next = logits / temperature

                # normalize the scaled logits with a softmax to obtain probabilities
                idx_next = F.softmax(idx_next, dim=-1)

                # sample from the scaled probability distribution
                idx_next = torch.multinomial(idx_next, num_samples=1)

            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


def load_pretrained(checkpoint: str) -> Llama:
    """Load a pre-trained Llama model from a checkpoint file and return the model."""

    # Determine the device to use for the model
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Decide mixed precision based on device availability
    dtype = (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float32"  # 'float32' or 'bfloat16' or 'float16'
    )

    # Adjust settings for CUDA to allow TensorFlow32 precision if available
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True  # Allow TensorFlow32 on matmul
        torch.backends.cudnn.allow_tf32 = True  # Allow TensorFlow32 on cuDNN

    # Setup automatic mixed precision context if CUDA is available
    device_type = device if device in ["cuda"] else "cpu"
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]

    ctx = (
        nullcontext()
        if device_type != "cuda"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )

    # Load the model checkpoint to the selected device
    # Apply automatic mixed precision context if relevant
    with ctx:
        # init from a model saved in a specific directory
        checkpoint_dict = torch.load(checkpoint, map_location=device)
        config = LlamaConfig(**checkpoint_dict["model_args"])
        model = Llama(config)
        state_dict = checkpoint_dict["model"]

        # If there are prefixed keys in the state dict (often from distributed training), adjust them
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

        model.load_state_dict(state_dict, strict=False)
        return model