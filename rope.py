from typing import Tuple

import torch


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Helper function to reshape frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)


def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query and key tensors.
    The rotation to each token embedding is a function of that token's position in the sequence, head_dim, and theta.
    The input tensors are reshaped as complex numbers to simplify your implementation.

    Please refer to slide 22 in https://phontron.com/class/anlp2024/assets/slides/anlp-05-transformers.pdf
    and Section 3 in https://arxiv.org/abs/2104.09864.

    Args:
        query (torch.Tensor):
            Query tensor to apply rotary embeddings.
            Shape: (batch_size, seqlen, n_local_heads, self.head_dim)
        key (torch.Tensor):
            Key tensor to apply rotary embeddings.
            Shape: (batch_size, seqlen, n_local_kv_heads, self.head_dim)
        head_dim (int): Dimension of each attention head.
        max_seq_len (int): Maximum sequence length supported by model.
        theta (float):
            Base of the exponentiation, which controls the rate of decay for higher frequency components. Defaults to 10000.0.
    Returns:
        query_out, key_out: Tuple[torch.Tensor, torch.Tensor]
            Tuple of modified query tensor and key tensor with rotary embeddings.
    """

    _, seqlen, _, _ = query.shape  # get d
    device = query.device

    # reshape xq and xk to match the complex representation
    # separate each query/key vector into its odd and even indices (assuming *one-indexing*).
    # query_real contains q_1, q_3, q_5, ... and query_imag contains q_2, q_4, q_6, ...
    query_real, query_imag = (
        query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
    )
    key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)

    # Compute the angles using the position IDs and a frequency tensor
    position_ids = torch.arange(seqlen, device=device)
    indices = torch.arange(head_dim // 2, device=device)  # get d/2
    angles = position_ids[:, None] / (theta ** (2 * indices / head_dim))

    # Compute the trigonometric values
    cos_vals = torch.cos(angles).to(device)  # (seqlen, head_dim//2)
    sin_vals = torch.sin(angles).to(device)

    # Reshape for broadcasting during element-wise operations, to match the shape of the query and key tensors, (1, seqlen, 1, head_dim//2)
    cos_vals = cos_vals.unsqueeze(0).unsqueeze(2)
    sin_vals = sin_vals.unsqueeze(0).unsqueeze(2)

    # Apply the rotary embeddings using the provided formula
    query_rot_real = query_real * cos_vals - query_imag * sin_vals
    query_rot_imag = query_real * sin_vals + query_imag * cos_vals
    key_rot_real = key_real * cos_vals - key_imag * sin_vals
    key_rot_imag = key_real * sin_vals + key_imag * cos_vals

    # Combine the real and imaginary parts along the last dimension
    # Reshape back into a single tensor
    query_out = torch.stack((query_rot_real, query_rot_imag), dim=-1).reshape(
        query.shape
    )
    key_out = torch.stack((key_rot_real, key_rot_imag), dim=-1).reshape(key.shape)

    # Return the rotary position embeddings for the query and key tensors
    return query_out, key_out