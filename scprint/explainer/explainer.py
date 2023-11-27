from .logger import *
from .visualization import *


def get_attention(layer_num=11, batch_size=8):

    torch.cuda.empty_cache()
    dict_sum_condition = {}
    model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        M = all_gene_ids.size(1)
        N = all_gene_ids.size(0)
        device = next(model.parameters()).device
        for i in tqdm(range(0, N, batch_size)):
            batch_size = all_gene_ids[i : i + batch_size].size(0)
            outputs = np.zeros((batch_size, M, M), dtype=np.float32)
            # Replicate the operations in model forward pass
            src_embs = model.encoder(
                torch.tensor(all_gene_ids[i : i + batch_size], dtype=torch.long).to(device)
            )
            val_embs = model.value_encoder(
                torch.tensor(all_values[i : i + batch_size], dtype=torch.float).to(device)
            )
            total_embs = src_embs + val_embs
            # total_embs = model.layer(total_embs.permute(0, 2, 1)).permute(0, 2, 1)
            # Send total_embs to attention layers for attention operations
            # Retrieve the output from second to last layer
            for layer in model.transformer_encoder.layers[:layer_num]:
                total_embs = layer(
                    total_embs,
                    src_key_padding_mask=src_key_padding_mask[i : i + batch_size].to(
                        device
                    ),
                )
            # Send total_embs to the last layer in flash-attn
            # https://github.com/HazyResearch/flash-attention/blob/1b18f1b7a133c20904c096b8b222a0916e1b3d37/flash_attn/flash_attention.py#L90
            qkv = model.transformer_encoder.layers[layer_num].self_attn.Wqkv(
                total_embs
            )
            # Retrieve q, k, and v from flast-attn wrapper
            qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, h=8)
            q = qkv[:, :, 0, :, :]
            k = qkv[:, :, 1, :, :]
            v = qkv[:, :, 2, :, :]
            # https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a
            # q = [batch, gene, n_heads, n_hid]
            # k = [batch, gene, n_heads, n_hid]
            # attn_scores = [batch, n_heads, gene, gene]
            attn_scores = q.permute(0, 2, 1, 3) @ k.permute(0, 2, 3, 1)
            # apply softmax to get attention weights
            attn_scores = softmax(attn_scores, dim=-1)
            if i == 0:
                sm_attn_scores = attn_scores.sum(0).detach().cpu().numpy()
            else:
                # take the sum
                sm_attn_scores += attn_scores.sum(0).detach().cpu().numpy()
    gene_vocab_idx = all_gene_ids[0].clone().detach().cpu().numpy()
    return sm_attn_scores
    # return [pd.DataFrame(data=sm_attn_scores[i], columns=vocab.lookup_tokens(gene_vocab_idx), index=vocab.lookup_tokens(gene_vocab_idx)) for i in range(0,8)]