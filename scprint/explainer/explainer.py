from .logger import *
from .visualization import *
import torch
import numpy as np
from tqdm import tqdm
from einops import rearrange
import numpy as np
import seaborn as sns
import itertools
import networkx as nx
import scanpy as sc
import anndata as ad
import pandas as pd

### @title Explainable Attention ####
class BaseExplainer:
    def __init__(self, model,) -> None:
        self.model = model
        
        
    def get_attention(self, input_ids, input_tokens, output_attentions=True, save_path=""):
        ### run the model with the data
        if len(save_path)>0:
        # save as pkl
        with open(save_path+"_attn_scores.pkl", "wb") as f:
            pickle.dump(attn_scores, f)
        ## TODO: show highly attended genes:
        
        return attn_scores

    def viz_attention(, joint_attention=True, layer=-1, loc=(0,10), key_genes=[], gene_names=[], random=False):
        attention = self.get_attentions()
        if len(loc) >0:
            todisp = [torch.FloatTensor([attention[:,loc[0]:loc[1],loc[0]:loc[1]]])]
            names = gene_names[loc[0]:loc[1]]
        elif len(key_genes) >0:
            
            names = key_genes
        elif random:
            random_indices = np.random.randint(low=0, high=attention.shape[1], size=80)
            todisp = [attention[:, random_indices, :][:,:,random_indices]]
            names = np.array(gene_names)[random_indices]
        else:
            raise ValueError("Please provide either a location or a list of genes to visualize")

        head_view(todisp, names)
        

    # @title Utilities
    def get_adjmat(mat, input_tokens):
        n_layers, length, _ = mat.shape
        adj_mat = np.zeros(((n_layers + 1) * length, (n_layers + 1) * length))
        labels_to_index = {}
        for k in np.arange(length):
            labels_to_index[str(k) + "_" + input_tokens[k]] = k

        for i in np.arange(1, n_layers + 1):
            for k_f in np.arange(length):
                index_from = (i) * length + k_f
                label = "L" + str(i) + "_" + str(k_f)
                labels_to_index[label] = index_from
                for k_t in np.arange(length):
                    index_to = (i - 1) * length + k_t
                    adj_mat[index_from][index_to] = mat[i - 1][k_f][k_t]

        return adj_mat, labels_to_index


    def draw_attention_graph(adjmat, labels_to_index, n_layers, length):
        A = adjmat
        G = nx.from_numpy_matrix(A, create_using=nx.DiGraph())
        for i in np.arange(A.shape[0]):
            for j in np.arange(A.shape[1]):
                nx.set_edge_attributes(G, {(i, j): A[i, j]}, "capacity")

        pos = {}
        label_pos = {}
        for i in np.arange(n_layers + 1):
            for k_f in np.arange(length):
                pos[i * length + k_f] = ((i + 0.5) * 2, length - k_f)
                label_pos[i * length + k_f] = (i * 2, length - k_f)

        index_to_labels = {}
        for key in labels_to_index:
            index_to_labels[labels_to_index[key]] = key.split("_")[-1]
            if labels_to_index[key] >= length:
                index_to_labels[labels_to_index[key]] = ""

        # plt.figure(1,figsize=(20,12))

        nx.draw_networkx_nodes(G, pos, node_color="green", node_size=50)
        nx.draw_networkx_labels(G, pos=label_pos, labels=index_to_labels, font_size=10)

        all_weights = []
        # 4 a. Iterate through the graph nodes to gather all the weights
        for node1, node2, data in G.edges(data=True):
            all_weights.append(
                data["weight"]
            )  # we'll use this when determining edge thickness

        # 4 b. Get unique weights
        unique_weights = list(set(all_weights))

        # 4 c. Plot the edges - one by one!
        for weight in unique_weights:
            # 4 d. Form a filtered list with just the weight you want to draw
            weighted_edges = [
                (node1, node2)
                for (node1, node2, edge_attr) in G.edges(data=True)
                if edge_attr["weight"] == weight
            ]
            # 4 e. I think multiplying by [num_nodes/sum(all_weights)] makes the graphs edges look cleaner

            w = weight  # (weight - min(all_weights))/(max(all_weights) - min(all_weights))
            width = w
            nx.draw_networkx_edges(
                G, pos, edgelist=weighted_edges, width=width, edge_color="darkblue"
            )

        return G


    def get_attention_graph(adjmat, labels_to_index, n_layers, length):
        A = adjmat
        G = nx.from_numpy_matrix(A, create_using=nx.DiGraph())
        for i in np.arange(A.shape[0]):
            for j in np.arange(A.shape[1]):
                nx.set_edge_attributes(G, {(i, j): A[i, j]}, "capacity")

        pos = {}
        label_pos = {}
        for i in np.arange(n_layers + 1):
            for k_f in np.arange(length):
                pos[i * length + k_f] = ((i + 0.5) * 2, length - k_f)
                label_pos[i * length + k_f] = (i * 2, length - k_f)

        index_to_labels = {}
        for key in labels_to_index:
            index_to_labels[labels_to_index[key]] = key.split("_")[-1]
            if labels_to_index[key] >= length:
                index_to_labels[labels_to_index[key]] = ""

        # plt.figure(1,figsize=(20,12))

        nx.draw_networkx_nodes(G, pos, node_color="green", node_size=50)
        nx.draw_networkx_labels(G, pos=label_pos, labels=index_to_labels, font_size=10)

        all_weights = []
        # 4 a. Iterate through the graph nodes to gather all the weights
        for node1, node2, data in G.edges(data=True):
            all_weights.append(
                data["weight"]
            )  # we'll use this when determining edge thickness

        # 4 b. Get unique weights
        unique_weights = list(set(all_weights))

        # 4 c. Plot the edges - one by one!
        for weight in unique_weights:
            # 4 d. Form a filtered list with just the weight you want to draw
            weighted_edges = [
                (node1, node2)
                for (node1, node2, edge_attr) in G.edges(data=True)
                if edge_attr["weight"] == weight
            ]
            # 4 e. I think multiplying by [num_nodes/sum(all_weights)] makes the graphs edges look cleaner

            w = weight  # (weight - min(all_weights))/(max(all_weights) - min(all_weights))
            width = w
            nx.draw_networkx_edges(
                G, pos, edgelist=weighted_edges, width=width, edge_color="darkblue"
            )

        return G


    def compute_flows(G, labels_to_index, input_nodes, length):
        number_of_nodes = len(labels_to_index)
        flow_values = np.zeros((number_of_nodes, number_of_nodes))
        for key in labels_to_index:
            if key not in input_nodes:
                current_layer = int(labels_to_index[key] / length)
                pre_layer = current_layer - 1
                u = labels_to_index[key]
                for inp_node_key in input_nodes:
                    v = labels_to_index[inp_node_key]
                    flow_value = nx.maximum_flow_value(
                        G, u, v, flow_func=nx.algorithms.flow.edmonds_karp
                    )
                    flow_values[u][pre_layer * length + v] = flow_value
                flow_values[u] /= flow_values[u].sum()

        return flow_values


    def compute_node_flow(G, labels_to_index, input_nodes, output_nodes, length):
        number_of_nodes = len(labels_to_index)
        flow_values = np.zeros((number_of_nodes, number_of_nodes))
        for key in output_nodes:
            if key not in input_nodes:
                current_layer = int(labels_to_index[key] / length)
                pre_layer = current_layer - 1
                u = labels_to_index[key]
                for inp_node_key in input_nodes:
                    v = labels_to_index[inp_node_key]
                    flow_value = nx.maximum_flow_value(
                        G, u, v, flow_func=nx.algorithms.flow.edmonds_karp
                    )
                    flow_values[u][pre_layer * length + v] = flow_value
                flow_values[u] /= flow_values[u].sum()

        return flow_values


    def compute_joint_attention(att_mat, add_residual=True):
        if add_residual:
            att_mat = att_mat + np.eye(att_mat.shape[1])[None, ...]
            att_mat = att_mat / att_mat.sum(axis=-1)[..., None]

        joint_attentions = np.zeros(att_mat.shape)

        layers = joint_attentions.shape[0]
        joint_attentions[0] = att_mat[0]
        for i in np.arange(1, layers):
            joint_attentions[i] = att_mat[i].dot(joint_attentions[i - 1])

        return joint_attentions


    def plot_attention_heatmap(att, s_position, t_positions, sentence):
        cls_att = np.flip(att[:, s_position, t_positions], axis=0)
        xticklb = input_tokens = list(
            itertools.compress(
                ["<cls>"] + sentence.split(),
                [i in t_positions for i in np.arange(len(sentence) + 1)],
            )
        )
        yticklb = [str(i) if i % 2 == 0 else "" for i in np.arange(att.shape[0], 0, -1)]
        ax = sns.heatmap(cls_att, xticklabels=xticklb, yticklabels=yticklb, cmap="YlOrRd")
        return ax


    def convert_adjmat_tomats(self,adjmat, n_layers, l):
        mats = np.zeros((n_layers, l, l))

        for i in np.arange(n_layers):
            mats[i] = adjmat[(i + 1) * l : (i + 2) * l, i * l : (i + 1) * l]

        return mats

    def viz_gene_embeddings(self, layer, resolutions=[0.5, 1, 4]):
        self.get_gene_embeddings(layer=layer)
        sc.pp.pca(adata, n_comps=50)
        sc.pp.neighbors(adata, n_pcs=50, n_neighbors=20)
        sc.tl.umap(adata, min_dist=0.3)

        [sc.tl.leiden(adata, resolution=0.5, key_added="leiden_res"+str(res)) for res in resolutions]
        sc.pl.umap(adata, color=["leiden_res"+str(res) for res in resolutions])

class GeneFormer_Explainer(BaseExplainer):
    def get_gene_embeddings(self, pickle_file="../geneformer/token_dictionary.pkl", layer=0) -> ad.AnnData:
        assert layer == 0, "GeneFormer only has one layer"
        di = pd.read_pickle(pickle_file)
        val = di.keys()
        adata = sc.AnnData(pd.DataFrame(index=list(val), data=self.model.embeddings.word_embeddings.weight.data.cpu()))
        return adata


class scGPT_Explainer(BaseExplainer):
    def __init__(self, vocab, gene_ids, values, src_key_padding_mask, **kwargs):
        super().__init__(**kwargs)
        self.vocab = vocab
        self.gene_ids = gene_ids
        self.values = values
        self.src_key_padding_mask = src_key_padding_mask

    def get_attention(self, layer_num=11, batch_size=8):
        torch.cuda.empty_cache()
        dict_sum_condition = {}
        self.model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            M = all_gene_ids.size(1)
            N = all_gene_ids.size(0)
            device = next(self.model.parameters()).device
            for i in tqdm(range(0, N, batch_size)):
                batch_size = all_gene_ids[i : i + batch_size].size(0)
                # Replicate the operations in self.model forward pass
                src_embs = self.model.encoder(
                    torch.tensor(all_gene_ids[i : i + batch_size], dtype=torch.long).to(
                        device
                    )
                )
                val_embs = self.model.value_encoder(
                    torch.tensor(all_values[i : i + batch_size], dtype=torch.float).to(
                        device
                    )
                )
                total_embs = src_embs + val_embs
                # total_embs = self.model.layer(total_embs.permute(0, 2, 1)).permute(0, 2, 1)
                # Send total_embs to attention layers for attention operations
                # Retrieve the output from second to last layer
                for layer in self.model.transformer_encoder.layers[:layer_num]:
                    total_embs = layer(
                        total_embs,
                        src_key_padding_mask=self.src_key_padding_mask[
                            i : i + batch_size
                        ].to(device),
                    )
                # Send total_embs to the last layer in flash-attn
                # https://github.com/HazyResearch/flash-attention/blob/1b18f1b7a133c20904c096b8b222a0916e1b3d37/flash_attn/flash_attention.py#L90
                qkv = self.model.transformer_encoder.layers[layer_num].self_attn.Wqkv(
                    total_embs
                )
                # Retrieve q, k, and v from flast-attn wrapper
                qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, h=8)
                q = qkv[:, :, 0, :, :]
                k = qkv[:, :, 1, :, :]
                # v = qkv[:, :, 2, :, :]
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
        return sm_attn_scores, gene_vocab_idx

    # return [pd.DataFrame(data=sm_attn_scores[i], columns=vocab.lookup_tokens(gene_vocab_idx), index=vocab.lookup_tokens(gene_vocab_idx)) for i in range(0,8)]

    # TODO: get the attention map with uncertainty across a cell

    # TODO: for an attention map, do GSEA and output any significant pathways

    # TODO: get the marker genes of a specific class or set of classes using something like LRP

    # TODO: get the effect of a knock out / knock in
