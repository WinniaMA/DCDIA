from ogb.utils.mol import smiles2graph # 紫薇修改
from torch_geometric.data import Data
import numpy as np
import torch
import dill
from sklearn.decomposition import PCA

def get_smiles_list(molecule, med_voc):
    smiles_all = []
    for index, ndc in med_voc.items():
        smilesList = list(molecule[ndc])
        smiles_all.append(smilesList)
    return smiles_all

# Step1:获取smiles_list
def graph_batch_from_smile(smiles_list):
    graphs_all = []
    for s_smile in smiles_list:
        graphs = []
        for x in s_smile:
            graph = smiles2graph(x)["node_feat"]
            graphs.append(graph)
        stacked_graph = np.vstack(graphs)
        graphs_flattened= np.array(stacked_graph).flatten().T
        graphs_all.append(graphs_flattened)
    print(len(graphs_all))
    # with open(output_file, 'wb') as f:
    #     dill.dump(node_features_final, f)
    return graphs_all

# pad
def pad_pca(graphs_all,output_file):
    max_length = max(len(seq) for seq in graphs_all)
    padded_sequences = np.array([np.pad(seq, (0, max_length - len(seq)), 'constant') for seq in graphs_all])
    pca = PCA(n_components=131)
    reduced_data = pca.fit_transform(padded_sequences)
    with open(output_file, 'wb') as f:
        dill.dump(reduced_data, f)
    return padded_sequences

# class FeatureExtractor(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(FeatureExtractor, self).__init__()
#         self.feature_embedding = nn.Linear(input_dim, output_dim)
#
#
#     def forward(self, node_features):
#         features_list = []
#         for feature in node_features:
#             # feature_tensor = torch.tensor(feature, dtype=torch.long)
#             # feature_tensor = feature.unsqueeze(0)
#             feature = self.feature_embedding(torch.tensor(feature, dtype=torch.float))# (1,131)
#             feature = feature.squeeze(0)
#             feature = feature.detach().numpy()
#             feature = np.transpose(feature)
#             features_list.append(feature)
#         print(len(features_list))
#         # features_list = [tuple_element[0] if isinstance(tuple_element, tuple) else tuple_element for tuple_element in
#         #                  features_list]
#         features_matrix = torch.stack(features_list, dim=0)
#         return features_matrix

def graph_batch_from_smile_molerec(smiles_list):
    edge_idxes, edge_feats, node_feats, lstnode, batch = [], [], [], 0, []
    graphs = [smiles2graph(x) for x in smiles_list]
    for idx, graph in enumerate(graphs):
        edge_idxes.append(graph['edge_index'] + lstnode)
        edge_feats.append(graph['edge_feat'])
        node_feats.append(graph['node_feat'])
        lstnode += graph['num_nodes']
        batch.append(np.ones(graph['num_nodes'], dtype=np.int64) * idx)

    result = {
        'edge_index': np.concatenate(edge_idxes, axis=-1),
        'edge_attr': np.concatenate(edge_feats, axis=0),
        'batch': np.concatenate(batch, axis=0),
        'x': np.concatenate(node_feats, axis=0)
    }
    result = {k: torch.from_numpy(v) for k, v in result.items()}
    result['num_nodes'] = lstnode
    return Data(**result)

if __name__ == '__main__':
    molecule_path = '../idx2SMILES.pkl'
    voc_path = '../data/voc_final.pkl'
    output_file = "../data/node_features.pkl"
    with open(molecule_path, 'rb') as Fin:
        molecule = dill.load(Fin)
    with open(voc_path, 'rb') as Fin:
        voc = dill.load(Fin)
    med_voc = voc['med_voc'].idx2word

    smiles_list = get_smiles_list(molecule, med_voc)
    graphs_all = graph_batch_from_smile(smiles_list)
    node_features = pad_pca(graphs_all,output_file)
    # feature_extroctor = FeatureExtractor(input_dim=2007, output_dim=131)
    # node_features_final = feature_extroctor(node_features)
    # with open(output_file, 'wb') as f:
    #     dill.dump(node_features_final, f)
    # print("hahah")

