"""
Copyright: Shaoheng Guan, Wellcome Sanger Institute 

File Description: This file aims to build a graph network using a compressed Hi-C matrix and the coverage of fragments as input and generate an adjacency probability matrix for the connections between gene fragments.

"""
import os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, ConcatDataset
from config import *
from utils import get_time_stamp, check_and_mkdir, DEBUG, get_newest_checkpoint
from utils_network import check_select_device
from restore_training_data import CompressedHicFragDataSet
from torchviz import make_dot
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.utils import coalesce



K_NEIBORS = 30  # The number of highest neighbors to sample
NUM_RANDOM_NEIGHBORS = 10  # The number of random neighbors to sample
NUM_RANDOM_NEIGHBORS_TEST = 5  # The number of random neighbors to sample for the model to predict the likelihood between the unselected training edges
NUM_RANDOM_SELECTED_EDGE_PER_EPOCH_PER_BATCH = 100 # The number of random selected edges per epoch per batch
FILTER_COEFFICIENT = 10 / (32768 + 1)  # The fragments smaller than # pixels will be filtered out
SIGMA=3.0   # Used to scale the likelihood matrix to avoid the 0 prediction

class MaxPoolingConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MaxPoolingConv, self).__init__(aggr='max')  # max pooling aggregation
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        # Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Linearly transform node feature matrix
        x = self.lin(x)

        # Normalize edge weights (optional for GCN)
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Perform the aggregation step (max pooling)
        return self.propagate(edge_index, x=x, norm=norm, edge_weight=edge_weight)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out
    

class Graph_HiC_Likelihood(nn.Module):
    def __init__(self, input_dim=74, hidden_dim=128, hidden_likelihood_dim=128, heads_num=4, num_edge_attr=5, output_dim=4):
        super().__init__()

        self.num_edge_attr = num_edge_attr
        
        self.conv_attention = GATConv(input_dim, hidden_dim, heads=heads_num)
        self.conv_list = nn.ModuleList([
            GCNConv(hidden_dim * heads_num, hidden_dim) for _ in range(num_edge_attr)
        ])
        self.conv_list_1 = nn.ModuleList([
            GCNConv(hidden_dim * self.num_edge_attr, hidden_dim) for _ in range(num_edge_attr)
        ])
        self.conv_list_2 = nn.ModuleList([
            GCNConv(hidden_dim * self.num_edge_attr, hidden_dim) for _ in range(num_edge_attr)
        ])
        self.conv_list_3 = nn.ModuleList([
            GCNConv(hidden_dim * self.num_edge_attr, hidden_dim) for _ in range(num_edge_attr)
        ])
        self.conv_list_4 = nn.ModuleList([
            GCNConv(hidden_dim * self.num_edge_attr, hidden_dim) for _ in range(num_edge_attr)
        ])
        
        self.link_likelihood = nn.Linear(hidden_dim * 2 * self.num_edge_attr, hidden_likelihood_dim)
        self.link_likelihood1 = nn.Linear(hidden_likelihood_dim, hidden_likelihood_dim)
        self.link_likelihood2 = nn.Linear(hidden_likelihood_dim, hidden_likelihood_dim)
        self.link_likelihood3 = nn.Linear(hidden_likelihood_dim, hidden_likelihood_dim)
        self.link_likelihood4 = nn.Linear(hidden_likelihood_dim, output_dim)

        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, batch:Batch):

        x = self.forward_x_embedding(batch)

        x = self.dropout(x) # [node_num, hidden_dim * self.num_edge_attr]

        x = self.predict_links(x, batch.edge_index_test)  # torch.any(x.isnan())
        return x  #  node embedding
    
    def predict_links(self, x, edge_index):
        """
            TODO there maybe some problem as the prediction of  [node_1, node_2] should be 
                different from the prediction of [node_2, node_1]

                检查这里的计算是不是有问题

                [node_1, node_2] : [head1->head1, head1->tail2, tail1->head2, tail1->tail2]
                [node_2, node_1] : [head1->head1, tail1->head2, head1->tail2, tail1->tail2]
        """
        edge_embeddings = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)
        edge_embeddings_1 = torch.cat([x[edge_index[1]], x[edge_index[0]]], dim=-1)
        tmp1 = self.link_likelihood4(self.relu(self.link_likelihood3(self.link_likelihood2(self.relu(self.link_likelihood1(self.relu(self.link_likelihood(edge_embeddings)))))))).squeeze(-1)
        tmp2 = self.link_likelihood4(self.relu(self.link_likelihood3(self.link_likelihood2(self.relu(self.link_likelihood1(self.relu(self.link_likelihood(edge_embeddings_1)))))))).squeeze(-1)[:, [0, 2, 1, 3]]
        return 0.5 * (tmp1 + tmp2)
    
    def forward_x_embedding(self, batch:Batch):
        x, edge_index, edge_attr, edge_index_test = batch.x, batch.edge_index, batch.edge_attr, batch.edge_index_test

        # four parallell layers of the graph convolution

        x = self.relu(
            self.conv_attention(x, edge_index)   # [64, 64 * heads_num]   
        )

        x = self.relu(
            torch.concat([self.conv_list[i](x, edge_index, edge_weight=edge_attr[:, i]) for i in range(self.num_edge_attr)], dim=-1)   # [64, 64 * self.num_edge_attr]
        )

        x = self.relu(
            torch.concat([self.conv_list_1[i](x, edge_index, edge_weight=edge_attr[:, i]) for i in range(self.num_edge_attr)], dim=-1)   # [64, 64 * self.num_edge_attr]
        )
        x = self.relu(
            torch.concat([self.conv_list_2[i](x, edge_index, edge_weight=edge_attr[:, i]) for i in range(self.num_edge_attr)], dim=-1)   # [64, 64 * self.num_edge_attr]
        )
        x = self.relu(
            torch.concat([self.conv_list_3[i](x, edge_index, edge_weight=edge_attr[:, i]) for i in range(self.num_edge_attr)], dim=-1)   # [64, 64 * self.num_edge_attr]
        )

        x = self.relu(
            torch.concat([self.conv_list_4[i](x, edge_index, edge_weight=edge_attr[:, i]) for i in range(self.num_edge_attr)], dim=-1)   # [64, 64 * self.num_edge_attr]
        )
        return x


def sample_top_k_neighbors(
        edge_index, edge_weight, num_nodes, truth, 
        k_neighbors:int, 
        random_neighbor_num:int, 
        random_neighbor_test_num:int,
        device:torch.device,
        )->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
        Sample the top k neighbors with the highest edge weight for each node in the graph.

        Note: Not only give the top k neibors, but also give some random selected neibors
    """
    # Initialize lists to store the selected edges
    topk_edge_index = []
    topk_edge_weight = []
    topk_edge_truth = []
    topk_edge_test_index = []

    num_edges = len(edge_weight)
    edges_selected_mask = torch.zeros(num_edges, dtype=torch.bool, device=device)

    for node in range(num_nodes):
        # Find all edges where the current node is the source
        mask_selected = torch.logical_or(edge_index[0] == node, edge_index[1] == node)
        num_neighbors = mask_selected.sum().item()
        node_edges = edge_index[:, mask_selected]
        node_edge_weights = edge_weight[mask_selected]
        node_truth = truth[mask_selected]

        node_selected_edges_global_idx = torch.arange(num_edges, device=device)[mask_selected]
        # edges_selected_mask[mask_selected] = False  # NOTE make sure all of the edges just be selected once
        node_edges_selected_mask = torch.zeros([num_neighbors, ], dtype=torch.bool, device=device)
        node_edges_selected_mask[edges_selected_mask[node_selected_edges_global_idx]] = True
        
        # If the node has fewer than k neighbors, take all; otherwise, take top k
        if num_neighbors > k_neighbors:
            node_edge_weights_max = node_edge_weights[:, :4].max(axis=1).values  # while select the nodes just consider the link_score calculated according to YaHS
            _, topk_indices = torch.topk(node_edge_weights_max, k_neighbors)  # top_indices is the local topk index
            topk_indices_test = topk_indices = topk_indices[~node_edges_selected_mask[topk_indices]]
            node_edges_selected_mask[topk_indices] = True
            node_edges_not_selected_num = (~node_edges_selected_mask).sum()
            # add random neibors to this node, to let the model know the information from other nodes with weak link to it 
            if random_neighbor_num>0 and node_edges_not_selected_num > random_neighbor_num:
                random_indices = torch.arange(num_neighbors, device=device)[~node_edges_selected_mask][torch.randperm(node_edges_not_selected_num, device=device)[:random_neighbor_num]]
                topk_indices_test = topk_indices = torch.concat([topk_indices, random_indices], dim=0) if len(topk_indices) != 0 else random_indices.clone()
                node_edges_selected_mask[random_indices] = True
                node_edges_not_selected_num = (~node_edges_selected_mask).sum()
            # add some random selected edges for train the model to predict the likelihood between the unselected training edges, NOTE while without inputing the edge in calculating
            if random_neighbor_test_num>0 and node_edges_not_selected_num >  random_neighbor_test_num:
                random_indices = torch.arange(num_neighbors, device=device)[~node_edges_selected_mask][torch.randperm(node_edges_not_selected_num, device=device)[:random_neighbor_test_num]]
                topk_indices_test = torch.concat([topk_indices_test, random_indices], dim=0) if len(topk_indices_test) != 0 else random_indices.clone()
            topk_weights = node_edge_weights[topk_indices]
            topk_edges = node_edges[:, topk_indices]
            topk_edges_test = node_edges[:, topk_indices_test]
            topk_truth = node_truth[topk_indices_test]
            edges_selected_mask[node_selected_edges_global_idx[topk_indices]] = True 
        else:
            topk_weights = node_edge_weights[~node_edges_selected_mask]
            topk_edges_test = topk_edges = node_edges[:, ~node_edges_selected_mask]
            topk_truth = node_truth[~node_edges_selected_mask]
            edges_selected_mask[node_selected_edges_global_idx]=True

        if len(topk_weights) == 0: continue
        topk_edge_index.append(topk_edges)
        topk_edge_weight.append(topk_weights)
        topk_edge_truth.append(topk_truth)
        topk_edge_test_index.append(topk_edges_test)

    # Concatenate all the selected edges and weights
    topk_edge_index = torch.cat(topk_edge_index, dim=1)
    topk_edge_test_index = torch.cat(topk_edge_test_index, dim=1)
    topk_edge_weight = torch.cat(topk_edge_weight, dim=0)
    topk_edge_truth = torch.cat(topk_edge_truth, dim=0)

    return topk_edge_index, topk_edge_weight, topk_edge_truth, topk_edge_test_index


def restore_model(
        file_dir:str, 
        device:torch.device)->Tuple[Graph_HiC_Likelihood, torch.optim.Adam, int, torch.optim.lr_scheduler.StepLR]:
    # =============================
    #         intialize model
    #         intialize model
    print("Initializing model...")
    model = Graph_HiC_Likelihood().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # TODO after the weight_decay over 0.006, the model will not be trained...
    epoch_already = 0
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.98)

    # =============================
    #           restore ckpt
    fname = get_newest_checkpoint(file_dir)
    if fname:
        checkpoint = torch.load(os.path.join(file_dir, fname), map_location=device)
        # Restoring model and optimiser state
        # Restoring model and optimiser state
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch_already = checkpoint['epoch']
        loss_train = checkpoint['loss_train']
        loss_vali = checkpoint['loss_vali']
        print("\n=============================")
        print(f"Restored: {os.path.join(file_dir, fname)}\n"
              f"Epoch: {epoch_already}\n"
              f"Loss_train: {loss_train:.4e}\n"
              f"Loss_vali: {loss_vali:.4e}\n"
              )

    return model, optimizer, epoch_already, scheduler


def restore_and_organize_training_validation_data(
        dirs_save_train_data:List[str], 
        coverage_required_flag=False, 
        madeup_flag=False,
        debug_flag=False,
        )->Tuple[DataLoader, DataLoader]:
    print("\n=============================")
    print("Data Loading...")
    dataset = CompressedHicFragDataSet(
        dirs_save_train_data, 
        coverage_required_flag=coverage_required_flag, 
        madeup_flag=madeup_flag,
        debug_flag=debug_flag)
    # Determine the size of the dataset and calculate the size of the training and test sets
    dataset_size = len(dataset)
    train_size = int(TRAIN_SET_RATIO * dataset_size)
    vali_size = dataset_size - train_size
    # Split the dataset into training and test sets using the random_split method
    # Split the dataset into training and test sets using the random_split method
    train_dataset, vali_dataset = random_split(dataset, [train_size, vali_size])
    # Create data loaders for the training set and test set separately
    # Create data loaders for the training set and test set separately
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    vali_loader = DataLoader(vali_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Number of data Loaded: {len(dataset)}"
          f"\nNumber of training set: {train_size}"
          f"\nNumber of validation set: {vali_size}"
          )
    return train_loader, vali_loader


def train():
    print("\n=============================")
    print("Training...")
    model.train()

    for epoch in range(epoch_already, epoch_already + N_EPOCHS):
        start_epoch = time.time()
        for batch_data in train_graphs:
            optimizer.zero_grad()
            # Forward pass through the model

            # num_edges = batch_data.truth.shape[0]
            # mask_selected_edges = torch.randint(0, num_edges, (NUM_RANDOM_SELECTED_EDGE_PER_EPOCH_PER_BATCH,), device=device) 

            pre_batch = model.forward(batch=batch_data)

            # check the prediction while debugging
            """
            mask = [
                    torch.logical_or(
                        torch.logical_and(batch_data.edge_index[0] == i[0], batch_data.edge_index[1] == i[1]), 
                        torch.logical_and(batch_data.edge_index[0] == i[1], batch_data.edge_index[1] == i[0])) 
                        for i in batch_data.edge_index_test[:, mask_selected_edges].T]
            mask_sum = [i.sum().item() for i in mask]
            hic_signal = [batch_data.edge_attr[i] for i in mask]
            hic_signal = [i[0].item() if i.size(0)>0 else -1 for i in hic_signal]

            for num, i in enumerate(mask_selected_edges):
                print(
                    f"[{batch_data.edge_index_test[0, i]}, {batch_data.edge_index_test[1, i]}] \t hic: {hic_signal[num]:.2e} \t pre: {pre_batch[num].item():.2e} \t truth: {batch_data.truth[i].item():.2e}"
                )
            """
            # Compute loss
            # loss = loss_cal(torch.log(pre_batch + 1e-8), torch.log(batch_data.truth + 1e-8))
            loss = loss_cal(pre_batch, batch_data.truth)
            loss.backward()
            optimizer.step()

        scheduler.step()

        if epoch%SHOW_PER_EPOCH==0:
            loss_vali, batch_num = 0, 0
            for batch_data in vali_graphs:
                with torch.no_grad():
                    pre_batch = model.forward(batch_data)
                    # loss_vali += loss_cal(torch.log(pre_batch + 1e-8), torch.log(batch_data.truth + 1e-8)).item()
                    loss_vali += loss_cal(pre_batch, batch_data.truth).item()
                batch_num += 1
            loss_vali /= batch_num
            print(f'Epoch [{epoch}/{N_EPOCHS + epoch_already}], Loss train: {loss.item():.4e}, Loss vali: {loss_vali:.4e}, EpochTime: {(time.time() - start_epoch)/60:.2f}mins ConsumedTime: {(time.time() - start_time)/60:.2f}mins')  # 

        if epoch % 1000==0 and epoch!=epoch_already:
            save_checkpoint(
                epoch=epoch,
                loss=loss.item(),
                loss_vali=loss_vali)
    return epoch, loss.item(), loss_vali


def get_graph_batch(
        name_batch:Tuple[str],
        hic_mx:torch.Tensor, 
        frag_lens:torch.Tensor, 
        frag_coverage:torch.Tensor, 
        likelihood_mx:torch.Tensor, 
        device:torch.device, 
        n_frags:List[int],
        frag_start_loc_relative:torch.Tensor,
        frag_mass_center_relative:torch.Tensor, 
        frag_average_density_global:torch.Tensor, 
        frag_average_hic_interaction:torch.Tensor,
        frag_coverage_flag:torch.Tensor,
        frag_repeat_density_flag:torch.Tensor, 
        frag_repeat_density:torch.Tensor,
        distance_mx:torch.Tensor,
        total_pixel_len:torch.Tensor,
        k_neighbors:int=K_NEIBORS,
        random_neighbor_num:int=NUM_RANDOM_NEIGHBORS,
        random_neighbor_test_num:int=NUM_RANDOM_NEIGHBORS_TEST,
        filter_coefficient:float=FILTER_COEFFICIENT,
        selected_edges_ratio:float=0.2,
        random_selected_edges_ratio:float=0.1,
        select_edges_mode:str="global_topk",
        output_selections:str="likelihood",   # distance or likelihood
        )->Batch:
    """ 
        Finished:
        add the filter_coefficient to filter the small fragments out
            Remove small fragments that don't belong to any chormosome and aren't moved to make the training set simple first
            Add the centre of gravity coordinates of the hic signal for each fragment
            Remove small fragments that don't belong to any chormosome and aren't moved to make the training set simple first
            Add the centre of gravity coordinates of the hic signal for each fragment


        organize the graph data for the graph network

        Args:  
            hic_mx (torch.Tensor): The compressed Hi-C matrix.
            frag_lens (torch.Tensor): The length of the fragments.
            frag_coverage (torch.Tensor): The coverage of the fragments.
            likelihood_mx (torch.Tensor): The likelihood matrix.
            device (torch.device): The device to run the model.
            n_frags (List[int]): The number of fragments in each sample.
            sigma (float): Used to scale the likelihood matrix to avoid the 0 prediction.
            k (int): The number of highest neighbors to sample.
    """

    data_list = []

    num_samples = len(n_frags)
    for sample_index in range(num_samples):
        name = name_batch[sample_index]
        node_num = n_frags[sample_index]
        total_len = frag_lens[sample_index].sum()
        frag_lens_tmp = frag_lens[sample_index, :node_num] / total_len
        mask_is_effective = (frag_lens_tmp >= filter_coefficient) # if not full_edges_flag else torch.ones(node_num, dtype=torch.bool) # use this to filter the small fragments out. (FINISHED: need to consider: all of the node should be considered while full edge is considered??, no just consider the fragments with at least 5 pixels)
        node_effective_num = mask_is_effective.sum().item()
        # mask_is_effective_2d = mask_is_effective.unsqueeze(0) * mask_is_effective.unsqueeze(1)
        # Feature matrix for the nodes  1 + 1 + 1 + 1 + 2 + 64 = 70

        """
         finished making sure that the density and hic interactions are between 0 and 1, which is already handled in restore_training_data.py
        """
        x = torch.cat([
            frag_lens[sample_index, :node_num].unsqueeze(1) / total_len,  # 1 normalize the length of the fragments
            frag_coverage[sample_index, :node_num].unsqueeze(1),  # 1 averaged coverage of the fragments
            frag_repeat_density[sample_index, :node_num].unsqueeze(1),  # 1 the repeat density of the fragments
            torch.full((node_num, 1), (1.0 if total_len > 32768 else 0.), dtype=torch.float, device=device),  # 1 distinguish the normal and high resolution
            frag_start_loc_relative[sample_index, :node_num].unsqueeze(1),  # 1 the relative start location of the fragments
            frag_mass_center_relative[sample_index, :node_num],  # 2 the relative mass center of the fragments
            frag_average_density_global[sample_index, :node_num],  # 64 the averaged density of the fragments
            frag_coverage_flag[sample_index].unsqueeze(0).repeat(node_num).unsqueeze(1),  # 1 the coverage flag of the fragments
            frag_repeat_density_flag[sample_index].unsqueeze(0).repeat(node_num).unsqueeze(1),  # 1 the coverage flag of the fragments
            ((1.0 / node_num) if node_num!= 0 else torch.tensor(2.0, device=device)).unsqueeze(0).repeat(node_num).unsqueeze(1),  # 1 the coverage flag of the fragments
            ], dim=1).float()[mask_is_effective]   # just select the effective fragments
        
        selected_node_idx = torch.arange(node_num, device=device)[mask_is_effective]

        # Edge index for the graph
        edge_index = torch.combinations(selected_node_idx, r=2).t().long()
        edge_index_new = torch.combinations(torch.arange(node_effective_num, device=device),  r=2).t().long()

        # check the edge attributes
        """
        tmp = hic_mx[sample_index].mean(dim=0)[:node_num, :node_num].cpu().numpy() 
        plt.imshow(tmp, cmap='jet'); plt.colorbar(); plt.tight_layout() ; plt.savefig(os.path.join("test_fig", f'test_hic.png'), dpi=300); plt.close("all")
        tmp = likelihood_mx[sample_index, : 2 * node_num, :2*node_num].reshape(node_num, 2, node_num, 2).mean(axis=(1, 3)).cpu().numpy()
        plt.imshow(tmp, cmap='jet'); plt.colorbar(); plt.tight_layout() ; plt.savefig(os.path.join("test_fig", f'test_likelihood.png'), dpi=300); plt.close("all")
        """

        # Edge attributes (hic_mx values)
        hic_mx_tmp = torch.concat([hic_mx[sample_index], frag_average_hic_interaction[sample_index].unsqueeze(0)], dim=0)
        edge_attr = hic_mx_tmp[:, edge_index[0], edge_index[1]].transpose(1, 0).float()  # indexing the attributes of the selected edges

        # check the edge attributes
        """
        likelihood_mx_tmp = sigma * likelihood_mx[sample_index, : 2 * node_num, :2*node_num].cpu().numpy()
        plt.imshow(likelihood_mx_tmp, cmap='jet'); plt.colorbar(); plt.tight_layout() ; plt.savefig(os.path.join("test_fig", f'test_likelihood.png'), dpi=300); plt.close("all")

        plt.imshow(likelihood_mx[sample_index, : 2 * node_num, :2*node_num].reshape(node_num, 2, node_num, 2).mean(axis=(1, 3)).cpu().numpy(), cmap='jet'); plt.colorbar(); plt.tight_layout() ; plt.savefig(os.path.join("test_fig", f'test_likelihood_mean.png'), dpi=300); plt.close("all")
        """

        if output_selections == "likelihood":
            # Ground truth for the edges
            # used to predict the averaged likelihood (between head) of the edges
            # Ground truth for the edges
            # used to predict the averaged likelihood (between head) of the edges
            if likelihood_mx[sample_index].shape != torch.Size([2 * MAX_NUM_FRAG, 2 * MAX_NUM_FRAG]):
                raise RuntimeError(f"Please check the shape of the likelihood matrix, the shape is {likelihood_mx[sample_index].shape}")
            tmp = 0.5 * (likelihood_mx[sample_index, : 2*node_num, :2*node_num] + likelihood_mx[sample_index, : 2*node_num, :2*node_num].T)
            tmp = torch.where(tmp == 0., torch.tensor(-1.0, device=device), tmp)  # set the 0 likelihood to -1 
            likelihood_mx_tmp = tmp.reshape(node_num, 2, node_num, 2).transpose(1, 2).reshape((node_num, node_num, 4)).float()  # 目前计算的是likelihood的均值
            # FINISHED This is a very big mistake ！！！！！Symmetry should not be done after transforming the shape to [node_num, node_num, 4] ！！！！！It should be done when the shape is [2 * node_num, 2 * node_num].
            # Because [1, 2, 3, 4] and [1, 2, 3, 4] don't really correspond.
            # The first [1, 2, 3, 4] means [head-head, head-tail, tail-head, tail-tail] respectively, and the second [head-head, tail-head, head-tail, tail-tail].
            '''
            mx = torch.arange(36).reshape(6, 6)
            mx = mx + mx.T 
            mx = mx.reshape(3,2,3,2).transpose(1,2).reshape(3,3,4)

            mx1 = torch.arange(36).reshape(6, 6)
            mx1 = mx1.reshape(3,2,3,2).transpose(1,2).reshape(3,3,4)
            mx1 = mx1 + mx1.transpose(0, 1)
            '''
            truth = likelihood_mx_tmp[edge_index[0], edge_index[1]] # indexing the likelihood of the selected edges 

        elif output_selections == "distance":
            distance_mx_tmp = distance_mx[sample_index, :2*node_num, :2*node_num].float() / total_pixel_len[sample_index].float()
            distance_mx_tmp = torch.where(distance_mx_tmp > 1.0, torch.tensor(1.0, device=device), distance_mx_tmp)
            distance_mx_tmp = distance_mx_tmp.reshape(node_num, 2, node_num, 2).transpose(1, 2).reshape((node_num, node_num, 4))
            truth = distance_mx_tmp[edge_index[0], edge_index[1]] # indexing the likelihood of the selected edges 
        else:
            raise RuntimeError(f"Please check the output_selections which is ({output_selections}), the value should be 'distance' or 'likelihood'")

        # Sample top k neighbors with highest edge weight of every node in the graph
        # TODO whether use the topk or use the threshold to select the edges??? 
        # check which one is better
        # actually, I don't think topk is a good idea, as there are some fragments with very low link score, 
        # while they are still connected with the other fragments
        # so, maybe select the edges with global interatction score is a better idea
        if select_edges_mode=="single_topk":
            edge_index_topk, edge_attr_topk, truth_topk, edge_index_topk_test = sample_top_k_neighbors(
                edge_index_new, 
                edge_attr, 
                node_effective_num,    # NOTE node_number should the number of effecitive nodes be given here
                truth, 
                k_neighbors=k_neighbors, 
                random_neighbor_num=random_neighbor_num, 
                random_neighbor_test_num=random_neighbor_test_num, 
                device=device)
        elif select_edges_mode == "global_topk": # select the first 10% of the edges to build the graphs
            tmp = edge_attr[:, :4].mean(-1).reshape(-1)
            # plt.hist(tmp.numpy(), bins=100); plt.savefig(os.path.join("test_fig", f"test_{name}_edge_distribution.png"), dpi=300); plt.close("all")
            # tmp_ = torch.sort(tmp, descending=True)
            # for i in [ 0.1]:
            #     print(f"{name}: {i} percentile: {tmp_.values[int(i * len(tmp))]:.2e}")
            topk_indices = torch.topk(tmp, int(selected_edges_ratio * len(tmp))).indices
            # ramdomly select some edges from the rest of the edges 
            left_after_topk = torch.ones(len(tmp), dtype=torch.bool, device=device)
            left_after_topk[topk_indices] = False
            unseen_indices = torch.arange(len(tmp), device=device)[left_after_topk]
            rand_indices = unseen_indices[torch.randperm(unseen_indices.size(0))[:int(len(tmp) * random_selected_edges_ratio)]]
            selected_indices = torch.concat([topk_indices, rand_indices], dim=0)
            selected_indices = selected_indices[torch.randperm(selected_indices.size(0))]
            edge_attr_topk = edge_attr[selected_indices]
            edge_index_topk = edge_index_new[:, selected_indices]  # NOTE the edge_index should be calculated according to the new node index
        
        # full edges are required
        edge_index_topk_test = edge_index_new
        truth_topk = truth

        # FINISHED: should save the map of node_idx_new to original node_idx
        data = Data(
            x=x, 
            edge_index=edge_index_topk, 
            edge_attr=edge_attr_topk, 
            truth=truth_topk, 
            edge_index_test=edge_index_topk_test, 
            new_node_idx_to_original_idx=selected_node_idx,
            )
        data_list.append(data)

    # NOTE in Batch, name containing 'index' will be treated as the local index of the local graph, thus will be updated into the global index with regarding to the big (global) graph
    # thus, both of `edge_index_test` and `edge_index` are updated into global index, while `new_node_idx_to_original_idx` remains local index 
    batch = Batch.from_data_list(data_list)

    # FINISHED check if there is duplicated edges:
    new_index = coalesce(
        edge_index=batch.edge_index
    )
    if (len(new_index)!=len(batch.edge_index)):
        raise RuntimeError(f"Please check, there are duplicated as new_index(len:{len(new_index)}) != batch.edge_index(len: {len(batch.edge_index)})")
    return batch


def get_graphs(data_loader:DataLoader, device:torch.device)->Tuple[List[Batch], List[List[str]]]:
    """
        Get and save the graph data in mem.
    """
    names = []
    batch_graphs = []
    for batch in data_loader:
        hic_mx, frag_id, scaffid, name_batch, \
            n_frags, frag_lens, distance_mx, likelihood_mx, \
            frag_coverage, \
            frag_start_loc_relative, frag_mass_center_relative,\
            frag_average_density_global, frag_average_hic_interaction, frag_coverage_flag, \
            frag_repeat_density_flag, frag_repeat_density, total_len_in_pixel  = batch
        
        hic_mx = hic_mx.to(device)
        likelihood_mx = likelihood_mx.to(device)
        frag_coverage = frag_coverage.to(device)
        n_frags = n_frags.to(device)
        frag_lens = frag_lens.to(device)
        distance_mx = distance_mx.to(device)
        total_len_in_pixel = total_len_in_pixel.to(device)

        frag_start_loc_relative = frag_start_loc_relative.to(device)           # [n_frags]
        frag_mass_center_relative = frag_mass_center_relative.to(device)       # [n_frags, 2]
        frag_average_density_global = frag_average_density_global.to(device)   # [n_frags, num_divided_parts] 
        frag_average_hic_interaction = frag_average_hic_interaction.to(device) # [n_frags, n_frags]

        frag_coverage_flag = frag_coverage_flag.to(device)                     # [1] every sample has one label to show wheather the coverage is available
        frag_repeat_density_flag = frag_repeat_density_flag.to(device)         # [1] every sample has one label to show wheather the repeat density is available
        frag_repeat_density = frag_repeat_density.to(device)                   # [n_frags,]

        batch_graphs.append(
            get_graph_batch(
                name_batch,
                hic_mx, frag_lens, frag_coverage, likelihood_mx, device, n_frags,
                frag_start_loc_relative, 
                frag_mass_center_relative, 
                frag_average_density_global, 
                frag_average_hic_interaction,
                frag_coverage_flag,
                frag_repeat_density_flag, frag_repeat_density,
                distance_mx, total_len_in_pixel,
                )
            )
        names.append(name_batch)
    return batch_graphs, names



def get_paded_tensor(num_samples:int, n_frags:List[int], device:torch.device)->Tuple[torch.Tensor, torch.Tensor]:
    """
        Get the paded tensor for the input and output

        Args:
            num_samples (int): The number of samples.
            n_frags (List[int]): The number of fragments in each sample.

        Returns:    
            The paded tensor for the input and output.
    """
    is_paded = torch.ones([num_samples, MAX_NUM_FRAG], device=device, dtype=torch.bool)  # [n_frags, MAX_NUM_FRAG] 
    is_paded_2d = torch.ones([num_samples, MAX_NUM_FRAG, MAX_NUM_FRAG], device=device, dtype=torch.bool)
    is_paded_2d_output = torch.ones([num_samples, 2*MAX_NUM_FRAG, 2*MAX_NUM_FRAG], device=device, dtype=torch.bool)
    for i in range(num_samples):
        is_paded[i, :n_frags[i]] = False
        is_paded_2d[i, :n_frags[i], :n_frags[i]] = False
        is_paded_2d_output[i, :2*n_frags[i], :2*n_frags[i]] = False
    return is_paded, is_paded_2d, is_paded_2d_output


def save_checkpoint(epoch:int, loss:float, loss_vali:float)->None:
    model_save_dir = PATH_SAVE_CHECHPOINT_GRAGPH_LIKELIHOOD
    fname = os.path.join(model_save_dir, f'checkpoint_{get_time_stamp()}.pth')
    checkpoint = {
        'epoch': epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss_train': loss,
        "loss_vali": loss_vali,
    }
    torch.save(checkpoint, fname)
    print(f"\n============================="
          f"\nSaved checkpoint at epoch {epoch+1}"
          f"\nloss train {loss:.4e}"
          f"\nloss vali {loss_vali:.4e}"
          f"\nSaved as {fname}")
    print()


def plot_model()->None:
    tmp_iter = iter(vali_loader)
    optimizer.zero_grad()
    hic_mx, frag_id, scaffid, name, n_frags, frag_lens, distance_mx, likelihood_mx, frag_coverage = next(tmp_iter)  
    hic_mx, likelihood_mx = hic_mx.to(device), likelihood_mx.to(device)
    frag_coverage = frag_coverage.to(device)
    frag_lens = frag_lens.to(device)
    # get the paded mask in 1d and 2d
    is_paded, is_paded_2d, is_paded_2d_output = get_paded_tensor(len(hic_mx), n_frags, device)
    
    likelihood_pre = model.forward(
        hic_mx, frag_coverage, len_frags=frag_lens, n_frags=n_frags, is_paded = is_paded, is_paded_2d=is_paded_2d, device=device)

    viz = make_dot(likelihood_pre, params=dict(
        list(model.named_parameters())+ \
        [("Compressed_hic", hic_mx), ("Frag_len", frag_lens), ("Frag_coverage", frag_coverage)]))
    viz.format = 'png'
    viz.render(os.path.join("./likelihood_model"))
    sys.exit()
    return  


def read_data(
        madeup_train_dir_list:List[str]=None,
        device=torch.device("cpu"),
        converage_required_flag=True,
        debug_flag=False,
        )->Tuple[List[Batch], List[List[str]], List[Batch], List[List[str]]]:
    
    train_loader, vali_loader = restore_and_organize_training_validation_data(
        madeup_train_dir_list, 
        coverage_required_flag=converage_required_flag, 
        madeup_flag=True,
        debug_flag=debug_flag)
    print("="*20)
    print(f"Number of train data combined: {len(train_loader.dataset)}")
    print(f"Number of vali data combined: {len(vali_loader.dataset)}\n")

    # =============================
    #         build graphs
    print("\n"+ "="*20)
    print(f"Building training graphs... \n  batch_size: {BATCH_SIZE}")   
    train_graphs, train_names = get_graphs(train_loader, device=device)
    print(f"Building vali graphs... \n  batch_size: {BATCH_SIZE}")   
    vali_graphs, vali_names = get_graphs(vali_loader, device=device)
    return train_graphs, train_names, vali_graphs, vali_names, train_loader, vali_loader


if __name__ == "__main__":

    print(f"Start time: {get_time_stamp()}")

    print("\n"+ "="*20)
    print(f"Dir to save the trained network: {PATH_SAVE_CHECHPOINT_GRAGPH_LIKELIHOOD}")
    check_and_mkdir(PATH_SAVE_CHECHPOINT_GRAGPH_LIKELIHOOD)

    loss_cal = torch.nn.MSELoss()

    start_time = time.time()
    torch.manual_seed(RANDOM_SEED)  # just set the seed within the __main__ function, to avoid the random seed affect the other parts of the code

    # cuda
    device = check_select_device()
    if not DEBUG:
        if not torch.cuda.is_available():
            print("Cuda not available while submitted to server, thus exit.")
            sys.exit()
        elif (gpu_mem:=torch.cuda.get_device_properties(0).total_memory) < 30:
            print(f"Exist with gpu with mem {gpu_mem}G < 40G. Exit.")
            sys.exit()
        else:
            print(f"The mem is: {torch.cuda.get_device_properties(0).total_memory}")

    # =============================
    #            Initialising the model
    model, optimizer, epoch_already, scheduler = restore_model(
        PATH_SAVE_CHECHPOINT_GRAGPH_LIKELIHOOD, 
        device)

    # =============================
    #           training data loading
    train_graphs, train_names, vali_graphs, vali_names, train_loader, vali_loader = read_data(
        [DIR_SAVE_TRAIN_DATA_HUMAN],
        device=device,
        converage_required_flag=True,
        debug_flag=DEBUG)
    
    # =============================
    #      model training
    epoch, loss, loss_vali = train()

    # =============================
    #          save the model
    save_checkpoint(epoch=epoch, loss=loss, loss_vali=loss_vali)
