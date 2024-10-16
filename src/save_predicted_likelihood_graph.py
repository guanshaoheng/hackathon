import torch
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader, ConcatDataset
from graphNN_hic_likelihood import Graph_HiC_Likelihood, restore_model, Batch, read_data
from config import *
from config_plot import *
from utils_network import check_select_device
from utils import check_and_mkdir, get_time_stamp, DEBUG
from tqdm import tqdm
from sort_likelihood import sort_fragments_according_likelihood_or_distance_matrix, get_curated_fragments_order, output_into_file

'''
export PYTHONPATH=src:$PYTHONPATH
nohup /nfs/users/nfs_s/sg35/.conda/envs/autohic_new/bin/python -u src/save_predicted_likelihood_graph.py -m -c 0 > results/predicted_likelihood_graph_madeup/save_predicted_likelihood_graph_madeup.log 2>&1 &
'''

"""
    This script is used to predict the likelihood matrix from the compressed Hi-C matrix
"""

# set the device
device = check_select_device()

# set the random seed to make sure the validation set is the same as those in training
torch.manual_seed(RANDOM_SEED)

model, _, epoch_already, _ = restore_model(PATH_SAVE_CHECHPOINT_GRAGPH_LIKELIHOOD, device)
model.eval()

TEST_FLAG = True

dir_save_train_data = DIR_SAVE_TRAIN_DATA_HUMAN_TEST if TEST_FLAG else DIR_SAVE_TRAIN_DATA_HUMAN
dir_save_full_hic_data = DIR_SAVE_FULL_HICMX_HUMAN_TEST if TEST_FLAG else DIR_SAVE_FULL_HICMX_HUMAN
path_save_predicted_results = PATH_SAVE_PREDICTED_LIKELIHOOD_GRAPH_TEST if TEST_FLAG else PATH_SAVE_PREDICTED_LIKELIHOOD_GRAPH

def save_batch_into_file(
        graph_batch:Batch, 
        name_list:List[str], 
        loader:DataLoader,
        train_data_flag:bool=False, plot_flag=False,
        file=sys.stdout,
        save_dir:str=path_save_predicted_results,
        )->None:

    check_and_mkdir(save_dir)
    
    with torch.no_grad():
        pre_batch = model.forward(graph_batch)
    
    hic_mx, frag_id, scaffid, names, \
            n_frags, frag_lens, distance_mx, likelihood_mx, \
            frag_coverage, \
            frag_start_loc_relative, frag_mass_center_relative,\
            frag_average_density_global, frag_average_hic_interaction, frag_coverage_flag, \
            frag_repeat_density_flag, frag_repeat_density, total_pixel_len  = loader
    node_idx_by_graph = graph_batch.batch
    edge_idx_by_graph = graph_batch.batch[graph_batch.edge_index_test[0]]
    num_graphs = graph_batch.num_graphs
    for i in range(num_graphs):
        node_mask = (node_idx_by_graph == i)
        edge_mask = (edge_idx_by_graph == i)
        x = graph_batch.x[node_mask]
        new_idx_to_original_idx = graph_batch.new_node_idx_to_original_idx[node_mask]
        n_frags_all = n_frags[i]
        node_num = graph_batch.ptr[i + 1] - graph_batch.ptr[i]
        edge_index_test = graph_batch.edge_index_test[:, edge_mask] - graph_batch.ptr[i]

        # Finished check the likelihood of all the edges between fragments whose length is no smaller than 5. 
        #   during training, just the edges (with high interation and some randomly selected) are included 
        #   while during checking all of the edges are included
        truth_mx = torch.zeros([node_num, node_num, 4], dtype=torch.float32, device=device)
        pre_mx = torch.zeros([node_num, node_num, 4], dtype=torch.float32, device=device)
        truth_mx[edge_index_test[0], edge_index_test[1]] = graph_batch.truth[edge_mask]
        pre_mx[edge_index_test[0], edge_index_test[1]] = pre_batch[edge_mask]
        
        name = name_list[i]
        if name != names[i]:
            raise ValueError(f"Error: the name is not consistent! {name} vs {names[i]}")

        chromosomes_order_curated, chromosomes_inverse_flag_curated = get_curated_fragments_order(
            scaffid=scaffid[i],
            frag_id=frag_id[i],
            frag_lens=frag_lens[i],
            name=name,
            echo_flag=True,
        )
        chromosomes_order, chromosomes_inverse_flag = sort_fragments_according_likelihood_or_distance_matrix(
            pre_mx,
            # truth_mx,
            nfrags=node_num,
            frag_lens=frag_lens[i],
            name=name,
            echo_flag=True,
            mx_type="likelihood",
        )
        # map the order to the original order, as there are some small fragments are filter out during orgainizing the graph data
        chromosomes_order = [[new_idx_to_original_idx[idx-1].item()+1 for idx in chromosome] for chromosome in chromosomes_order]
        filtered_out_idx = get_the_filtered_fragments_order(new_idx_to_original_idx, n_frags_all)
        if len(filtered_out_idx) > 0:
            chromosomes_order += [filtered_out_idx]
            chromosomes_inverse_flag += [[False] * len(filtered_out_idx)]

        if plot_flag:
            plt.imshow(pre_mx.mean(dim=-1).cpu().detach().numpy(), cmap='jet'); plt.colorbar(); plt.tight_layout() ; plt.savefig(os.path.join(save_dir, f'{"train" if train_data_flag else "vali"}_{name}_pre.png'), dpi=300); plt.close("all")
            plt.imshow(truth_mx.mean(dim=-1).cpu().detach().numpy(), cmap='jet'); plt.colorbar(); plt.tight_layout() ; plt.savefig(os.path.join(save_dir, f'{"train" if train_data_flag else "vali"}_{name}_original.png'), dpi=300); plt.close("all")
            plt.close('all')

        output_into_file(
            chromosomes_original=chromosomes_order_curated,
            inversed_flag_original=chromosomes_inverse_flag_curated,
            chromosomes_ai=chromosomes_order,
            inversed_flag_ai=chromosomes_inverse_flag,
            file=file,
            train_data_flag=train_data_flag,
            name=name,
        )
    return 


def get_the_filtered_fragments_order(new_idx_to_original_idx: torch.Tensor, n_frags_all:int)->List[int]:
    """
        get the filtered fragments order
    """
    tmp = torch.zeros([n_frags_all], dtype=torch.bool)
    tmp[new_idx_to_original_idx] = True
    idx = torch.where(tmp == False)
    return [i.item()+1 for i in idx[0]]


def main():

    print(f"Start: {get_time_stamp()}")
    
    # =============================
    #         training data
    train_graphs, train_names, vali_graphs, vali_names, train_loader, vali_loader = read_data(
        [dir_save_train_data], 
        device=device,
        converage_required_flag=True,
        debug_flag=DEBUG)

    check_and_mkdir(path_save_predicted_results)

    curated_order_save_file = open(os.path.join(path_save_predicted_results, "curated_order.txt"), 'w')
    fig_save_dir = os.path.join(path_save_predicted_results, "likelihood_fig")
    check_and_mkdir(fig_save_dir)

    print("="*20)
    print(f"train data ...")
    for batch, name, loader in zip(train_graphs, train_names, train_loader):
        save_batch_into_file(
            batch, 
            name, 
            loader=loader, 
            train_data_flag=True, 
            plot_flag=True, 
            file=curated_order_save_file, 
            save_dir=fig_save_dir)

    print(f"vali data ...")
    for batch, name, loader in zip(vali_graphs, vali_names, vali_loader):
        save_batch_into_file(
            batch, 
            name, 
            loader=loader, 
            train_data_flag=False, 
            plot_flag=True, 
            file=curated_order_save_file, 
            save_dir=fig_save_dir)

    curated_order_save_file.close()
    print(f"Finished: {get_time_stamp()}")
    return 

if __name__ == "__main__":
    main()