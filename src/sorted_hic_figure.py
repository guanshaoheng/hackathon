import os
from multiprocessing import Pool, cpu_count
import numpy as np 
# from python.sortHiC.config import * 
from config import *
from utils import check_and_mkdir, DEBUG
import matplotlib.pyplot as plt 


"""
export PYTHONPATH=$PYTHONPATH:src
nohup /nfs/users/nfs_s/sg35/.conda/envs/autohic_new/bin/python -u src/sorted_hic_figure.py -c 0 -m > results/predicted_likelihood_graph_madeup/sorted_hic_figure_madeup.log 2>&1 &

"""

"""
    Draw the sorted hic matrix according to the predicted results
"""
saved_human_hic = DIR_SAVE_FULL_HICMX_HUMAN
predicted_results_file_pth = os.path.join(PATH_SAVE_PREDICTED_LIKELIHOOD_GRAPH, "curated_order.txt")
curated_hic_figrue_save_dir = os.path.join(PATH_SAVE_PREDICTED_LIKELIHOOD_GRAPH, "curated_hic_fig")
check_and_mkdir(curated_hic_figrue_save_dir)


def read_line(line:str)->List[int]:
    ptr = 0
    while line[ptr] != ':':
        ptr += 1
    if ptr > len(line): raise RuntimeError(f"Error: ptr({ptr}) exceeds the boundary({len(line)})!")
    ptr += 1
    target_fragid = []
    target_scaffid = []
    
    chromosomes = line[ptr:].strip().split(';')
    for i, chrom in enumerate(chromosomes):
        chrom = chrom.strip()
        tmp = chrom.split(',')
        target_fragid += [int(j.strip()) for j in tmp]
        target_scaffid += [i+1] * len(tmp)

    return target_fragid, target_scaffid


def restore_predict_results(predicted_results_file_pth:str, finished_names:set[str])->dict:

    with open(predicted_results_file_pth, 'r') as f:
        lines = f.readlines()
    
    results = {}
    line_ptr = 0
    while line_ptr < len(lines):
        if not lines[line_ptr].startswith("#"):
            line_ptr += 1
            continue
        line_ptr += 1  # name line 
        tmp = lines[line_ptr].strip().split(' ')
        name = tmp[1]
        if name in finished_names:  # skip the finished samples
            continue
        if len(tmp) < 3:
            train_flag = False
        else:
            train_flag = ("(train)" == tmp[2])
        line_ptr += 1 # target order line
        target_fragid, target_scaffid = read_line(lines[line_ptr])
        line_ptr += 1 # target inversed flag
        target_inversed, _ = read_line(lines[line_ptr])
        line_ptr += 1 # ai-curated order line
        target_fragid_ai, target_scaffid_ai = read_line(lines[line_ptr])
        line_ptr += 1 # ai-curated inversed flag
        target_inversed_ai, _ = read_line(lines[line_ptr])

        results[name] = [
            target_fragid, target_scaffid, target_inversed, 
            target_fragid_ai, target_scaffid_ai, target_inversed_ai, 
            train_flag]

        line_ptr += 1 # new line
        
    return results
        

def rebuild_hic(
        hic_mx:np.ndarray, 
        splitloc_0:np.ndarray, 
        splitloc_1:np.ndarray, 
        sorted_order:List[int], # make sure the sorted_order starts from 1
        )->np.ndarray:
    check_if_sorted_order_start_from_1(sorted_order)
    len_hic_mx = len(hic_mx)
    hic_mx_reordered = np.copy(hic_mx)
    num_sorted = len(sorted_order)
    h_ptr = w_ptr = 0
    for h_num in range(num_sorted):
        h_inverse = False
        i = sorted_order[h_num] - 1
        if i >= MAX_NUM_FRAG:
            h_inverse=True
            i -= MAX_NUM_FRAG
        w_ptr = h_ptr  # NOTE  do not let those two with same address, cuz while using torch.tensor, all of the tensor will be changed to torch tensor. torch tensor will have the same address if using `=`
        h = min(splitloc_1[i] - splitloc_0[i], len_hic_mx - h_ptr)  # make sure not exceeds the boundary of hic_mx
        hic_mx_reordered[h_ptr:h_ptr+h, h_ptr:h_ptr+h] = inverse_process(
            hic_mx[splitloc_0[i]:splitloc_0[i]+h, splitloc_0[i]:splitloc_0[i]+h],
            h_flag=h_inverse, 
            w_flag=h_inverse)
        w_ptr = w_ptr + h
        for w_num in range(h_num+1, num_sorted):  # solved: problem when w_num reaches the last one, indices exceed the boundary.
            w_inverse = False
            j = sorted_order[w_num] - 1
            if j >= MAX_NUM_FRAG:
                w_inverse = True
                j -= MAX_NUM_FRAG
            w = min(splitloc_1[j] - splitloc_0[j], len_hic_mx - w_ptr)  # make sure not exceeds the boundary of hic_mx
            hic_mx_reordered[h_ptr:h_ptr+h, w_ptr:w_ptr+w] = inverse_process(
                hic_mx[splitloc_0[i]:splitloc_0[i]+h, splitloc_0[j]:splitloc_0[j]+w],
                h_flag=h_inverse,
                w_flag=w_inverse)
            w_ptr += w
        h_ptr += h
    rows, cols = np.tril_indices(hic_mx_reordered.shape[0], -1)

    # Copy the upper triangle values to the lower triangle
    hic_mx_reordered[rows, cols] = hic_mx_reordered[cols, rows]
    return hic_mx_reordered


def plot_hic_mx(name:str, hic_mx:np.ndarray, plot_type:str="original", train_flag:bool=False)->None:
    plot_save_figure(os.path.join(curated_hic_figrue_save_dir, f"{'train' if train_flag else 'vali'}.{name}.{plot_type}.png"), hic_mx)


def plot_save_figure(fname:str, mx:np.ndarray, dpi=300)->None:
    plt.imshow(mx)
    plt.tight_layout()
    plt.savefig(fname, dpi=dpi)
    plt.close("all")


def inverse_process(mx:np.ndarray, h_flag=False, w_flag=False, )->np.ndarray:
    if h_flag:
        mx= mx[::-1, :] # traverse block
    if w_flag:
        mx = mx[:, ::-1] # vertical block
    return mx


def main_worker(
        args:Tuple[str, List[int], List[int], List[bool], List[int], List[int], List[bool], bool])->None:
    name, target_fragid, target_scaffid, target_inversed, \
        target_fragid_ai, target_scaffid_ai, target_inversed_ai,\
        train_flag  = args
    if "chm13" in name:
        insert_idx = name.find("_numCuts")
        name = name[:insert_idx] + "_madeUp" + name[insert_idx:]
        npz_fname = os.path.join(saved_human_hic, f"{name}.orignal.HiC.npy.npz")
    else:
        raise ValueError(f"Error: {name} is not a valid name!")
    if not os.path.exists(npz_fname):
        raise RuntimeError(f"Error: {npz_fname} not exists!")
    data = np.load(npz_fname)
    # Access the arrays
    #TODO check why the split locs are all 0
    orignal_hic = data['orignal_hic']
    split_start_locs = data['split_start_locs']
    split_end_locs = data['split_end_locs']

    print(f"{name} starting...")
    plot_hic_mx(name, orignal_hic, plot_type="original", train_flag=train_flag)
    new_hic = rebuild_hic(orignal_hic, split_start_locs, split_end_locs, [target_fragid[i] + target_inversed[i] * MAX_NUM_FRAG for i in range(len(target_fragid))])
    plot_hic_mx(name, new_hic, plot_type="target", train_flag=train_flag)
    del new_hic
    new_hic = rebuild_hic(orignal_hic, split_start_locs, split_end_locs, [target_fragid_ai[i] + target_inversed_ai[i] * MAX_NUM_FRAG for i in range(len(target_inversed_ai))])
    plot_hic_mx(name, new_hic, plot_type="sorted", train_flag=train_flag)
    print(f"{name} is finished.")
    plt.close("all")
    del new_hic, orignal_hic, data
    return


def main():
    results = restore_predict_results(predicted_results_file_pth, filter_the_finished_samples()) 
    names = list(results.keys())
    if DEBUG:
        name = names[0]
        main_worker([name, 
                     results[name][0], 
                     results[name][1], 
                     results[name][2], 
                     results[name][3], 
                     results[name][4], 
                     results[name][5], 
                     results[name][6]
                     ])
    else:
        num_cpu = 16
        print(f"Num_cpu: {cpu_count()}, num_sampels: {len(names)}, NUM_CPU: {NUM_CPU}, thus num process: {num_cpu}")
        with Pool(num_cpu) as pool:
            pool.map(main_worker, 
                     zip(names, 
                         [results[i][0] for i in names],  
                         [results[i][1] for i in names], 
                         [results[i][2] for i in names],
                         [results[i][3] for i in names],  
                         [results[i][4] for i in names], 
                         [results[i][5] for i in names],
                         [results[i][6] for i in names]
                         )
                         )
    return 0


def check_if_sorted_order_start_from_1(index:np.array):
    sorted_index = np.sort(index) 
    if sorted_index[0] == 0:
        print(f"Warning: The index does not start from 1, please check!")
    return 


def filter_the_finished_samples()->set[str]:
    type_dic = {
        "original": 0,
        "sorted": 1,
        "target": 2
    }
    exist_set = {}
    for fname in os.listdir(curated_hic_figrue_save_dir):
        line = fname.split('.')
        name = line[1]
        if name not in exist_set:
            exist_set[name] = set()
        exist_set[name].add(type_dic[line[2]])
    finished_names = set()
    for key in exist_set:
        if len(exist_set[key]) == 3:
            finished_names.add(key)
    del exist_set
    return finished_names


if __name__ == "__main__":
    
    main()
    
