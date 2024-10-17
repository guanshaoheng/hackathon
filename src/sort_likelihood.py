import torch
from config import *
from restore_training_data import CompressedHicFragDataSet
from config_plot import *
from typing import List
import os, sys
import argparse
from collections import deque

"""
    Draw the order of the fragments according to the likelihood matrix.
"""

"""
export PYTHONPATH=src:$PYTHONPATH
/nfs/users/nfs_s/sg35/.conda/envs/autohic_new/bin/python src/sort_likelihood.py -o -d results/sorted_hic_madeup/sorted_results.txt
"""

def sort_fragments_according_likelihood_or_distance_matrix(
        likelihood_mx:torch.tensor, nfrags:int, frag_lens:torch.Tensor, threshold:float=0.005, name:str=None,
        echo_flag:bool=False,
        mx_type:str="likelihood",
        )->Tuple[List[List[int]], List[List[bool]]]:
    """
        split the likelihood matrix by chromosomes

        Args:
            likelihood_mx: the likelihood matrix, in shape of [nfrags, nfrags, 4]
            nfrags: the number of fragments
            frag_lens: the length of the fragments
            threshold: the threshold for the likelihood
            name: the name of the data set
        Returns:
            chromosomes: the list of the fragments included in each chromosome
            inversed_flag: the flag of the orientation of the fragments, same shape with chromosomes
    """
    likelihood_mx = likelihood_mx.clone()
    if mx_type == "distance":
        likelihood_mx = -likelihood_mx  # NOTE: after this in-place operation, the likelihood_mx is changed in the out scope    
        threshold = -0.8
    elif mx_type == "likelihood":
        pass
    else:
        raise ValueError(f"The mx_type {mx_type} is not in [distance, likelihood]")

    frag_lens = frag_lens[:nfrags]
    if likelihood_mx.shape == torch.Size([nfrags, nfrags, 4]):
        pass
    elif likelihood_mx.shape == torch.Size([2 * nfrags, 2 * nfrags]) or likelihood_mx.shape == torch.Size([2 * MAX_NUM_FRAG, 2 * MAX_NUM_FRAG]):
        likelihood_mx = likelihood_mx[:2*nfrags, :2*nfrags].reshape([nfrags, 2, nfrags, 2]).transpose(1, 2).reshape([nfrags, nfrags, 4])
    else:
        raise ValueError(f"The shape of the likelihood matrix is {likelihood_mx.shape}, not equal to {nfrags, nfrags, 4} and not equal to {2 * nfrags, 2 * nfrags}, please organize the shape before input the likelihood matrix.")

    # make sure the value is not all 0 in the left-bottom corner
    # Finished: make sure the two indices are mirrored
    lower_triangle_indices = torch.tril_indices(nfrags, nfrags, offset=-1)
    likelihood_mx[lower_triangle_indices[0], lower_triangle_indices[1]] = likelihood_mx[lower_triangle_indices[1], lower_triangle_indices[0]][:, [0, 2, 1, 3]]

    # FINISHED sort the fragments acoording to fragment size, before sorting the fragments according to likelihood
    order_lens = torch.argsort(frag_lens, descending=True)
    likelihood_mx_new = likelihood_mx[order_lens][:, order_lens]

    # TODO PENDING adjust the threshold according to the global likelihood 
    # calculating the threshold according to the global likelihood maybe not a good idea
    # likelihood_values, likelihood_idx = torch.sort(likelihood_mx_new.flatten(), descending=True)
    """
    # plot and check the likelihood values
    while i < len(likelihood_values) and likelihood_values[i] > 0:
        i += 1
    plt.plot(likelihood_values[: i ].numpy())
    plt.text(i, likelihood_values[i-1].item(), f"{likelihood_values[i].item():.4e}")
    plt.savefig(f"test_likelihood_values_{name}.png")
    plt.close("all")
    return 0 
    """

    not_visited = torch.ones(nfrags, dtype=torch.bool)
    chromosomes = []
    chromosomes_inversed = []
    for frag in range(nfrags): # iterate over all the fragments
        if not_visited.sum() == 0: break
        if not not_visited[frag]: continue
        chromosome = deque()
        chromosome_inversed = deque()
        dfs(frag, False, not_visited, likelihood_mx_new, chromosome, chromosome_inversed, threshold, head_flag=False)  # depth first search, find the fragments with the highest likelihood and link them
        chromosomes.append(chromosome)
        chromosomes_inversed.append(chromosome_inversed)

    chromosomes = [list(chromosome) for chromosome in chromosomes]
    chromosomes_inversed = [list(chromosome_inversed) for chromosome_inversed in chromosomes_inversed]

    # resume the order into the original order, as it was ordered according to the fragment size
    chromosomes = [[order_lens[i].item() for i in chromosome] for chromosome in chromosomes]

    # sort the chromosomes according to their length
    chromosome_lens = torch.tensor([sum([frag_lens[i] for i in chromosome]) for chromosome in chromosomes])
    chromosome_order_lens = torch.argsort(chromosome_lens, descending=True)
    chromosomes = [chromosomes[i.item()] for i in chromosome_order_lens]
    chromosomes_inversed = [chromosomes_inversed[i.item()] for i in chromosome_order_lens]

    if echo_flag:
        print("Fragment order Likelihood:")
        for i in range(len(chromosomes)):
            len_chromosome = sum([frag_lens[i] for i in chromosomes[i]])
            line = f"Chromosome [{i+1}] len({len_chromosome}) num_frags({len(chromosomes[i])}): "
            for j in range(len(chromosomes[i])):
                inversed = chromosomes_inversed[i][j]
                idx = chromosomes[i][j]
                line += (('-' if inversed else '')+ str(idx+1) + ' ')
            print(line)
            if i>=10: break
    return [[j+1 for j in i] for i in chromosomes], chromosomes_inversed

def dfs(
        frag:int, 
        inversed_flag: bool, 
        not_visited:torch.tensor, 
        likelihood_mx:torch.tensor, 
        chromosome:deque[int], 
        chromosome_inversed:deque[bool], 
        threshold:float, 
        head_flag)->None:
    """
        depth first search for the chromosome

        Args:
            - frag: the current fragment
            - not_visited: the flag of the visited fragments
            - likelihood_mx: the likelihood matrix
            - chromosome: the current chromosome
            - threshold: the threshold of the likelihood
            - head_flag: false (default) -> link to the end, true -> link to the head
    """
    
    if head_flag: # if the head is the starting point
        chromosome.appendleft(frag) # add the fragment to the chromosome
        chromosome_inversed.appendleft(inversed_flag)
    else:
        chromosome.append(frag)# add the fragment to the chromosome
        chromosome_inversed.append(inversed_flag)
    not_visited[frag] = False

    if not_visited.sum() == 0: return

    # check the frag with its most likely neighbor
    head_is_inversed = chromosome_inversed[0]
    end_is_inversed = chromosome_inversed[-1]

    max_val, max_idx = torch.max(
            torch.cat(
            [
                likelihood_mx[chromosome[0], not_visited][:, [0, 1] if not head_is_inversed else [2, 3]],
                likelihood_mx[chromosome[-1], not_visited][:, [2, 3] if not end_is_inversed else [0, 1]],
            ],
            dim=-1
        ),
        dim = 0,
    )

    max_of_4 = torch.argmax(max_val)

    if max_val[max_of_4] < threshold: return # end of dfs: if the likelihood is lower than the threshold, return

    if max_of_4 == 0: # head to head
        head_flag = True
        inversed_flag = True
        if max_val[0] == max_val[1]: # if the likelihood is different
            inversed_flag = False
    elif max_of_4 == 1: # head to end
        head_flag = True
        inversed_flag = False
    elif max_of_4 == 2: # end to head
        head_flag = False
        inversed_flag = False
    elif max_of_4 == 3: # end to end
        head_flag = False
        inversed_flag = True
        if max_val[2] == max_val[3]:
            inversed_flag = False
    else:
        raise ValueError(f"max_argsort {max_of_4.item()} is not in [0, 1, 2, 3]")
    dfs(torch.nonzero(not_visited)[max_idx[max_of_4]].item(), inversed_flag, not_visited, likelihood_mx, chromosome, chromosome_inversed, threshold, head_flag)
    

def get_num_frags_scaff(scaffid:torch.tensor)->List[int]:
    """
        return the list of number of frags in each scaffold
    """
    num_frags_scaff = []
    last_scaffid, i = -1, 0
    tmp_num_frags = 0
    while last_scaffid!=0 and i < len(scaffid):
        if last_scaffid != scaffid[i]:
            if tmp_num_frags != 0:
                num_frags_scaff.append(tmp_num_frags)
            tmp_num_frags = 1
            last_scaffid = scaffid[i]
        else:
            tmp_num_frags += 1
        i += 1
    return num_frags_scaff


def plot_hic(hic_mx:torch.tensor, nfrags:int, name:str)->None:
    for i in range(4):
        plt.imshow(hic_mx[i, :nfrags, :nfrags], cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f"test_{name}_hic_{i}.png")
        plt.close("all")


def form_line(chromosomes:List[List[int]], inversed_flag:List[List[bool]])->Tuple[str, str]:
    num_chr = len(chromosomes)
    line_frag_id = "; ".join(", ".join([str(j) for j in chromosomes[i]]) for i in range(num_chr))
    line_frag_inversed = "; ".join(", ".join([str(int(j)) for j in inversed_flag[i]]) for i in range(num_chr))
    return line_frag_id, line_frag_inversed


def output_into_file(
        name:str,
        chromosomes_original:List[List[int]],
        inversed_flag_original:List[List[bool]],
        chromosomes_ai:List[List[int]],
        inversed_flag_ai:List[List[bool]],
        file=sys.stdout,
        train_data_flag:bool=False,
):
    """
        output the results into the file
    """
    line = "#" + "="*20 + "\n" +f"Name: {name}" + (" (train)" if train_data_flag else "") + "\n"
    line_frag_id, line_frag_inversed = form_line(chromosomes_original, inversed_flag_original)
    line += "Curated fragments id: " + line_frag_id + "\n" +\
            "Inversed flag: " + line_frag_inversed  + "\n"
    
    line_frag_id, line_frag_inversed = form_line(chromosomes_ai, inversed_flag_ai)
    line += "AI-curated fragments id: " + line_frag_id + "\n" +\
            "AI-curated inversed flag: " + line_frag_inversed  + "\n"
    line += "\n" * 2 
    print(line, file=file)
    return


def get_curated_fragments_order(
        scaffid:torch.Tensor,
        frag_id:torch.Tensor,  
        frag_lens:torch.Tensor,
        name:str,
        echo_flag:bool=False,
        max_num_frag = 1024,
):
    '''
        # NOTE the idx of scaff and frag all start from 1
    '''
    num_frags_scaff = get_num_frags_scaff(scaffid)
        
    start = 0
    chrmomoses_original = []
    for num, i in enumerate(num_frags_scaff):  # iterate over all the scaffolds
        chrmomoses_original.append([j.item() for j in frag_id[start:start + i]])
        start += i
    start = 0
    scaff_lens = torch.tensor(
        [sum([frag_lens[i-1 if i <= max_num_frag else i - max_num_frag-1] for i in chromosome]) for chromosome in chrmomoses_original]
        )
    argsort_lens = torch.argsort(scaff_lens, descending=True)
    num_frags_scaff = [num_frags_scaff[i.item()] for i in argsort_lens]
    chrmomoses_original = [chrmomoses_original[i.item()] for i in argsort_lens]
    
    if echo_flag:
        print("\n" + "=" * 20 + f"\nName: {name}")
        print("Frags order:")
        for num, i in enumerate(num_frags_scaff):
            print(f"Scaffold [{num+1}] ({i}) len({scaff_lens[num]}): " +  ', '.join([str(j) if j <= max_num_frag else '-'+str(j - max_num_frag) for j in chrmomoses_original[num]]))
            start += i
            if num>=10: break
            
    inversed_flag_original=[[(i>max_num_frag) for i in tmp] for tmp in chrmomoses_original]
    chromosomes_original=[[i if i<=max_num_frag else i-max_num_frag for i in tmp] for tmp in chrmomoses_original]
    return chromosomes_original, inversed_flag_original


def main():
    saved_train_data_path = "dataset/train_data_madeup"
    
    parser = argparse.ArgumentParser(description='Sort the fragments according to the likelihood matrix.')
    parser.add_argument("-o", '--output', type=str, default=None, help='The output file path.')
    parser.add_argument("-d", '--debug', action='store_true', help='The debug flag.')
    parser.add_argument("-p", '--path', type=str, default=saved_train_data_path, help='The path of the saved training data.')
    args = parser.parse_args()
    if args.output is not None and os.path.exists(os.path.split(args.output)[0]):
        print(f"Update Arg: Output file: {args.output}")
        output_stream = open(args.output, 'w')
    else:
        output_stream = sys.stdout
    global debug_flag
    if args.debug:
        print(f"Update Arg: Debug mode {args.debug}")
        debug_flag = args.debug
    else:
        debug_flag = False
    if args.path is not None and os.path.exists(args.path):
        print(f"Update Arg: Path of the saved training data: {args.path}")
        saved_train_data_path = args.path
    

    dataset = CompressedHicFragDataSet([saved_train_data_path], debug_flag=debug_flag, madeup_flag=True)
    # data_iter = iter(dataset)
    for i in range(len(dataset)):
        hic_mx, frag_id, scaffid, name, n_frags, frag_lens, distance_mx, likelihood_mx, frag_coverage, frag_start_loc_relative, frag_masss_center_relative, frag_average_density_global, frag_average_hic_interaction, _, _, _, _  = dataset[i]

        chromosomes_original, inversed_flag_original = get_curated_fragments_order(
            scaffid=scaffid, frag_id=frag_id, frag_lens=frag_lens, name=name, echo_flag=debug_flag
        )

        chromosomes_sorted_from_likelihood, inversed_flag_sorted_from_likelihood = sort_fragments_according_likelihood_or_distance_matrix(likelihood_mx, nfrags=n_frags, frag_lens=frag_lens, name=name, echo_flag=True)

        output_into_file(
            name=name,
            chromosomes_original=chromosomes_original,
            inversed_flag_original=inversed_flag_original,
            chromosomes_ai=chromosomes_sorted_from_likelihood,
            inversed_flag_ai=inversed_flag_sorted_from_likelihood,
            file=output_stream,
        )
    output_stream.close()
    return 


if __name__ == "__main__":
    main()