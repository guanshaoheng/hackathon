import torch 
import re
import os
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader
from config import *   


"""
    Restore the training samples from the saved .pt file.
"""

class CompressedHicFragDataSet(Dataset):
    def __init__(self, dir_lists:List[str], coverage_required_flag=False, madeup_flag=False, debug_flag=False) -> None:
        self.names, self.nFrags, self.maxFrags, self.fpaths = [], [], [], []
        self.hicMx = []
        self.fragID = []
        self.coverage_flag = []
        self.repeat_density_flag = []
        self.scaffID = []
        self.frag_lens = []
        self.distance_mx = []
        self.likelihood_mx = []
        self.frag_coverage = []
        self.frag_repeats_density = []
        self.frag_start_loc_relative = []
        self.frag_masss_center_relative = []
        self.frag_average_density_global = []
        self.frag_average_hic_interaction = []
        self.total_len_in_pixels = []
        self.coverage_required_flag = coverage_required_flag
        self.madeup_flag = madeup_flag
        self.debug_flag = debug_flag
        self.general_train_file_pattern = r"([\w-]+)_CompHicOuputID_nFrags(\d+)_maxFrags(\d+)_coverage(\d+)_repeat(\d+).pt"
        self.madeup_train_file_pattern = r"([\w-]+)_CompHicOuputID_nFrags(\d+)_maxFrags(\d+)_coverage(\d+)_repeat(\d+)_numCuts(\d+)_repeatNum(\d+).pt" # aPelFus1_1_CompHicOuputID_nFrags101_maxFrags1024_coverage1_repeat1.pt
        # chm13_1_normal_CompHicOuputID_nFrags65_maxFrags1024_coverage1_repeat1_numCuts3_repeatNum22.pt
        
        # initial the data sets
        self.__initial_data_sets(dir_lists)

    def __initial_data_sets(self, dir_lists:List[str])->None:
        for dirPath in dir_lists:
            for fname in (os.listdir(dirPath) if not self.debug_flag else os.listdir(dirPath)[:20]): # (os.listdir(dirPath) if not DEBUG else os.listdir(dirPath)[:9]):
                if self.madeup_flag:
                    matched = re.match(self.madeup_train_file_pattern, fname)
                else:
                    matched = re.match(self.general_train_file_pattern, fname)
                if not matched: continue
                if self.madeup_flag:
                    name = f"{matched[1]}_numCuts{matched[6]}_numRepeat{matched[7]}"
                else:
                    name = matched[1]
                n_frags = int(matched[2])
                max_num_frags = int(matched[3])
                coverage_found = bool(int(matched[4]))
                repeat_density_found = bool(int(matched[5]))
                # if name in NAMES_SHORT_WINDOW: continue # skip the samples with short window size D
                if self.coverage_required_flag and not coverage_found: continue
                self.names.append(name)
                self.nFrags.append(n_frags)
                self.maxFrags.append(max_num_frags)
                self.coverage_flag.append(coverage_found)
                self.repeat_density_flag.append(repeat_density_found) 
                fpth = os.path.join(dirPath, fname)
                self.fpaths.append(fpth)
                '''
                training data saved as:

                1. compressed hic mx             [4, n_frags, n_frags]
                2. frags order                   [n_frags]
                3. frags inversed flag           [n_frags]
                4. frags scaffid                 [n_frags]
                5. frags length in pixels        [n_frags]
                6. frags start location in pixels[n_frags]
                7. frags mass center in pixels   [n_frags, 2]
                8. frags average interaction density to whole genome   [n_frags, num_divided_parts]
                9. frags averaged hic interaction[n_frags, n_frags]
                10. frags distance matrix in bp   [2 * n_frags, 2 * n_frags]
                11. frags contact likelihood      [2 * n_frags, 2 * n_frags]
                12. frags averaged coverage       [n_frags]   
                '''
                hicmx, \
                    fragID, \
                    inverseFlag, \
                    scaffid, \
                    frag_lens, \
                    frag_start_loc_relative, \
                    mass_center_relative, \
                    fragments_average_density_global, \
                    averaged_hic_interaction, \
                    distance_mx, \
                    likelihood_mx, \
                    *optional = torch.load(fpth)
                coverage = self.__pad(optional[0]) if (coverage_found and optional) else torch.ones(MAX_NUM_FRAG) * PAD_TOKEN
                if coverage.sum() == 0:  # make sure the coverage is valid, or turn the coverage_flag off
                    self.coverage_flag[-1] = False
                    coverage_found = False
                repeat_density = self.__pad(optional[1]) if (repeat_density_found and optional) else torch.ones(MAX_NUM_FRAG) * PAD_TOKEN
                if repeat_density.sum() == 0:
                    self.repeat_density_flag[-1] = False
                    repeat_density_found = False
                self.hicMx.append(hicmx)
                self.fragID.append(self.__pad(fragID.long() + inverseFlag.long() * MAX_NUM_FRAG))
                self.scaffID.append(self.__pad(scaffid))
                self.frag_lens.append(self.__pad(frag_lens))
                if len(optional) ==3 :
                    total_len_in_pixels = optional[2]
                    self.total_len_in_pixels.append(total_len_in_pixels)
                else:
                    raise ValueError(f"Error: the total_len_in_pixels is not found in the training data {fname}")

                self.frag_start_loc_relative.append(self.__pad(frag_start_loc_relative))  # [n_frags]
                self.frag_masss_center_relative.append(self.__pad(mass_center_relative, pad_1d_flag=True))  # [n_frags, 2]

                # NOTE make sure the fragments_average_density_global and averaged_hic_interaction are normalized to [0, 1]
                self.frag_average_density_global.append(self.__pad(fragments_average_density_global/255., pad_1d_flag=True)) # [n_frags, num_divided_parts]
                self.frag_average_hic_interaction.append(self.__pad(averaged_hic_interaction/255., head_tail_flag=False)) # [n_frags, n_frags]

                self.distance_mx.append(self.__pad(distance_mx, head_tail_flag=True))
                self.likelihood_mx.append(self.__pad(likelihood_mx, head_tail_flag=True))
                self.frag_coverage.append(coverage)
                self.frag_repeats_density.append(repeat_density)
                
    def __len__(self)->int:
      return len(self.hicMx)

    def __getitem__(self, index)->Tuple:
      return self.hicMx[index], self.fragID[index], self.scaffID[index], \
            self.names[index], self.nFrags[index], self.frag_lens[index], \
            self.distance_mx[index], self.likelihood_mx[index], self.frag_coverage[index], \
            self.frag_start_loc_relative[index], self.frag_masss_center_relative[index], \
            self.frag_average_density_global[index], self.frag_average_hic_interaction[index], self.coverage_flag[index], \
            self.repeat_density_flag[index], self.frag_repeats_density[index], self.total_len_in_pixels[index]

    def __pad(self, t:torch.Tensor, head_tail_flag=False, mode="constant", val=PAD_TOKEN, pad_1d_flag=False)->torch.Tensor:
        ndim = t.ndim
        if ndim ==1:
            return torch.nn.functional.pad(t, (0, MAX_NUM_FRAG - t.size(0)), mode=mode, value=val)
        elif ndim == 2:
            if pad_1d_flag:  # pad 2d tensor but in the first dimension
                return torch.nn.functional.pad(t, (0, 0, 0, MAX_NUM_FRAG - t.size(0)), mode=mode, value=val)
            else:
                return torch.nn.functional.pad(
                    t, 
                    (0, (MAX_NUM_FRAG * 2 if head_tail_flag else MAX_NUM_FRAG) - t.size(0), 
                    0, (MAX_NUM_FRAG * 2 if head_tail_flag else MAX_NUM_FRAG) - t.size(1)), 
                    mode=mode, value=val)
        else:
            raise ValueError(f"Invalid tensor shape {t.shape}")
        
    def __repr__(self):
        return f"CompressedHicFragDataSet: {len(self.hicMx)} samples."
        
        
if __name__ == "__main__":
    dataset = CompressedHicFragDataSet(DIR_SAVE_TRAIN_DATA_HUMAN, coverage_required_flag=True, madeup_flag=True, debug_flag=True)

    hicMx, fragID, scaffID, \
        names, nFrags, frag_lens, \
        distance_mx, likelihood_mx, frag_coverage, \
        frag_start_loc_relative, frag_masss_center_relative, \
        frag_average_density_global, frag_average_hic_interaction, coverage_flag, \
        repeat_density_flag, frag_repeats_density, total_len_in_pixel = next(iter(dataset))
    print()
    



    
