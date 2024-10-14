#!/bin/bash

work_place="/lustre/scratch125/ids/team117-assembly/hackathon"  # NOTE: no soft link should be in the path, or it can't be recognized
src_dir="${work_place}/src"
python_bin="/nfs/users/nfs_s/sg35/.conda/envs/autohic_new/bin/python"

export PYTHONPATH="${work_place}/src:$PYTHONPATH"
nohup ${python_bin} -u ${src_dir}/graphNN_hic_likelihood.py -n 10000 -s 20 > train_log.txt 2>&1 &

echo "Job submitted as $!"