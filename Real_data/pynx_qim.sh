# LSBATCH: User input
#!/bin/bash
#BSUB -J pynx_qim_justctf
#BSUB -q gpul40s
#BSUB -W 12:00
#BSUB -n 16
#BSUB -R "rusage[mem=12GB]"
#BSUB -gpu "num=1:mode=exclusive_process:mps=yes"
#BSUB -R "span[hosts=1]"
#BSUB -B
#BSUB -N
#BSUB -o batch_out//pynx_qim/pynx_qim_justctf%J.out
#BSUB -e batch_out/pynx_qim/pynx_qim_justctf%J.err

# Job for computing PyNX phase retreival on raw data

PATH_ROOT_DATA="/zhome/95/1/155570/QIM/projects/2025_QIM_Mathcrete/raw_data_extern"
TIME_FRAME="4h"
DATA_NAME="MgO_insitu_water_35nm_bottom_011"

# Initialize Python environment for pynx and nabu
source ~/miniforge3/bin/activate
conda activate pynx_env

pynx-holotomo-id16b --data $PATH_ROOT_DATA"/"$DATA_NAME"_1_" --sino_filter none \
 --delta_beta 150 --nxtomo --algorithm CTF --nz 4 --padding 200 --ngpu 1 --binning 1 \
 --prefix_output $PATH_ROOT_DATA"/phasemaps/4h/just_ctf" --normalise 

pynx-holotomo-id16b --data $PATH_ROOT_DATA"/"$DATA_NAME"_1_" --sino_filter none \
 --delta_beta 150 --nxtomo --algorithm ctf-long --nz 4 --padding 200 --ngpu 1 --binning 1 \
 --prefix_output $PATH_ROOT_DATA"/phasemaps/4h/long_ctf" --save_fbp_vol tiff 

