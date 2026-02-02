# LSBATCH: User input
#!/bin/bash
#BSUB -J nf_real
#BSUB -q gpua100
#BSUB -W 16:00
#BSUB -n 4
#BSUB -R "rusage[mem=5GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -B
#BSUB -N
#BSUB -o batch_out/nf_real/nf_real%J.out
#BSUB -e batch_out/nf_real/nf_real%J.err

source ~/miniforge3/bin/activate
conda activate mc_env

# Job for running NF reconstruction

python -u /dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation/nf_sim_embedding.py --nits 180 --depth 3 --width 300 \
    --plot_folder "mgo_real/nf_embed/" \
    --phantom "mgo_real" \
     --np_save "/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation/embed_saves/mgo_real/nf_embed.npy" \
     --activation "SIREN" --lr 0.002 --phantom "mgo_real" \
     --use_pos 1 --encoding_std 30

python -u /dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation/nf_real_reconstruct.py  \
    --folder "mgo_real/nf_recon" \
    --path_meas "/dtu-compute/Mathcrete_thesis/MgO_hydrate/CHR_4h_ctf_nofilter/MgO_insitu_water_35nm_bottom_011_pm_slice_1000.npy" \
    --load_base True --base_path "neural_field_saves/fbp_mgo_real/net_fbp" --skip_ss True \
    --path_gt "/dtu-compute/Mathcrete_thesis/mathcrete_dtu/real_data/fbp_ctf_4h_z1000.npy" \
    --prox_nits 100 --depth 3 --width 300 --tv True --tv_reg 1 --lr_prox 0.0001 --lr_sub 0.0001 --nb_prox 10