import matplotlib.pyplot as plt
import numpy as np
import torch
import os
os.chdir('/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation')
import helper_funcs as hf
#assert torch.cuda.is_available()
import sys
import argparse

# Script for comparing sigma_B values in positional encoding for NF embeddings

if __name__ == "__main__":
    plt.rcParams.update({'font.size': 11})
    parser = argparse.ArgumentParser(description="Encoding parameter sweep for for NF embeds")
    parser.add_argument("--folder", type=str, default="plots/new_shepp_logan_smoothed/nf_embed", help="Folder to save plot results")
    parser.add_argument("--np_folder", type=str, default="embed_saves/new_shepp_logan_smoothed", help="Folder to save/load recons")
    parser.add_argument("--phantom",type=str,default="shepp_logan")
    args = parser.parse_args()
    
    if args.phantom == "shepp_logan":
        print(f"Loading phantom: phantoms/new_shepp_logan_smoothed.npy")
        phantom = np.load("phantoms/new_shepp_logan_smoothed.npy")
        f_gt = phantom # Ground truth
        f_norm = np.mean(f_gt**2)**(1/2)
        f_gt /= f_norm
        mb_values = np.array([0,0.1,0.2,0.3,0.4,1])/ f_norm 
        print("Loading nf recons with various pos encs")
        sigma_bs = [1,10,20]
        small = np.load(args.np_folder+"/d2_w200_pos1.npy")
        medium = np.load(args.np_folder+"/d2_w200_pos10.npy")
        large = np.load(args.np_folder+"/d2_w200_pos20.npy")
        zooms = [1130,1430,1130,1430]
    elif args.phantom == "mgo":
        print(f"Loading phantom: phantoms/full2560_2d_hann.npy")
        phantom = np.load("phantoms/full2560_2d_hann.npy")
        f_gt = phantom # Ground truth
        f_norm = np.mean(f_gt**2)**(1/2)
        #f_gt /= f_norm
        mb_values = np.array([0,0.25/2,(0.25+0.4)/2,(0.56+0.4)/2,(0.56+0.68)/2,1]) / f_norm
        sigma_bs = [5,20,40]
        small = np.load(args.np_folder+"/pos_5enc/d3_w300_enc5.npy") * f_norm
        medium = np.load(args.np_folder+"/pos_20enc/d3_w300_enc20.npy") * f_norm
        large = np.load(args.np_folder+"/pos_40enc/d3_w300_enc40.npy") * f_norm
        zooms = [1200,1500,1700,2000] # particle 
    print(f"Shape of full phantom {phantom.shape}")
    
    recs_z = []
    recs = []
    recs.append(f_gt)
    recs_z.append(f_gt[zooms[0]:zooms[1],zooms[2]:zooms[3]])
    titles = []
    titles.append("Ground truth")

    recs.append(small)
    recs_z.append(small[zooms[0]:zooms[1],zooms[2]:zooms[3]])
    titles.append(r'$\sigma_B=$'+f"{sigma_bs[0]}")

    recs.append(medium)
    recs_z.append(medium[zooms[0]:zooms[1],zooms[2]:zooms[3]])
    titles.append(r'$\sigma_B=$'+f"{sigma_bs[1]}")

    recs.append(large)
    recs_z.append(large[zooms[0]:zooms[1],zooms[2]:zooms[3]])
    titles.append(r'$\sigma_B=$'+f"{sigma_bs[2]}")
    hf.compute_metrics(f_gt, recs[1:],get_values=True,print_out=True,list=True,mb_values=mb_values)
    hf.plot_compare(recs,title=titles,savefig=True,
                    savepath=args.folder+"/nf_enc_compare.png")
    hf.plot_compare(recs_z,title=titles,savefig=True,
                    savepath=args.folder+"/nf_zoom_enc_compare.png")