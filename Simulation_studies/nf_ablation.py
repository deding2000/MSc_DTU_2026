import matplotlib.pyplot as plt
import numpy as np
import torch
import os
os.chdir('/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation')
import helper_funcs as hf
#assert torch.cuda.is_available()
import sys
import argparse

# Script for comparing NF architectures

if __name__ == "__main__":
    plt.rcParams.update({'font.size': 11})
    parser = argparse.ArgumentParser(description="Ablation study for for NF recons")
    parser.add_argument("--folder", type=str, default="plots/new_shepp_logan_smoothed/nf_embed", help="Folder to save plot results")
    parser.add_argument("--np_folder", type=str, default="embed_saves/new_shepp_logan_smoothed", help="Folder to load recons")
    parser.add_argument("--phantom",type=str,default="shepp_logan")
    args = parser.parse_args()
    if args.phantom == "shepp_logan":
        print(f"Loading phantom: phantoms/new_shepp_logan_smoothed.npy")
        phantom = np.load("phantoms/new_shepp_logan_smoothed.npy")
        f_gt = phantom # Ground truth
        f_norm = np.mean(f_gt**2)**(1/2)
        f_gt /= f_norm
        mb_values = np.array([0,0.1,0.2,0.3,0.4,1])/ f_norm 
        print("Loading nf recons")
        simple = np.load(args.np_folder+"/d2_w200_simple.npy")
        siren = np.load(args.np_folder+"/d2_w200_siren.npy")
        pos_enc = np.load(args.np_folder+"/d2_w200_pos10.npy")
    elif args.phantom == "mgo":
        print(f"Loading phantom: phantoms/full2560_2d_hann.npy")
        phantom = np.load("phantoms/full2560_2d_hann.npy")
        f_gt = phantom # Ground truth
        f_norm = np.mean(f_gt**2)**(1/2)
        f_gt /= f_norm
        mb_values = np.array([0,0.25/2,(0.25+0.4)/2,(0.56+0.4)/2,(0.56+0.68)/2,1]) / f_norm
        simple = np.load(args.np_folder+"/no_pos/d3_w300.npy")
        siren = np.load(args.np_folder+"/siren/d3_w300.npy")
        pos_enc = np.load(args.np_folder+"/pos_20enc/d3_w300_enc20.npy")
         
    print(f"Shape of full phantom {phantom.shape}")
    zooms = [1130,1430]
    recs_z = []
    recs = []
    recs.append(f_gt)
    recs_z.append(f_gt[zooms[0]:zooms[1],zooms[0]:zooms[1]])
    titles = []
    titles.append("Ground truth")



    recs.append(simple)
    recs_z.append(simple[zooms[0]:zooms[1],zooms[0]:zooms[1]])
    titles.append("Simple network")

    recs.append(siren)
    recs_z.append(siren[zooms[0]:zooms[1],zooms[0]:zooms[1]])
    titles.append("With SIREN activation")

    recs.append(pos_enc)
    recs_z.append(pos_enc[zooms[0]:zooms[1],zooms[0]:zooms[1]])
    titles.append("With pos. encoding")
    hf.compute_metrics(f_gt, recs[1:],get_values=True,print_out=True,list=True,mb_values=mb_values)
    hf.plot_compare(recs,title=titles,savefig=True,
                    savepath=args.folder+"/nf_compare.png")
    hf.plot_compare(recs_z,title=titles,savefig=True,
                    savepath=args.folder+"/nf_zoom_compare.png")