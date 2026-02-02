import matplotlib.pyplot as plt
import numpy as np
import torch
import os
os.chdir('/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation')
import helper_funcs as hf
assert torch.cuda.is_available()
import sys
import argparse

# Script to train neural field for neural embedding

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural field embedding training")
    parser.add_argument("--nits", type=int, default=201, help="Number of iterations for training")
    parser.add_argument("--lr",type=float,default=1e-3,help="Learning rate")
    parser.add_argument("--depth",type=int,default=2,help="Depth of network")
    parser.add_argument("--width",type=int,default=200,help="Width of network")
    parser.add_argument("--use_pos",type=int,default=1)
    parser.add_argument("--encoding_std",type=int,default=20,help="Std in fourier feature postional encoding of network")
    parser.add_argument("--encoding_width",type=int,default=None,help="Width of positional encoding layer (default 2*width)")
    parser.add_argument("--plot_folder",type=str,default="new_shepp_logan_smoothed/nf_embed",help="Folder to save plots")
    parser.add_argument("--np_save",type=str,default="nf_embed_saves",help="Folder to save np recons")
    parser.add_argument("--phantom",type=str,default="shepp_logan")
    parser.add_argument("--activation",type=str,default="SIREN")
    
    args = parser.parse_args()

    # if not os.path.exists("/plots/"+args.plot_folder):
    #     os.makedirs("/plots/"+args.plot_folder)
    save_network = False
    if args.phantom == "shepp_logan":
        print(f"Loading phantom: phantoms/new_shepp_logan_smoothed.npy")
        phantom = np.load("phantoms/new_shepp_logan_smoothed.npy")
        f_gt = phantom # Ground truth
        f_norm = np.mean(f_gt**2)**(1/2)
        f_gt /= f_norm
        mb_values = np.array([0,0.1,0.2,0.3,0.4,1])/ f_norm 
    elif args.phantom == "mgo":
        print(f"Loading phantom: phantoms/full2560_2d_hann.npy")
        phantom = np.load("phantoms/full2560_2d_hann.npy")
        f_gt = phantom # Ground truth
        f_norm = np.mean(f_gt**2)**(1/2)
        f_gt /= f_norm
        mb_values = np.load("/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation/mgo_mb_values.npy")
        mb_values = mb_values / f_norm
    elif args.phantom == "mgo_real":
        print(f"Loading phantom: /dtu-compute/Mathcrete_thesis/mathcrete_dtu/real_data/fbp_ctf_4h_z1000.npy")
        phantom = np.load("/dtu-compute/Mathcrete_thesis/mathcrete_dtu/real_data/fbp_ctf_4h_z1000.npy")
        f_gt = phantom # Ground truth
        f_norm = np.mean(f_gt**2)**(1/2)
        f_gt /= f_norm
        mb_values = np.load("/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation/mgo_mb_values.npy")
        mb_values = mb_values / f_norm
        save_network = True
    elif args.phantom == "mgo_roi_full":
        print(f"Loading phantom: /dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation/roi_saves/cgls_mask_full.npy")
        phantom = np.load("/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation/roi_saves/cgls_mask_full.npy")
        f_gt = phantom # Ground truth
        f_norm = np.mean(f_gt**2)**(1/2)
        f_gt /= f_norm
        mb_values = np.load("/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation/mgo_mb_values.npy")
        mb_values = mb_values / f_norm
        save_network = True
    elif args.phantom == "mgo_roi_trunc":
        print(f"Loading phantom: /dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation/roi_saves/cgls_mask_roi.npy")
        phantom = np.load("/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation/roi_saves/cgls_mask_roi.npy")
        f_gt = phantom # Ground truth
        f_norm = np.mean(f_gt**2)**(1/2)
        f_gt /= f_norm
        mb_values = np.load("/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation/mgo_mb_values.npy")
        mb_values = mb_values / f_norm
        save_network = True
    elif args.phantom == "mgo_fbp":
        print(f"Loading phantom: mgo_recon_saves/fbp_rr.npy")
        phantom = np.load("mgo_recon_saves/fbp_rr.npy")
        f_gt = phantom # Ground truth
        f_norm = np.mean(f_gt**2)**(1/2)
        f_gt /= f_norm
        mb_values = np.load("/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation/mgo_mb_values.npy")
        mb_values = mb_values / f_norm
        save_network = True
    elif args.phantom[0] == "n": # fbp of sl phantom
        print(f"Loading phantom: fbp_saves/fbp_"+args.phantom+".npy")
        phantom = np.load("fbp_saves/fbp_"+args.phantom+".npy")
        f_gt = phantom # Ground truth
        f_norm = np.mean(f_gt**2)**(1/2)
        f_gt /= f_norm
        mb_values = mb_values = np.array([0,0.1,0.2,0.3,0.4,1])/ f_norm
        save_network = True

    print(f"Shape of full phantom {phantom.shape}")
    print(f"Saving to {args.np_save}")

    dev = torch.device("cuda:0")
    N = f_gt.shape[0]
    grid_mks = (2*np.arange(N) + 1)/(2*N) - 1/2
    c0, c1 = np.meshgrid(grid_mks, grid_mks)
    val_coords = c0**2 + c1**2 < 1 # circular grid
    f_gt *= val_coords
    f_gt_torch = torch.from_numpy(f_gt[val_coords]).to(dev)
    XY = torch.stack((torch.from_numpy(c0.flatten()),torch.from_numpy(c1.flatten())), axis = 1).float()
    XY = XY[val_coords.flatten(), :].to(dev)

    # network size
    width = args.width
    depth = args.depth
    print(f"Encdoing std: {args.encoding_std}")
    if args.use_pos:
        net = hf.real_nf_sim(width, depth,activation=args.activation,encoding_std=args.encoding_std).to(dev)
    else:
        net = hf.real_nf_sim_nopos(width, depth,activation=args.activation).to(dev)

    ###################################
    # Training
    opt = torch.optim.Adam(net.parameters(), lr = args.lr)
    nreport = -1 # plot intermediate results
    nprint = 5 # print intermediate results
    nb = 100
    nits = args.nits
    print(f"Running {nits} iterations")
    binds = np.arange(nb) 
    inds = np.arange(XY.shape[0])
    f_np = np.zeros((N,N))
    LOSS = []
    for it in range(nits):
        np.random.shuffle(binds)
        np.random.shuffle(inds)
        rrmse = 0
        for j, b in enumerate(binds):
            opt.zero_grad()
            f = net(XY[inds[b::nb], :]).flatten()
            loss = torch.mean( (f- f_gt_torch[inds[b::nb]])**2)
            loss.backward()
            opt.step()
            rrmse += loss.item()

        rrmse = (rrmse/nb)**(1/2)
        LOSS.append(rrmse)
        if (it%nprint == 0) and nprint >0 :
            print(f"Epoch {it}, RRMSE: {rrmse}")

        if it%nreport == 0 and nreport > 0:
            with torch.no_grad():
                g = 0*f_np[val_coords]
                for b in range(nb):
                    g[b::nb] =  net(XY[b::nb,:]).flatten().cpu().detach().numpy()
                f_np[val_coords] = g

            plt.close(), plt.clf()
            fig, a = plt.subplots(1,3)
            a[0].imshow(f_gt, vmin = f_gt.min(), vmax = f_gt.max())
            a[0].title.set_text("Ground truth")
            a[1].imshow(f_np, vmin = f_gt.min(), vmax = f_gt.max())
            a[1].title.set_text("Neural field rep.")
            a2 = a[2].imshow(f_np - f_gt)
            a[2].title.set_text("Difference")
            plt.colorbar(a2)
            plt.show()
            #plt.savefig('Figures/embedding.png', dpi = 1000)
            del fig
            del a
    #         torch.save(net.state_dict(), "Nets/embedding")
    # torch.save(net.state_dict(), "Nets/embedding")

    plt.plot(range(it+1),LOSS)
    plt.title("Training loss curve")
    plt.xlabel("Epochs")
    plt.ylabel("RRMSE")          
    plt.show()
    plt.savefig("plots/"+args.plot_folder+"/losscurve.png")

    ################################

    with torch.no_grad():
        g = 0*f_np[val_coords]
        for b in range(nb):
            g[b::nb] =  net(XY[b::nb,:]).flatten().cpu().detach().numpy()
        f_np[val_coords] = g
    # plt.close(), plt.clf()
    # fig, a = plt.subplots(1,2)
    # fig.set_size_inches(10,20)
    # a[0].imshow(f_gt, vmin = f_gt.min(), vmax = f_gt.max(),cmap="gray")
    # a[0].title.set_text("Ground truth")
    # a[1].imshow(f_np, vmin = f_gt.min(), vmax = f_gt.max(),cmap="gray")
    # a[1].title.set_text("Neural field rep.")
    # plt.show()
    hf.plot_compare([f_gt,f_np,f_np-f_gt],title=[f"Ground Truth","Neural field embedding","Difference"], \
                    savefig=True,savepath="plots/"+args.plot_folder+"/global_compare.png")

    #from cil.utilities.display import show2D
    hf.plot_compare([(f_gt)[1130:1430,1130:1430],f_np[1130:1430,1130:1430],(f_np - f_gt)[1130:1430,1130:1430]], \
                    title=[f"Ground Truth","Neural field embedding","Difference"] , \
                    savefig=True,savepath="plots/"+args.plot_folder+"/central_compare.png")
    # dark sport
    hf.plot_compare([(f_gt)[1200:1500,1700:2000],f_np[1200:1500,1700:2000],(f_np - f_gt)[1200:1500,1700:2000]], \
                    title=[f"Ground Truth","Neural field embedding","Difference"],
                    savefig=True,savepath="plots/"+args.plot_folder+"/particle_compare.png")
    # boundary
    hf.plot_compare([(f_gt)[1000:1500,500:1000],f_np[1000:1500,500:1000],(f_np - f_gt)[1000:1500,500:1000]], \
                    title=[f"Ground Truth","Neural field embedding","Difference"], \
                    savefig=True,savepath="plots/"+args.plot_folder+"/boundary_compare.png")

    print(f"Metrics Nf embedding)")
    hf.compute_metrics(f_gt,f_np,mb_values=mb_values)
    np.save(args.np_save,f_np)
    if save_network:
        if args.phantom[0] == "n":
            torch.save(net.state_dict(), "neural_field_saves/fbp_"+args.phantom+"/net_fbp")
        elif args.phantom == "mgo_real" or args.phantom == "mgo_roi_full" or args.phantom == "mgo_roi_trunc" :
            torch.save(net.state_dict(), "neural_field_saves/fbp_"+args.phantom+"/net_fbp")
            print("Saved to " + f"neural_field_saves/fbp_"+args.phantom+"/net_fbp")
        else:
            torch.save(net.state_dict(), "neural_field_saves/mgo/fbp_embedding") 