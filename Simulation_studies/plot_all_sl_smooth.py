import matplotlib.pyplot as plt
import numpy as np

# CIL methods
from cil.plugins.astra import ProjectionOperator
from cil.optimisation.functions import OperatorCompositionFunction
from cil.optimisation.functions import SmoothMixedL21Norm
from cil.optimisation.operators import GradientOperator
from cil.optimisation.algorithms import FISTA
from cil.optimisation.functions import LeastSquares
from cil.optimisation.functions import OperatorCompositionFunction
from cil.optimisation.utilities import callbacks
from cil.optimisation.operators import GradientOperator
from cil.optimisation.utilities import callbacks
from cil.framework import ImageData
from cil.plugins.ccpi_regularisation.functions import FGP_TV

import os
os.chdir('/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation')
import helper_funcs as hf
import argparse

# Script to compare reconstruction methods for the half-smoothed Shepp-Logan phantom.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model based recon for shepp logan")
    parser.add_argument("--folder", type=str, default="plots/new_shepp_logan_smoothed", help="Folder to save plot results")
    parser.add_argument("--np_folder_justmb", type=str, default="justmb_saves", help="Folder to save plot results")
    parser.add_argument("--np_folder_tv", type=str, default="tv_saves", help="Folder to save plot results")
    parser.add_argument("--np_folder_mb", type=str, default="mb_saves", help="Folder to save plot results")
    parser.add_argument("--np_folder_nf", type=str, default="neural_field_saves/new_shepp_logan_smoothed", help="Folder to save plot results")
    parser.add_argument("--load_old",type=bool,default=True,help="To load old recons")

    args = parser.parse_args()

    gt = np.load("phantoms/new_shepp_logan_smoothed.npy")
    gt_norm = np.mean(gt**2)**(1/2)
    gt /= gt_norm # for easier comparison with nf
    mb_values=np.array([0,0.1,0.2,0.3,0.4,1]) / gt_norm   

    noise_levels = [0.001,0.01,0.05,0.1]
    num_angles = [2560,1280,320,40]

    params = [[1,1280],[2,1280],[3,1280],[4,1280],[2,2560],[2,1280],[2,320],[2,40]] #[noise_level, no_angles]
    tv_alphas = [5e-5,0.002,0.005,0.01,0.0045,0.002,0.002,8e-05] # MDP
    mb_alphas = [7e-05,2e-05,0.0001,0.0005,5e-05,2e-05,2e-05,5e-06] #MDP
    justmb_alphas = [t*100 for t in mb_alphas]
    justmb_alphas[1] = 3e-05
    justmb_alphas[4] = 4e-05
    justmb_alphas[5] = 3e-05
    tv_alphas_smooth = [t/100 for t in mb_alphas]  
    epsilon = 1e-6 # for smoothed TV
    optimize_tv = False 
    optimize_mb = False
    opt_number = 5
    opt_step = 1.5

    noise_rrmse = []
    noise_ssim = []
    noise_mb = []
    angle_rrmse = []
    angle_ssim = []
    angle_mb = []

    zooms = [1130,1430] # zoom in region
    recs_all = []
    recs_z_all = []
    recs_z_angles = []
    for i, param in enumerate(params):
        print("##########################################")
        print(f"Global iteration: {i}, noise: {noise_levels[int(param[0]-1)]}, No. angles: {param[1]}")
        print("##########################################")
        recs = []
        recs_z = []
        recs.append(gt)
        recs_z.append(gt[zooms[0]:zooms[1],zooms[0]:zooms[1]])
        sino = np.load(f"Simulated_sinogram/new_shepp_logan_smoothed/sino_nl_{str(param[0])}_na_{str(param[1])}.npy")
        print(f"Loaded Simulated_sinogram/new_shepp_logan_smoothed/sino_nl_{str(param[0])}_na_{str(param[1])}.npy")
        sino /= gt_norm
        ag, ig, data = hf.CIL_setup_cement(sino,pixel_size=2/gt.shape[0],angle_range=180)
        PO = ProjectionOperator(ig,ag,"gpu")
        LS = LeastSquares(A=PO, b=data)
        delta = np.linalg.norm(PO.direct(ig.allocate(gt)).array-sino) # for morozov
        print(f"delta = {delta}")
        recfbp = (hf.CIL_FBP(data,ig,padding=False)).array
        recs.append(hf.circle_mask(recfbp))
        recs_all.append(recfbp)
        recs_z.append(recfbp[zooms[0]:zooms[1],zooms[0]:zooms[1]])
        recs_z_all.append(recfbp[zooms[0]:zooms[1],zooms[0]:zooms[1]])

        ### TV ####
        print(f"TV: alpha =  {tv_alphas[i]}")
        
        alpha_tv = tv_alphas[i]
        best_alpha_tv = alpha_tv
        try:
            if args.load_old:
                TV_reco = np.load(args.np_folder_tv+f"/sl_smooth_nl_{str(param[0])}_na_{str(param[1])}_alpha_{alpha_tv}.npy")
                print("Loaded TV recon")
        except:
            if optimize_tv:
                tv_rrmse_best = 1
                print(f"optimizing with {opt_number} values")
                for i in range(opt_number):
                    print(f"Alpha_tv = {alpha_tv}")
                    x0 = ig.allocate(0.0)
                    cb1 = callbacks.ProgressCallback() # This is the progress bar 
                    cb2 = callbacks.EarlyStoppingObjectiveValue(threshold=1e-5)
                    # Grad = GradientOperator(ig)
                    # TV = alpha_tv * OperatorCompositionFunction(SmoothMixedL21Norm(epsilon), Grad)
                    TV = FGP_TV(alpha=alpha_tv, nonnegativity=False, device='gpu')
                    fista_TV = FISTA(initial=x0, f=LS, g=TV, update_objective_interval=2)
                    fista_TV.run(400,callbacks=[cb1,cb2]) 
                    TV_reco = (fista_TV.solution)
                    TV_reco = hf.circle_mask(TV_reco.array)
                    tv_rrmse, _ = hf.compute_metrics(gt,TV_reco,get_values=True,print_out=True,list=False)
                    if tv_rrmse < tv_rrmse_best:
                        tv_best_recon = TV_reco
                        tv_rrmse_best = tv_rrmse
                        best_alpha_tv = alpha_tv
                    alpha_tv = alpha_tv*opt_step
                print(f"best alpha, and error {[best_alpha_tv,tv_rrmse_best]}")
                TV_reco = tv_best_recon
            else:
                print(f"Single iteration with alpha_tv = {alpha_tv}")
                x0 = ig.allocate(0.0)
                cb1 = callbacks.ProgressCallback() # This is the progress bar 
                cb2 = callbacks.EarlyStoppingObjectiveValue(threshold=1e-5)
                # Grad = GradientOperator(ig)
                # TV = alpha_tv * OperatorCompositionFunction(SmoothMixedL21Norm(epsilon), Grad)
                TV = FGP_TV(alpha=alpha_tv, nonnegativity=False, device='gpu')
                fista_TV = FISTA(initial=x0, f=LS, g=TV, update_objective_interval=2)
                fista_TV.run(400,callbacks=[cb1,cb2]) 
                TV_reco = (fista_TV.solution)
                TV_reco = hf.circle_mask(TV_reco.array)
                tv_rrmse, _ = hf.compute_metrics(gt,TV_reco,get_values=True,print_out=True,list=False)
            
            np.save(args.np_folder_tv+f"/sl_smooth_nl_{str(param[0])}_na_{str(param[1])}_alpha_{best_alpha_tv}.npy",TV_reco)
            print("Saved TV recon")

        TV_data = ImageData(TV_reco,geometry=ig)
        mres = np.linalg.norm((PO.direct(TV_data)).array-sino)
        print(f"TV mres = {mres} (delta is {delta})")
        recs.append(TV_reco)
        recs_all.append(TV_reco)
        recs_z.append(TV_reco[zooms[0]:zooms[1],zooms[0]:zooms[1]])
        recs_z_all.append(TV_reco[zooms[0]:zooms[1],zooms[0]:zooms[1]])

        ## MB ##
        alpha_mb = justmb_alphas[i]
        try:
            if args.load_old:
                MB_reco = np.load(args.np_folder_justmb+f"/sl_smooth_nl_{str(param[0])}_na_{str(param[1])}_alphas_{alpha_mb}.npy")
                print(f"MB with alpha_mb {alpha_mb}")
                print("Loaded MB recon")
        except:
                #x0_mbtv = ImageData(recfbp,geometry=ig)
                print(f"MB with alpha_mb {alpha_mb}")
                MB = hf.Multibang(u=mb_values)
                x0_mb = ig.allocate(0.0)
                Grad = GradientOperator(ig)
                cb1 = callbacks.ProgressCallback() # This is the progress bar 
                G = alpha_mb*MB
                fista_MB = FISTA(f=LS,g=G,initial=x0_mb,update_objective_interval=1,max_iteration=400)
                fista_MB.run(300,callbacks=[cb1])
                MB_reco = (fista_MB.solution).array
                mb_rrmse, _ = hf.compute_metrics(gt,MB_reco,get_values=True,print_out=True,list=False)
                np.save(args.np_folder_justmb+f"/sl_smooth_nl_{str(param[0])}_na_{str(param[1])}_alphas_{alpha_mb}",MB_reco)
                print("Saved MB recon")
        MB_data = ImageData(MB_reco,geometry=ig)
        mres = np.linalg.norm((PO.direct(MB_data)).array-sino)
        print(f"MB mres = {mres} (delta is {delta})")
            
        recs.append(MB_reco)
        recs_all.append(MB_reco)
        recs_z.append(MB_reco[zooms[0]:zooms[1],zooms[0]:zooms[1]])
        recs_z_all.append(MB_reco[zooms[0]:zooms[1],zooms[0]:zooms[1]])

        ### MB + TV ###
        alpha_mb = mb_alphas[i]
        alpha_tv_smooth = tv_alphas_smooth[i]
        try:
            if args.load_old:
                MBTV_reco = np.load(args.np_folder_mb+f"/sl_smooth_nl_{str(param[0])}_na_{str(param[1])}_alphas_{alpha_mb}.npy")
                print(f"MB + TV with alpha_tv {alpha_tv_smooth}, alpha_mb {alpha_mb}")
                print("Loaded MB recon")
                if i > 0:
                    MBTV_reco /= gt_norm
        except:
            if optimize_mb:
                mbtv_rrmse_best = 1
                print(f"optimizing with {opt_number} values")
                for i in range(opt_number):
                    print(f"MB + TV with alpha_tv {alpha_tv_smooth}, alpha_mb {alpha_mb}")
                    #x0_mbtv = ImageData(recfbp,geometry=ig)
                    MB = hf.Multibang(u=mb_values)
                    x0_mbtv = ig.allocate(0.0)
                    Grad = GradientOperator(ig)
                    cb1 = callbacks.ProgressCallback() # This is the progress bar 
                    G = alpha_mb*MB
                    F = LS + alpha_tv_smooth * OperatorCompositionFunction(SmoothMixedL21Norm(epsilon), Grad)
                    fista_MBTV = FISTA(f=F,g=G,initial=x0_mbtv,update_objective_interval=1,max_iteration=400)
                    fista_MBTV.run(300,callbacks=[cb1])
                    MBTV_reco = (fista_MBTV.solution).array
                    mbtv_rrmse, _ = hf.compute_metrics(gt,MBTV_reco,get_values=True,print_out=True,list=False)
                    if mbtv_rrmse < mbtv_rrmse_best:
                        mbtv_best_recon = MBTV_reco
                        mbtv_rrmse_best = mbtv_rrmse
                        best_alpha_tvsmooth = alpha_tv_smooth
                    alpha_tv_smooth = alpha_tv_smooth*opt_step
                MBTV_reco = mbtv_best_recon
                print(f"best alpha (tv smooth), and error {[best_alpha_tvsmooth,mbtv_rrmse_best]}")
            else:
                print(f"Single iteration with with alpha_tv_smooth {alpha_tv_smooth}, alpha_mb {alpha_mb}")
                x0_mbtv = ImageData(recfbp,geometry=ig)
                MB = hf.Multibang(u=mb_values)
                #x0_mb = ig.allocate(fbp_data)
                Grad = GradientOperator(ig)
                cb1 = callbacks.ProgressCallback() # This is the progress bar 
                G = alpha_mb*MB
                F = LS + alpha_tv_smooth * OperatorCompositionFunction(SmoothMixedL21Norm(epsilon), Grad)
                fista_MBTV = FISTA(f=F,g=G,initial=x0_mbtv,update_objective_interval=1,max_iteration=400)
                fista_MBTV.run(300,callbacks=[cb1])
                MBTV_reco = (fista_MBTV.solution).array
                mbtv_rrmse, _ = hf.compute_metrics(gt,MBTV_reco,get_values=True,print_out=True,list=False)
                np.save(args.np_folder_mb+f"/sl_smooth_nl_{str(param[0])}_na_{str(param[1])}_alphas_{alpha_mb}",MBTV_reco)
                print("Saved MB + TV recon")
        
        MBTV_data = ImageData(MBTV_reco,geometry=ig)
        mres = np.linalg.norm((PO.direct(MBTV_data)).array-sino)
        print(f"MBTV mres = {mres} (delta is {delta})")

        recs.append(MBTV_reco)
        recs_all.append(MBTV_reco)
        recs_z.append(MBTV_reco[zooms[0]:zooms[1],zooms[0]:zooms[1]])
        recs_z_all.append(MBTV_reco[zooms[0]:zooms[1],zooms[0]:zooms[1]])

        # NF + TV
        try: 
            NF_reco = np.load(args.np_folder_nf+f"/nl_{str(param[0])}_na_{str(param[1])}/nf_recon.npy")#*np.mean(gt**2)**(1/2)
            print("Loaded NF recon")
        except:
            import torch
            print(f"Missing nf recon")
            print("Reconstructing from network")
            assert torch.cuda.is_available()
            dev = torch.device("cuda:0")
            net = hf.real_nf_sim(200, 2,activation="SIREN").to(dev)
            state_dict = torch.load(args.np_folder_nf+f"/nl_{str(param[0])}_na_{str(param[1])}/net_0")
            net.load_state_dict(state_dict)
            N = gt.shape[0]
            grid_mks = (2*np.arange(N) + 1)/(2*N) - 1/2
            c0, c1 = np.meshgrid(grid_mks, grid_mks)
            XY = torch.stack((torch.from_numpy(c0.flatten()),torch.from_numpy(c1.flatten())), axis = 1).float()
            F_np = np.zeros(N**2)

            nb = 10
            with torch.no_grad():
                for b in range(nb):
                    F_np[b::nb] = net(XY[b::nb,:].to(dev)).cpu().detach().numpy().flatten()
                NF_reco = hf.circle_mask(F_np.reshape((N,N)))*np.mean(gt**2)**(1/2)
            np.save(args.np_folder_nf+f"/nl_{str(param[0])}_na_{str(param[1])}/nf_recon.npy",NF_reco)
            print("NF reco saved")
        recs.append(NF_reco)
        recs_all.append(NF_reco)
        recs_z.append(NF_reco[zooms[0]:zooms[1],zooms[0]:zooms[1]])
        recs_z_all.append(NF_reco[zooms[0]:zooms[1],zooms[0]:zooms[1]])
        
        RRMSE, SSIM, MB_SCORE = hf.compute_metrics(gt,recs[1:],get_values=True,print_out=True,list=True,mb_values=mb_values)

    print("Creating metrics plots")
    plt.rcParams.update({'font.size': 16})
    plt.close()
    plt.clf()
    plt.semilogy(noise_levels,np.squeeze(np.array(noise_rrmse))[:,0],'--o',label="FBP")
    plt.semilogy(noise_levels,np.squeeze(np.array(noise_rrmse))[:,1],'--o',label="TV")
    plt.semilogy(noise_levels,np.squeeze(np.array(noise_rrmse))[:,2],'--o',label="MB")
    plt.semilogy(noise_levels,np.squeeze(np.array(noise_rrmse))[:,3],'--o',label="MB+TV")
    plt.semilogy(noise_levels,np.squeeze(np.array(noise_rrmse))[:,4],'--o',label="NF+TV")
    plt.legend(fontsize="x-small")
    plt.xlabel("Noise level")
    plt.ylabel("RRMSE")
    plt.savefig(args.folder+f"/compare_noise/rrmse_noise.png", bbox_inches='tight',dpi = 1000)
    
    plt.close()
    plt.clf()
    plt.plot(noise_levels,np.squeeze(np.array(noise_ssim))[:,0],'--o',label="FBP")
    plt.plot(noise_levels,np.squeeze(np.array(noise_ssim))[:,1],'--o',label="TV")
    plt.plot(noise_levels,np.squeeze(np.array(noise_ssim))[:,2],'--o',label="MB")
    plt.plot(noise_levels,np.squeeze(np.array(noise_ssim))[:,3],'--o',label="MB+TV")
    plt.plot(noise_levels,np.squeeze(np.array(noise_ssim))[:,4],'--o',label="NF+TV")
    plt.legend(fontsize="x-small")
    plt.xlabel("Noise level")
    plt.ylabel("SSIM")
    plt.savefig(args.folder+f"/compare_noise/ssim_noise.png", bbox_inches='tight',dpi = 1000)
    
    plt.close()
    plt.clf()
    plt.plot(noise_levels,np.squeeze(np.array(noise_mb))[:,0],'--o',label="FBP")
    plt.plot(noise_levels,np.squeeze(np.array(noise_mb))[:,1],'--o',label="TV")
    plt.plot(noise_levels,np.squeeze(np.array(noise_mb))[:,2],'--o',label="MB")
    plt.plot(noise_levels,np.squeeze(np.array(noise_mb))[:,3],'--o',label="MB+TV")
    plt.plot(noise_levels,np.squeeze(np.array(noise_mb))[:,4],'--o',label="NF+TV")
    plt.legend(fontsize="x-small")
    plt.xlabel("Noise level")
    plt.ylabel("SS")
    plt.savefig(args.folder+f"/compare_noise/mbs_noise.png", bbox_inches='tight',dpi = 1000)

    ############### angles plot ###################
   
    plt.close()
    plt.clf()
    plt.semilogy(num_angles,np.squeeze(np.array(angle_rrmse))[:,0],'--o',label="FBP")
    plt.semilogy(num_angles,np.squeeze(np.array(angle_rrmse))[:,1],'--o',label="TV")
    plt.semilogy(num_angles,np.squeeze(np.array(angle_rrmse))[:,2],'--o',label="MB")
    plt.semilogy(num_angles,np.squeeze(np.array(angle_rrmse))[:,3],'--o',label="MB+TV")
    nnf = np.squeeze(np.array(angle_rrmse))[:,4]
    plt.semilogy(num_angles,nnf,'--o',label="NF+TV")
    plt.legend(fontsize="x-small")
    plt.xlabel("No. of angles")
    plt.ylabel("RRMSE")
    plt.savefig(args.folder+f"/compare_angles/rrmse_angle.png", bbox_inches='tight',dpi = 1000)
    
    plt.close()
    plt.clf()
    plt.plot(num_angles,np.squeeze(np.array(angle_ssim))[:,0],'--o',label="FBP")
    plt.plot(num_angles,np.squeeze(np.array(angle_ssim))[:,1],'--o',label="TV")
    plt.plot(num_angles,np.squeeze(np.array(angle_ssim))[:,2],'--o',label="MB")
    plt.plot(num_angles,np.squeeze(np.array(angle_ssim))[:,3],'--o',label="MB+TV")
    plt.plot(num_angles,np.squeeze(np.array(angle_ssim))[:,4],'--o',label="NF+TV")
    plt.legend(fontsize="x-small")
    plt.xlabel("No. of angles")
    plt.ylabel("SSIM")
    plt.savefig(args.folder+f"/compare_angles/ssim_angle.png", bbox_inches='tight',dpi = 1000)
    
    plt.close()
    plt.clf()
    plt.plot(num_angles,np.squeeze(np.array(angle_mb))[:,0],'--o',label="FBP")
    plt.plot(num_angles,np.squeeze(np.array(angle_mb))[:,1],'--o',label="TV")
    plt.plot(num_angles,np.squeeze(np.array(angle_mb))[:,2],'--o',label="MB")
    plt.plot(num_angles,np.squeeze(np.array(angle_mb))[:,3],'--o',label="MB+TV")
    plt.plot(num_angles,np.squeeze(np.array(angle_mb))[:,4],'--o',label="NF+TV")
    plt.legend(fontsize="x-small")
    plt.xlabel("No. of angles")
    plt.ylabel("SS")
    plt.savefig(args.folder+f"/compare_angles/mbs_angle.png", bbox_inches='tight',dpi = 1000)

    plt.rcParams.update({'font.size': 18})
    print(f"Length of all recs {len(recs_z_all)}")
    all_noise_z = [*recs_z_all[0:20:5],*recs_z_all[1:20:5],*recs_z_all[2:20:5],*recs_z_all[3:20:5],*recs_z_all[4:20:5]]
    all_angle_z = [*recs_z_all[20::5],*recs_z_all[21::5],*recs_z_all[22::5],*recs_z_all[23::5],*recs_z_all[24::5]]
    all_noise = [*recs_all[0:20:5],*recs_all[1:20:5],*recs_all[2:20:5],*recs_all[3:20:5],*recs_all[4:20:5]]
    all_angle = [*recs_all[20::5],*recs_all[21::5],*recs_all[22::5],*recs_all[23::5],*recs_all[24::5]]
    hf.plot_compare_big(all_noise_z
        ,title=["0.1% Noise","1% Noise","5% Noise","10% Noise"],
        row_titles=["FBP","TV","MB","MB+TV","NF+TV"],
        rows=5,
        savefig=True,
        savepath=args.folder+"/compare_noise/zoom_noisy_all.png")

    hf.plot_compare_big(all_angle_z,
        title=["2560 angles","1280 angles","320 angles","40 angles"],
        row_titles=["FBP","TV","MB","MB+TV","NF+TV"],
        rows=5,
        savefig=True, use_range=4,
        savepath=args.folder+"/compare_angles/zoom_angles_all.png")
    
    hf.plot_compare([recs_all[20::5][-1],recs_all[21::5][-1],recs_all[23::5][-1],recs_all[24::5][-1]],
                    title=["FBP","TV","MB+TV","NF+TV"], use_range=-1,
                    savefig=True,savepath=args.folder+"/compare_angles/global_final_angle.png")
    # Line plot
    detector = 2500
    x1 = [1280,1280]
    y1 = [0,1280]
    # plt.plot(x1,y1,color="blue",linewidth=3)
    plt.rcParams.update({'font.size': 15})
    plt.close()
    plt.clf()
    plt.plot(range(2560)[1050:1500],(gt)[1050:1500,1280],label="GT")
    #plt.plot(range(2560)[1050:1500],(recs_all[20::5][-1])[1050:1500,1280],label="FBP",linestyle="--")
    plt.plot(range(2560)[1050:1500],(recs_all[24::5][-1])[1050:1500,1280],label="NF+TV")
    plt.plot(range(2560)[1050:1500],(recs_all[23::5][-1])[1050:1500,1280],label="MB+TV",linestyle="--")
    #plt.plot(range(2560)[1050:1500],(recs_all[22::5][-1])[1050:1500,1280],label="MB",linestyle="--")
    plt.plot(range(2560)[1050:1500],(recs_all[21::5][-1])[1050:1500,1280],label="TV",linestyle="--")
    
   
    plt.xlabel("Vertical axis")
    plt.ylabel("Pixel value")
    plt.legend(loc=3)
    plt.show()
    plt.savefig(args.folder+"/compare_angles/lineplot_final_angle.png",bbox_inches='tight',dpi = 1000)

    plt.close()
    plt.clf()
    plt.plot(range(2560)[1050:1500],(gt)[1050:1500,1280],label="GT")
    #plt.plot(range(2560)[1050:1500],(recs_all[0:20:5][-1])[1050:1500,1280],label="FBP",linestyle="--")
    #plt.plot(range(2560)[1050:1500],(recs_all[2:20:5][-1])[1050:1500,1280],label="MB",linestyle="--")
    plt.plot(range(2560)[1050:1500],(recs_all[3:20:5][-1])[1050:1500,1280],label="MB+TV",linestyle="--")
    plt.plot(range(2560)[1050:1500],(recs_all[1:20:5][-1])[1050:1500,1280],label="TV",linestyle="--")
    plt.plot(range(2560)[1050:1500],(recs_all[4:20:5][-1])[1050:1500,1280],label="NF+TV")
    plt.ylim((0.6,1.8))
    plt.xlabel("Vertical axis")
    plt.ylabel("Pixel value")
    plt.legend(loc=3)
    plt.show()
    plt.savefig(args.folder+"/compare_noise/lineplot_final_noise.png",bbox_inches='tight',dpi = 1000)
    # plot_compare_big(recs_z_all[16::4]+recs_z_all[17::4]+recs_z_all[18::4]+recs_z_all[19::4],
    #     title=["2560 angles","1280 angles","320 angles","40 angles"]
    #     ,savefig=True,
    #     savepath=args.folder+"/compare_angles/zoom_angles_all.png")

    



    
