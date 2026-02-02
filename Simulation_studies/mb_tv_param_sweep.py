import matplotlib.pyplot as plt
import numpy as np

# CIL methods
from cil.framework import AcquisitionGeometry, AcquisitionData, BlockDataContainer, ImageGeometry
from cil.utilities.display import show2D, show_geometry
from cil.processors import CentreOfRotationCorrector, Slicer, TransmissionAbsorptionConverter, Normaliser, Padder, RingRemover
from cil.io import TIFFStackReader
from cil.plugins.astra import ProjectionOperator, FBP
from cil.recon import FBP
from cil.optimisation.functions import OperatorCompositionFunction, Function
from cil.optimisation.functions import L2NormSquared, L1Norm, BlockFunction, MixedL21Norm, IndicatorBox, TotalVariation
from cil.optimisation.operators import GradientOperator, BlockOperator, IdentityOperator
from cil.optimisation.algorithms import PDHG, SIRT, CGLS
from cil.optimisation.algorithms import FISTA, ISTA
from cil.optimisation.functions import Function, LeastSquares, ZeroFunction
from cil.optimisation.functions import OperatorCompositionFunction
from cil.optimisation.utilities import callbacks
from cil.optimisation.algorithms import PDHG, GD
from cil.optimisation.functions import BlockFunction, MixedL21Norm, L2NormSquared, SmoothMixedL21Norm
from cil.optimisation.operators import BlockOperator, GradientOperator
from cil.framework import ImageData

from cil.plugins.ccpi_regularisation.functions import FGP_TV

import os
os.chdir('/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation')
import helper_funcs as hf
import argparse
import time

# Script for parameter sweep for MB, TV and MB + TV

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="reg. parameter sweep for for Multi-Bang regularizer")
    parser.add_argument("--folder", type=str, default="plots/new_shepp_logan_smoothed/mb_tv_sweep2", help="Folder to save plot results")
    parser.add_argument("--np_folder", type=str, default="mb_tv_sweep_saves6", help="Folder to save recons")

    args = parser.parse_args()

    gt = np.load("phantoms/new_shepp_logan_smoothed.npy")
    noise_level = 2
    num_angles = 320
    mb_values=np.array([0,0.1,0.2,0.3,0.4,1]) 
    alphas_mb = [1e-8,1e-7,1e-6,5e-6,8e-6,1e-5,2e-5,5e-5,8e-5,1e-4,5e-4,1e-3,1.5e-3,1.75e-3,2e-3,3e-3,5e-3,1e-2,0.1]
    tv_alphas_smooth = [t/100 for t in alphas_mb]
    alphas_tv = [1e-7,1e-6,5e-6,8e-6,1e-5,5e-5,1e-4,2e-4,5e-4,8e-4,9e-4,1e-3,2e-3,5e-3,1e-2,1e-1,1e-2,1e-1,1]
    mdp_tv = 10
    mdp_mbtv = 7
    mdp_mb = 13
    epsilon = 1e-6 # for smoothed TV

    skip_tv = False

    sino = np.load(f"Simulated_sinogram/new_shepp_logan_smoothed/sino_nl_{noise_level}_na_{num_angles}.npy")
    n = gt.shape[0]
    angles = np.linspace(0, 180, num_angles, endpoint=False)
    pixel_size = 2/n
    ag = AcquisitionGeometry.create_Parallel2D(rotation_axis_position=[0, 0])  \
                .set_panel(num_pixels=(n),pixel_size=(pixel_size,pixel_size))        \
                .set_angles(angles=angles) 
    ag.set_labels(['angle','horizontal'])
    ig = ImageGeometry(voxel_num_x=n, 
                voxel_num_y=n,
                voxel_size_x=pixel_size,
                voxel_size_y=pixel_size)
    PO = ProjectionOperator(ig,ag,"gpu")
    data = AcquisitionData(sino, geometry=ag, deep_copy=False)
    data.reorder('astra')
    gt_data = ImageData(gt,geometry=ig)
    delta = np.linalg.norm((PO.direct(gt_data)).array-sino)
    print(f"delta = {delta}")
    print(f"Norm of clean data = {np.linalg.norm(PO.direct(ImageData(gt,geometry=ig)).array)}")
    print(f"delta/norm(data): {delta/np.linalg.norm(PO.direct(ImageData(gt,geometry=ig)).array)}")
    LS = LeastSquares(A=PO, b=data)

    zooms = [1130,1430]
    gt_z = gt[zooms[0]:zooms[1],zooms[0]:zooms[1]]
    recs_mb = [gt]
    recs_mbtv = [gt]
    recs_tv = [gt]
    recs_mb_z = [gt_z]
    recs_tv_z = [gt_z]
    recs_mbtv_z = [gt_z]


    rrmse = []
    ssim = []
    mb_score = []
    titles_mb = []
    titles_tv = []
    titles_mbtv = []
    for i, alpha_mb in enumerate(alphas_mb):
        print(f"Using MB alpha:{alpha_mb*100}")
        try:
            MB_reco = np.load(args.np_folder+f"/MB_alpha_{alpha_mb}.npy")
            print("Loaded existing MB reco")
        except:
            MB = hf.Multibang(u=mb_values)
            x0_mb = ig.allocate(0)
            G = (alpha_mb*100)*MB
            fista_mb = FISTA(f=LS,g=G,initial=x0_mb,update_objective_interval=10)
            fista_mb.run(300)
            MB_reco = (fista_mb.solution).array
            np.save(args.np_folder+f"/MB_alpha_{alpha_mb}.npy",MB_reco)
        MB_data = ImageData(MB_reco,geometry=ig)
        mres = np.linalg.norm((PO.direct(MB_data)).array-sino)
        print(f"MB mres = {mres} at iteration {i}")
        titles_mb.append(r'$\alpha_{MB}$'+f"={alpha_mb}")
        recs_mb.append(MB_reco)
        recs_mb_z.append(MB_reco[zooms[0]:zooms[1],zooms[0]:zooms[1]])

        # MB + TV #######
        alpha_tvs = tv_alphas_smooth[i]
        print(f"Using MB+TV with mb alpha:{alpha_mb}, alpha_tv_smooth {alpha_tvs}")
        try:
            MBTV_reco = np.load(args.np_folder+f"/MBTV_alphamb_{alpha_mb}.npy")
            print("Loaded existing MB reco")
        except:
            MB = hf.Multibang(u=mb_values)
            x0_mbtv = ig.allocate(0)
            Grad = GradientOperator(ig)
            G = alpha_mb*MB
            F = LS + alpha_tvs * OperatorCompositionFunction(SmoothMixedL21Norm(epsilon), Grad)
            fista_mbtv = FISTA(f=F,g=G,initial=x0_mbtv,update_objective_interval=10)
            fista_mbtv.run(300)
            MBTV_reco = (fista_mbtv.solution).array
            np.save(args.np_folder+f"/MBTV_alphamb_{alpha_mb}.npy",MBTV_reco)
        MBTV_data = ImageData(MBTV_reco,geometry=ig)
        mres = np.linalg.norm((PO.direct(MBTV_data)).array-sino)
        print(f"MBTV mres = {mres} at iteration {i}")

        titles_mbtv.append(r'$\alpha_{MB}$'+f"={alpha_mb}"+r'$\alpha_{MB}$'+f"={alpha_tvs}")
        recs_mbtv.append(MBTV_reco)
        recs_mbtv_z.append(MBTV_reco[zooms[0]:zooms[1],zooms[0]:zooms[1]])

        print("MBTV metrics")
        RRMSE_MBTV, SSIM_MBTV, MB_SCORE_MBTV = hf.compute_metrics(gt,recs_mbtv[1:],get_values=True,print_out=True,list=True,mb_values=mb_values)
        # TV for comparison
        if skip_tv:
            print("Skipping TV")
            continue
        alpha_tv = alphas_tv[i]
        print(f"Using TV alpha:{alpha_tv}")
        try:
            TV_reco = np.load(args.np_folder+f"/TV_alpha_{alpha_tv}.npy")
            print("Loaded existing TV reco")
        except:
            Grad = GradientOperator(ig)
            x0 = ig.allocate(0.0)
            cb1 = callbacks.ProgressCallback() # This is the progress bar 
            cb2 = callbacks.EarlyStoppingObjectiveValue(threshold=1e-1)
            # TV = alpha_tv * OperatorCompositionFunction(SmoothMixedL21Norm(epsilon), Grad)
            # fista_TV = FISTA(initial=x0, f=LS+TV, g=None, update_objective_interval=1,max_iteration=300)
            TV = FGP_TV(alpha=alpha_tv, nonnegativity=False, device='gpu')
            fista_TV = FISTA(initial=x0, f=LS, g=TV, update_objective_interval=2)
            fista_TV.run(300,callbacks=[cb1]),#cb2]) 
            TV_reco = (fista_TV.solution).array
            np.save(args.np_folder+f"/TV_alpha_{alpha_tv}.npy",TV_reco)
        TV_data = ImageData(TV_reco,geometry=ig)
        mres = np.linalg.norm((PO.direct(TV_data)).array-sino)
        print(f"TV mres = {mres} at iteration {i}")
        recs_tv.append(TV_reco)
        titles_tv.append(r'$\alpha_{TV}$'+f"={alpha_tv}")
        recs_tv_z.append(TV_reco[zooms[0]:zooms[1],zooms[0]:zooms[1]])
    plt.rcParams.update({'font.size': 18})
    hf.plot_compare_big([recs_mb_z[1],recs_mb_z[mdp_mb+1],recs_mb_z[-1],
                           recs_mbtv_z[2],recs_mbtv_z[mdp_mbtv+1],recs_mbtv_z[-3],
                           recs_tv_z[1],recs_tv_z[mdp_tv-1],recs_tv_z[-2]],title=[r'$\alpha=10^{-7}$',"MDP",r'$\alpha=0.1$'],savefig=True,savepath=args.folder+"/3x3_compare.png",
                                                                            row_titles=["MB","MB+TV","TV"],
                                                                            use_range=1,rows=3)
    print("Created comparison plot")
    plt.rcParams.update({'font.size': 15})
    print("MB metrics")
    RRMSE, SSIM, MB_SCORE = hf.compute_metrics(gt,recs_mb[1:],get_values=True,print_out=True,list=True,mb_values=mb_values)
    print("MBTV metrics")
    RRMSE_MBTV, SSIM_MBTV, MB_SCORE_MBTV = hf.compute_metrics(gt,recs_mbtv[1:],get_values=True,print_out=True,list=True,mb_values=mb_values)
    print("TV metrics")
    RRMSE_TV, SSIM_TV, MB_SCORE_TV = hf.compute_metrics(gt,recs_tv[1:],get_values=True,print_out=True,list=True,mb_values=mb_values)
   
    plt.close()
    plt.clf()
    plt.figure()
    plt.loglog(alphas_mb,np.squeeze(np.array(RRMSE)),'--o',label="MB")
    plt.loglog(alphas_mb,np.squeeze(np.array(RRMSE_MBTV)),'--o',color="green",label="MB+TV")
    plt.loglog(alphas_mb[mdp_mbtv],np.array(RRMSE_MBTV)[mdp_mbtv],marker="o",color="orange")
    plt.loglog(alphas_mb[mdp_mb],np.array(RRMSE)[mdp_mb],marker="o",color="orange",label="Morozov's")
    plt.legend()
    plt.xlabel(r"$\alpha_{MB}$")
    plt.ylabel("RRMSE")
    plt.title("MB and MB+TV")
    plt.savefig(args.folder+f"/rrmse_sweep_mb.png", bbox_inches='tight',dpi = 1000)

    plt.close()
    plt.clf()
    plt.loglog(alphas_tv[4:-2],np.squeeze(np.array(RRMSE_TV)[4:-2]),'--o')
    plt.loglog(alphas_tv[mdp_tv],np.array(RRMSE_TV)[mdp_tv],marker="o",color="orange",label="Morozov's")
    plt.legend()
    plt.xlabel(r"$\alpha$")
    plt.ylabel("RRMSE")
    plt.title("TV")
    plt.savefig(args.folder+f"/rrmse_sweep_tv.png", bbox_inches='tight',dpi = 1000)


    ##############################################################################
    plt.close()
    plt.clf()
    plt.figure()
    plt.semilogx(alphas_mb,np.squeeze(np.array(SSIM)),'--o')
    plt.semilogx(alphas_mb[mdp_mb],np.array(SSIM)[mdp_mb],marker="o",color="orange",label="MDP")
    plt.legend()
    plt.xlabel(r"$\alpha_{MB}$")
    plt.ylabel("SSIM")
    plt.savefig(args.folder+f"/ssim_sweep_mb.png", bbox_inches='tight',dpi = 1000)

    plt.close()
    plt.clf()
    plt.semilogx(alphas_tv,np.squeeze(np.array(SSIM_TV)),'--o')
    plt.semilogx(alphas_tv[mdp_tv],np.array(SSIM_TV)[mdp_tv],marker="o",color="orange",label="MDP")
    plt.legend()
    plt.xlabel(r"$\alpha_{TV}$")
    plt.ylabel("SSIM")
    plt.savefig(args.folder+f"/ssim_sweep_tv.png", bbox_inches='tight',dpi = 1000)


    ##############################################################################3
    plt.close()
    plt.clf()
    plt.figure()
    plt.semilogx(alphas_mb,np.squeeze(np.array(MB_SCORE)),'--o')
    plt.semilogx(alphas_mb[mdp_mb],np.array(MB_SCORE)[mdp_mb],marker="o",color="orange",label="MDP")
    plt.legend()
    plt.xlabel(r"$\alpha_{MB}$")
    plt.ylabel("SS")
    plt.savefig(args.folder+f"/ss_sweep_mb.png", bbox_inches='tight',dpi = 1000)

    plt.close()
    plt.clf()
    plt.figure()
    plt.semilogx(alphas_tv,np.squeeze(np.array(MB_SCORE_TV)),'--o')
    plt.semilogx(alphas_tv[mdp_tv],np.array(MB_SCORE_TV)[mdp_tv],marker="o",color="orange",label="MDP")
    plt.legend()
    plt.xlabel(r"$\alpha_{TV}$")
    plt.ylabel("SS")
    plt.savefig(args.folder+f"/ss_sweep_tv.png", bbox_inches='tight',dpi = 1000)

    #################################################################################
    hf.plot_compare([recs_mb[0],recs_mb[1],recs_mb[mdp_mb+1],recs_mb[-1]],
                    title=[titles_mb[0],titles_mb[1],titles_mb[mdp_mb+1],titles_mb[-1]],savefig=True,
                    savepath=args.folder+"/MB_sweep_compare.png")
    hf.plot_compare([recs_mb_z[0],recs_mb_z[1],recs_mb_z[mdp_mb+1],recs_mb_z[-1]],
                    title=[titles_mb[0],titles_mb[1],titles_mb[mdp_mb+1],titles_mb[-1]],savefig=True,
                    savepath=args.folder+"/MB_sweep_zoom_compare.png")
    hf.plot_compare([recs_tv[0],recs_tv[1],recs_tv[mdp_tv+1],recs_tv[-1]],
                    title=[titles_tv[0],titles_tv[1],titles_tv[mdp_tv+1],titles_tv[-1]],savefig=True,
                    savepath=args.folder+"/TV_sweep_compare.png")
    hf.plot_compare([recs_tv_z[0],recs_tv_z[1],recs_tv_z[mdp_tv+1],recs_tv_z[-1]],
                    title=[titles_tv[0],titles_tv[1],titles_tv[mdp_tv+1],titles_tv[-1]],savefig=True,
                    savepath=args.folder+"/TV_sweep_zoom_compare.png")

    detector = 2500
    x1 = [1280,1280]
    y1 = [0,1280]

    plt.close()
    plt.clf()
    plt.plot(range(2560)[1000:1500],(recs_mb[mdp_mb])[1000:1500,1280],label="MB")
    plt.plot(range(2560)[1000:1500],(gt)[1000:1500,1280],label="GT")
    plt.plot(range(2560)[1000:1500],(recs_tv[mdp_tv])[1000:1500,1280],label="TV",linestyle="--")
    plt.xlabel("Vertical axis")
    plt.ylabel("Pixel value")
    plt.legend(loc=3)
    plt.show()
    plt.savefig("/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation/plots/new_shepp_logan_smoothed/mb_tv_sweep/mb_tv_mdp_lineplot.png",bbox_inches='tight',dpi = 1000)

    plt.close()
    plt.clf()
    plt.plot(range(2560)[1000:1500],(gt)[1000:1500,1280],label="GT")
    #plt.plot(range(2560)[480:1500],(recs_tv[mdp_tv])[480:1500,1280],label="TV",linestyle="--")
    plt.xlabel("Vertical axis")
    plt.ylabel("Pixel value")
    plt.legend(loc=3)
    plt.show()
    plt.savefig("/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation/plots/new_shepp_logan_smoothed/mb_tv_sweep/gt_lineplot.png",bbox_inches='tight',dpi = 1000)

