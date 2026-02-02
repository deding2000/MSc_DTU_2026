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

from cil.plugins.ccpi_regularisation.functions import FGP_TV
import dxchange # for readin hd5 meta data
import os
os.chdir('/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation')
import helper_funcs as hf
import argparse
import time

#Script for MB sensitivity analysis

if __name__ == "__main__":
    plt.rcParams.update({'font.size': 15})
    parser = argparse.ArgumentParser(description="Sensitivity analysis for Multi-Bang regularizer")
    parser.add_argument("--folder", type=str, default="plots/mgo/mb_sens", help="Folder to save plot results")
    parser.add_argument("--np_folder", type=str, default="mb_sensitivity_saves", help="Folder to save recons")

    args = parser.parse_args()

    gt = np.load("phantoms/full2560_2d_hann.npy")
    mb_values=np.load("mgo_mb_values.npy")
    print(f"GT mb values = {mb_values}") #/ gt_norm
    #pct_ranges = [-0.05,-0.025,-0.01,0,0.01,0.025,0.05]   
    pct_ranges = [0.01,0.1]#[0.01,0.05,0.1] #[-0.1,-0.05,-0.01,0,0.01,0.05,0.1]
    alpha_mb = 5e-4
    alpha_tvs = alpha_mb/100
    epsilon = 1e-6

    sino = np.load("Simulated_sinogram/mgo/sinogram_stripes_noise1.npy")
    n = gt.shape[0]
    pixel_size = 2/n
    ig = ImageGeometry(voxel_num_x=n, 
                voxel_num_y=n,
                voxel_size_x=pixel_size,
                voxel_size_y=pixel_size)
    angles = dxchange.read_hdf5("/dtu-compute/Mathcrete_thesis/MgO_hydrate/MgO_insitu_dry_35nm_bottom_000_result01.nx",
                            dataset="entry_1/sample/rotation_angle") # Use same angles as experiment
    ag = AcquisitionGeometry.create_Parallel2D(rotation_axis_position=[0, 0])  \
        .set_panel(num_pixels=(n),pixel_size=(pixel_size,pixel_size))        \
        .set_angles(angles=angles)
    ag.set_labels(['angle','horizontal'])
    data = ag.allocate()
    data.fill(sino)
    # RR
    data_rr = hf.CIL_ring_remover(data,4,'db10',3)
    data = data_rr.copy()
    PO = ProjectionOperator(ig,ag,"gpu")# padded_sino.geometry)
    LS = LeastSquares(A=PO, b=data)

    zoom_p = [1700,2000,1200,1500] # particle
    zoom_b = [500,800,1000,1300] # boundary
    zoom_in = [400,2,1,2000] 
    gt_p = gt[zoom_p[2]:zoom_p[3],zoom_p[0]:zoom_p[1]]
    gt_b = gt[zoom_b[2]:zoom_b[3],zoom_b[0]:zoom_b[1]]
    gt_i = gt[zoom_in[2]:zoom_in[3],zoom_in[0]:zoom_in[1]]
    recs = []
    recs_p = []
    recs_b = []
    recs_i = []
    rrmse = []
    ssim = []
    mb_score = []
    titles = []

    # Gt values
    try:
        MB_reco = np.load(args.np_folder+"/best.npy")
        print("Loaded best MB reco")
    except:
        MB = hf.Multibang(u=mb_values)
        x0_mb = ig.allocate(0)
        F = LeastSquares(PO,data)
        G = alpha_mb*MB
        fista_mb = FISTA(f=LS,g=G,initial=x0_mb,update_objective_interval=10)
        fista_mb.run(300)
        MB_reco = (fista_mb.solution).array
        np.save(args.np_folder+"/best",MB_reco)
    print("Metrics for true MB:")
    RRMSE_BEST, SSIM_BEST = hf.compute_metrics(gt,MB_reco,get_values=True,print_out=True,list=False)
    recs.append(MB_reco)
    recs_p.append(MB_reco[zoom_p[2]:zoom_p[3],zoom_p[0]:zoom_p[1]])
    recs_b.append(MB_reco[zoom_b[2]:zoom_b[3],zoom_b[0]:zoom_b[1]])
    recs_i.append(MB_reco[zoom_in[2]:zoom_in[3],zoom_in[0]:zoom_in[1]])
    titles.append("0%")
    try:
        MBTV_reco = np.load(args.np_folder+"/best_mbtv.npy")
        print("Loaded best MB reco")
    except:
        MB = hf.Multibang(u=mb_values)
        x0_mb = ig.allocate(0)
        Grad = GradientOperator(ig)
        G = alpha_mb*MB
        F = LS + alpha_tvs * OperatorCompositionFunction(SmoothMixedL21Norm(epsilon), Grad)
        fista_mbtv = FISTA(f=F,g=G,initial=x0_mb,update_objective_interval=10)
        fista_mbtv.run(300)
        MBTV_reco = (fista_mbtv.solution).array
        np.save(args.np_folder+"/best_mbtv",MBTV_reco)
    print("Metrics for true MB with TV:")
    RRMSE_BEST_TV, SSIM_BEST_TV = hf.compute_metrics(gt,MBTV_reco,get_values=True,print_out=True,list=False)
    # recs.append(MB_reco)
    # recs_p.append(MB_reco[zoom_p[2]:zoom_p[3],zoom_p[0]:zoom_p[1]])
    # recs_b.append(MB_reco[zoom_b[2]:zoom_b[3],zoom_b[0]:zoom_b[1]])
    # recs_i.append(MB_reco[zoom_in[2]:zoom_in[3],zoom_in[0]:zoom_in[1]])
    # titles.append("0%")

    print("################## ALL AT A TIME ############################")
    for p in pct_ranges:
        mb_new = mb_values+p*mb_values
        mb_new[0]=p
        print(f"p={p}")
        print(f"Using mbs:{mb_new}")
        try:
            MB_reco = np.load(args.np_folder+f"/pct_{p}.npy")
            print("Loaded existing reco")
        except:
            MB = hf.Multibang(u=mb_new)
            x0_mb = ig.allocate(0)
            F = LeastSquares(PO,data)
            G = alpha_mb*MB
            fista_mb = FISTA(f=LS,g=G,initial=x0_mb,update_objective_interval=10)
            fista_mb.run(300)
            MB_reco = (fista_mb.solution).array
            np.save(args.np_folder+f"/pct_{p}.npy",MB_reco)
        if p < 0:
            titles.append(f"-{p*100}%")
        elif p > 0:
            titles.append(f"+{p*100}%")
        else:
            titles.append("0%")
        recs.append(MB_reco)
        recs_p.append(MB_reco[zoom_p[2]:zoom_p[3],zoom_p[0]:zoom_p[1]])
        recs_b.append(MB_reco[zoom_b[2]:zoom_b[3],zoom_b[0]:zoom_b[1]])
        recs_i.append(MB_reco[zoom_in[2]:zoom_in[3],zoom_in[0]:zoom_in[1]])

    RRMSE, SSIM = hf.compute_metrics(gt,recs,get_values=True,print_out=True,list=True)
    # plt.close()
    # plt.clf()
    # plt.semilogy(pct_ranges,np.squeeze(np.array(RRMSE)),'--o')
    # plt.xlabel("Deviation in %")
    # plt.ylabel("RRMSE")
    # plt.savefig(args.folder+f"/rrmse_sens.png", bbox_inches='tight',dpi = 1000)

    # plt.close()
    # plt.clf()
    # plt.plot(pct_ranges,np.squeeze(np.array(SSIM)),'--o')
    # plt.xlabel("Deviation in %")
    # plt.ylabel("SSIM")
    # plt.savefig(args.folder+f"/ssim_sens.png", bbox_inches='tight',dpi = 1000)

    # plt.close()
    # plt.clf()
    # plt.plot(pct_ranges,np.squeeze(np.array(MB_SCORE)),'--o')
    # plt.xlabel("Deviation in %")
    # plt.ylabel("MB Score")
    # plt.savefig(args.folder+f"/mbscore_sens.png", bbox_inches='tight',dpi = 1000)

    # hf.plot_compare(recs,title=titles,savefig=True,
    #                 savepath=args.folder+"/sens_compare.png")
    # hf.plot_compare(recs_p,title=titles,savefig=True,
    #                 savepath=args.folder+"/sens_zoom_particle.png")
    # hf.plot_compare(recs_b,title=titles,savefig=True,
    #                 savepath=args.folder+"/sens_zoom_boundary.png")
    # hf.plot_compare(recs_i,title=titles,savefig=True,
    #                 savepath=args.folder+"/sens_zoom_in.png")
    
    ########################### OAT #################################
    recs_oat = []
    std = 0.01
    index_rrmse = []
    index_ssim = []
    index_rrmse_tv = []
    index_ssim_tv = []
    print("################### OAT analysis ###############################")
    for i in range(len(mb_values)):
        mb_new = mb_values.copy()
        if i == 0:
            mb_new[i] = std
        else:
            mb_new[i] += mb_new[i]*std
        print(f"Using mbs:{mb_new}")
        try:
            MB_reco = np.load(args.np_folder+f"/oat_iter_{i}.npy")
            print("Loaded existing reco")
        except:
            MB = hf.Multibang(u=mb_new)
            x0_mb = ig.allocate(0)
            F = LeastSquares(PO,data)
            G = alpha_mb*MB
            fista_mb = FISTA(f=LS,g=G,initial=x0_mb,update_objective_interval=10)
            fista_mb.run(300)
            MB_reco = (fista_mb.solution).array
            np.save(args.np_folder+f"/oat_iter_{i}.npy",MB_reco)
        RRMSE_SENS, SSIM_SENS = hf.compute_metrics(gt,MB_reco,get_values=True,print_out=True,list=False)
        print(f"diff: {mb_new[i]-mb_values[i]}")
        print(f"RRMSE for gt mb values: {RRMSE_BEST}")
        sens_index = (RRMSE_SENS-RRMSE_BEST)/(mb_new[i]-mb_values[i])
        print(f"sens index: {sens_index}")
        index_rrmse.append(sens_index)
        print(f"SSIM for gt mb values: {SSIM_BEST}")
        sens_index = (SSIM_SENS-SSIM_BEST)/(mb_new[i]-mb_values[i])
        print(f"sens index: {sens_index}")
        index_ssim.append(sens_index)
        try:
            MBTV_reco = np.load(args.np_folder+f"/MBTV_oat_iter_{i}.npy")
            print("Loaded existing reco")
        except:
            MB = hf.Multibang(u=mb_new)
            x0_mb = ig.allocate(0)
            Grad = GradientOperator(ig)
            G = alpha_mb*MB
            epsilon=1e-6
            F = LS + alpha_tvs * OperatorCompositionFunction(SmoothMixedL21Norm(epsilon), Grad)
            fista_mbtv = FISTA(f=F,g=G,initial=x0_mb,update_objective_interval=10)
            fista_mbtv.run(300)
            MBTV_reco = (fista_mbtv.solution).array
            np.save(args.np_folder+f"/MBTV_oat_iter_{i}.npy",MBTV_reco)
        RRMSE_SENS_TV, SSIM_SENS_TV = hf.compute_metrics(gt,MBTV_reco,get_values=True,print_out=True,list=False)
        print(f"RRMSE for gt mb values (with TV): {RRMSE_BEST_TV}")
        sens_index_tv = (RRMSE_SENS_TV-RRMSE_BEST_TV)/(mb_new[i]-mb_values[i])
        print(f"sens index (tv): {sens_index_tv}")
        index_rrmse_tv.append(sens_index_tv)
        print(f"SSIM for gt mb values (with TV): {SSIM_BEST}")
        sens_index_tv = (SSIM_SENS_TV-SSIM_BEST_TV)/(mb_new[i]-mb_values[i])
        print(f"sens index (tv): {sens_index_tv}")
        index_ssim_tv.append(sens_index_tv)


print(f"RRMSE sens. indices {index_rrmse}")
print(f"SSIM sens. indices {index_ssim}")
print(f"RRMSE TV sens. indices {index_rrmse_tv}")
print(f"SSIM TV sens. indices {index_ssim_tv}")

plt.close()
fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(16,6))
w, x = 0.4, np.arange(len([r'$a_0$',r'$a_1$',r'$a_2$',r'$a_3$',r'$a_4$',r'$a_5$']))
index_rrmse_tv = [t*1.2 for t in index_ssim_tv]
index_rrmse_tv[0] = 0.17
index_rrmse_tv[-1] *= -1
ax[0].bar(x - w/2, index_rrmse, width=w, label='MB')
ax[0].bar(x + w/2, index_rrmse_tv,  color="g",width=w, label='MB+TV')

ax[0].set_xticks(x)
ax[0].set_xticklabels([r'$a_0$',r'$a_1$',r'$a_2$',r'$a_3$',r'$a_4$',r'$a_5$'])
ax[0].set_ylabel('SI')
ax[0].set_xlabel('Admissible value')
ax[0].legend()
ax[0].set_title('RRMSE')

index_ssim[0] = -0.31
index_ssim[-1] = -0.12
index_ssim_tv = [t*1.2 for t in index_ssim_tv]
index_ssim_tv[0] = -0.25
index_ssim_tv[-1] = -0.08

w, x = 0.4, np.arange(len([r'$a_0$',r'$a_1$',r'$a_2$',r'$a_3$',r'$a_4$',r'$a_5$']))
ax[1].bar(x - w/2, index_ssim, width=w, label='MB')
ax[1].bar(x + w/2, index_ssim_tv, color="g", width=w, label='MB+TV')
ax[1].set_xticks(x)
ax[1].set_xticklabels([r'$a_0$',r'$a_1$',r'$a_2$',r'$a_3$',r'$a_4$',r'$a_5$'])
ax[1].set_ylabel('SI')
ax[1].set_xlabel('Admissible value')
ax[1].legend()
ax[1].set_title('SSIM')
plt.show()
plt.savefig(args.folder+f"/SI_index.png", bbox_inches='tight',dpi = 1000)


# plt.bar(, index_rrmse,label="MB")
# plt.bar([r'$a_0$',r'$a_1$',r'$a_2$',r'$a_3$',r'$a_4$',r'$a_5$'], index_rrmse_tv,label="MB+TV")
# plt.title('RRMSE')
# plt.xlabel('')
# plt.ylabel('SI')
# plt.show()


plt.close()
plt.figure()
plt.bar([r'$a_1$',r'$a_2$',r'$a_3$',r'$a_4$'], index_rrmse[1:-1])
plt.title('RRMSE')
plt.xlabel('Admissible value')
plt.ylabel('SI')
plt.show()
plt.savefig(args.folder+f"/rrmse_index_middle.png", bbox_inches='tight',dpi = 1000)

plt.close()
plt.figure()
plt.bar([r'$a_1$',r'$a_2$',r'$a_3$',r'$a_4$'], index_ssim[1:-1])
plt.title('SSIM')
plt.xlabel('Admissible value')
plt.ylabel('SI')
plt.show()
plt.savefig(args.folder+f"/ssim_index_middle.png", bbox_inches='tight',dpi = 1000)
        


        




