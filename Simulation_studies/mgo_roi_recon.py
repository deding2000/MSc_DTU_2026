import matplotlib.pyplot as plt
import numpy as np

# CIL methods
from cil.plugins.astra import ProjectionOperator
from cil.recon import FBP
from cil.optimisation.functions import OperatorCompositionFunction
from cil.optimisation.functions import SmoothMixedL21Norm
from cil.optimisation.operators import GradientOperator
from cil.optimisation.algorithms import CGLS
from cil.optimisation.algorithms import FISTA
from cil.optimisation.functions import LeastSquares
from cil.optimisation.functions import OperatorCompositionFunction
from cil.optimisation.utilities import callbacks
from cil.optimisation.operators import GradientOperator
from cil.optimisation.utilities import callbacks
from cil.framework import ImageData
from cil.plugins.ccpi_regularisation.functions import FGP_TV

import helper_funcs as hf
import time

from cil.optimisation.operators import LinearOperator
from cil.optimisation.operators import Operator, CompositionOperator

# Scrip for model-based reconstructions of MgO ROI simulations 

if __name__ == "__main__":
    gt_roi = np.load("/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation/phantoms/full2560_2d_hannROI.npy")
    sino_trunc = np.load("/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation/Simulated_sinogram/mgo/sinogram_noisy_ROI_TRUNC.npy")
    zoom = [0,1500,0,1500]
    ag, ig, data_trunc = hf.CIL_setup_cement(sino_trunc,pixel_size=1)
    data_trunc = hf.CIL_ring_remover(data_trunc,4,'db10',3)

    zoom2 = [750,1050,200,500]
    n = gt_roi.shape[0]
    ## CGLS with masked forward operator ###
    extend = int(0.5*data_trunc.shape[0])
    y_pad = np.pad(data_trunc.array,((0,0),(extend,extend))) 

    ag_full, ig_full, data_pad = hf.CIL_setup_cement(y_pad,pixel_size=1,angle_range=360)
    Radon_full = ProjectionOperator(ig_full, data_pad.geometry, device = "gpu")
    MaskOP = hf.MaskOperator(domain_geometry=data_pad.geometry,range_geometry=data_pad.geometry,r=[extend,extend+2560])
    AM = CompositionOperator(MaskOP,Radon_full) 
    LS_mask = LeastSquares(A=AM, b=data_pad)#padded_sino)#b=data_nf)
    try:
        cgls_roi = np.load("/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation/roi_saves/cgls_mask_roi.npy")
        cgls_full = np.load("/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation/roi_saves/cgls_mask_full.npy")
    except:
        zero_data = ig_full.allocate(0)
        cgls_mask = CGLS(operator=AM, data=data_pad, update_objective_interval=2)
        cb1=callbacks.ProgressCallback()
        cgls_mask.run(300,callbacks=[cb1])
        mask_reco = (cgls_mask.solution)
        np.save("/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation/roi_saves/cgls_mask_full",mask_reco.array)
        cgls_roi = mask_reco[extend:extend+2560,extend:extend+2560]
        np.save("/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation/roi_saves/cgls_mask_roi",cgls_roi)
    
    fbp50 = np.load("/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation/roi_saves/fbp_50_padding.npy")
    gt_m = hf.circle_mask(gt_roi)
    gt_mz = gt_m[zoom2[0]:zoom2[1],zoom2[2]:zoom2[3]]
    recfbp_m = hf.circle_mask(fbp50)
    recfbp_mz = recfbp_m[zoom2[0]:zoom2[1],zoom2[2]:zoom2[3]]
    CGLS_m = hf.circle_mask(cgls_roi)
    CGLS_mz = CGLS_m[zoom2[0]:zoom2[1],zoom2[2]:zoom2[3]]
    hf.compute_metrics(gt_m,[recfbp_m,CGLS_m],list=True)

    ######################## TV ################################
    alpha_tv = 1000 #100 too low
    try:
        TV_reco = np.load(f"/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation/roi_saves/tv_alpha_{alpha_tv}.npy")
        print("Loaded TV")
    except:
        TV = FGP_TV(alpha=alpha_tv, nonnegativity=False, device='gpu')
        zero_data = ig_full.allocate(0)
        fista_TV = FISTA(initial=zero_data, f=LS_mask, g=TV, update_objective_interval=2)
        cb1=callbacks.ProgressCallback()
        fista_TV.run(300,callbacks=[cb1])
        TV_reco = (fista_TV.solution).array[extend:extend+n,extend:extend+n]
        np.save(f"/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation/roi_saves/tv_alpha_{alpha_tv}",TV_reco)
        hf.plot_compare([gt_roi[zoom[0]:zoom[1],zoom[0]:zoom[1]],
                        fbp50[zoom[0]:zoom[1],zoom[0]:zoom[1]],
                        cgls_roi[zoom[0]:zoom[1],zoom[0]:zoom[1]],
                        TV_reco[zoom[0]:zoom[1],zoom[0]:zoom[1]]],
                        title=["True ROI", "Padding 50%", "Flat prior", "TV"],
                                            savefig=True,
                    savepath="/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation/plots/mgo_roi/gt_cgls_tv")
    TV_reco_m = hf.circle_mask(TV_reco)
    hf.compute_metrics(gt_m,[recfbp_m,CGLS_m,TV_reco_m],list=True)
    TV_reco_mz = TV_reco_m[zoom2[0]:zoom2[1],zoom2[2]:zoom2[3]]
    hf.plot_compare([gt_mz,recfbp_mz,CGLS_mz,TV_reco_mz],title=["GT","FBP","Flat prior","TV mask"],
                                        savefig=True,
                    savepath="/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation/plots/mgo_roi/gt_cgls_tv_zoom")

    ############################################ MB + TV ###########################################
    mb_values = np.load("/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation/mgo_mb_values.npy")
    print(f"Mbs: {mb_values}")
    alpha_tv_smooth = 950 
    alpha_mb = 3000 # mb 3000 and tv 800 best so far
    try:
        MBTV_reco = np.load(f"/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation/roi_saves/mbtv_alphamb_{alpha_mb}_alphatv_{alpha_tv_smooth}.npy")
        print("Loaded MB+TV")
    except:
        x0_mbtv = ImageData(cgls_full,geometry=ig_full)
        MB = hf.Multibang(u=mb_values)
        Grad = GradientOperator(ig_full)
        cb1 = callbacks.ProgressCallback()
        G = alpha_mb*MB
        epsilon = 1e-6 # for smoothed TV
        F = LS_mask + alpha_tv_smooth * OperatorCompositionFunction(SmoothMixedL21Norm(epsilon), Grad)
        fista_MBTV = FISTA(f=F,g=G,initial=x0_mbtv,update_objective_interval=1,max_iteration=400)
        fista_MBTV.run(300,callbacks=[cb1])
        MBTV_reco = (fista_MBTV.solution).array[extend:extend+n,extend:extend+n]
        np.save(f"/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation/roi_saves/mbtv_alphamb_{alpha_mb}_alphatv_{alpha_tv_smooth}",MBTV_reco)
    MBTV_reco_m = hf.circle_mask(MBTV_reco)
    MBTV_reco_mz = MBTV_reco_m[zoom2[0]:zoom2[1],zoom2[2]:zoom2[3]]
    try: 
        NF_reco = np.load("/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation/neural_field_saves/fbp_mgo_roi_trunc/nf_recon.npy")#*np.mean(gt_roi**2)**(1/2)
        print("Loaded NF recon")
    except:
        import torch
        print(f"Missing nf recon")
        print("Reconstructing from network")
        assert torch.cuda.is_available()
        dev = torch.device("cuda:0")
        net = hf.real_nf_sim(300, 3,activation="SIREN").to(dev)
        state_dict = torch.load("/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation/neural_field_saves/fbp_mgo_roi_trunc/net_fbp")
        net.load_state_dict(state_dict)
        N = gt_roi.shape[0]
        grid_mks = (2*np.arange(N) + 1)/(2*N) - 1/2
        c0, c1 = np.meshgrid(grid_mks, grid_mks)
        XY = torch.stack((torch.from_numpy(c0.flatten()),torch.from_numpy(c1.flatten())), axis = 1).float()
        F_np = np.zeros(N**2)

        nb = 10
        with torch.no_grad():
            for b in range(nb):
                F_np[b::nb] = net(XY[b::nb,:].to(dev)).cpu().detach().numpy().flatten()
            NF_reco = (F_np.reshape((N,N)))*np.mean(gt_roi**2)**(1/2)
        np.save("/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation/neural_field_saves/fbp_mgo_roi_trunc/nf_recon",NF_reco)
        print("NF reco saved")
    NF_reco_m = hf.circle_mask(NF_reco)
    NF_reco_mz = NF_reco_m[zoom2[0]:zoom2[1],zoom2[2]:zoom2[3]]

    # Metrics and plot
    hf.compute_metrics(gt_m,[recfbp_m,CGLS_m,TV_reco_m,NF_reco_m,MBTV_reco_m],list=True,mb_values=mb_values)

    # hf.plot_compare([gt_roi[zoom[0]:zoom[1],zoom[0]:zoom[1]],
    #                 fbp50[zoom[0]:zoom[1],zoom[0]:zoom[1]],
    #                 cgls_roi[zoom[0]:zoom[1],zoom[0]:zoom[1]],
    #                 TV_reco[zoom[0]:zoom[1],zoom[0]:zoom[1]],
    #                 NF_reco[zoom[0]:zoom[1],zoom[0]:zoom[1]],
    #                 MBTV_reco[zoom[0]:zoom[1],zoom[0]:zoom[1]]],
    #                 title=["True ROI", "FBP", "CGLS", "TV", "NF+TV","MB+TV"],savefig=True,
    #                 savepath="/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation/plots/mgo_roi/compare1.png")
    hf.plot_compare([gt_mz,recfbp_mz,CGLS_mz,TV_reco_mz],
                    title=["Ground truth", "FBP", "Flat prior", "TV"],savefig=True,
                    savepath="/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation/plots/mgo_roi/compare1.png")
    hf.plot_compare([gt_mz,TV_reco_mz,NF_reco_mz,MBTV_reco_mz],
                    title=["Ground truth", "TV", "NF+TV", "MB+TV"],savefig=True,
                    savepath="/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation/plots/mgo_roi/compare2.png")
    hf.plot_compare([gt_roi[zoom[0]:zoom[1],zoom[0]:zoom[1]],
                    fbp50[zoom[0]:zoom[1],zoom[0]:zoom[1]],
                    cgls_roi[zoom[0]:zoom[1],zoom[0]:zoom[1]],
                    TV_reco[zoom[0]:zoom[1],zoom[0]:zoom[1]]],
                    title=["Ground truth", "FBP", "Flat prior", "TV"],savefig=True,
                    savepath="/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation/plots/mgo_roi/compareglobal1.png")
    hf.plot_compare([gt_roi[zoom[0]:zoom[1],zoom[0]:zoom[1]],TV_reco[zoom[0]:zoom[1],zoom[0]:zoom[1]],
                    NF_reco[zoom[0]:zoom[1],zoom[0]:zoom[1]],
                    MBTV_reco[zoom[0]:zoom[1],zoom[0]:zoom[1]]],
                    title=["Ground truth", "TV", "NF+TV", "MB+TV"],savefig=True,
                    savepath="/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation/plots/mgo_roi/compareglobal2.png")
    
    #hf.plot_compare([gt_b,NF_b,MB_b,MBTV_b],title=["GT","NF+TV","MB","TV + MB"],savefig=True,savepath=args.folder+f"/tv_mb_fbp_nf_compare_zoom_boundary2.png")
    hf.plot_compare_mrows([gt_roi[zoom[0]:zoom[1],zoom[0]:zoom[1]],
                    fbp50[zoom[0]:zoom[1],zoom[0]:zoom[1]],
                    cgls_roi[zoom[0]:zoom[1],zoom[0]:zoom[1]],
                    TV_reco[zoom[0]:zoom[1],zoom[0]:zoom[1]],
                    NF_reco[zoom[0]:zoom[1],zoom[0]:zoom[1]],
                    MBTV_reco[zoom[0]:zoom[1],zoom[0]:zoom[1]]],
                    title=["True ROI", "FBP", "Flat prior", "TV", "NF+TV","MB+TV"],savefig=True,
                    savepath="/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation/plots/mgo_roi/gt_fbp_cgls_tv_mbtv_nf",rows=2)
    
    
    
    hf.plot_compare_mrows([gt_mz,recfbp_mz,CGLS_mz,TV_reco_mz,NF_reco_mz,MBTV_reco_mz],title=["GT","FBP","Flat prior","TV","NF+TV","MB+TV"],
                    savefig=True,
                    savepath="/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation/plots/mgo_roi/gt_fbp_cgls_tv_mbtv_nf_zoom",rows=2)