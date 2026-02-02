# CIL methods
from cil.framework import ImageData, AcquisitionGeometry, AcquisitionData, BlockDataContainer, ImageGeometry
from cil.utilities.display import show2D, show_geometry
from cil.processors import CentreOfRotationCorrector, Slicer, TransmissionAbsorptionConverter, Normaliser, Padder, RingRemover
from cil.io import TIFFStackReader
from cil.plugins.astra import ProjectionOperator, FBP
from cil.recon import FBP
from cil.optimisation.functions import OperatorCompositionFunction, Function
from cil.optimisation.functions import L2NormSquared, L1Norm, BlockFunction, MixedL21Norm, IndicatorBox, TotalVariation
from cil.optimisation.operators import GradientOperator, BlockOperator, IdentityOperator
from cil.optimisation.algorithms import CGLS
from cil.optimisation.algorithms import FISTA, ISTA
from cil.optimisation.functions import Function, LeastSquares, ZeroFunction
from cil.optimisation.functions import OperatorCompositionFunction
from cil.optimisation.utilities import callbacks
from cil.optimisation.functions import SmoothMixedL21Norm
from cil.optimisation.operators import BlockOperator, GradientOperator
from cil.plugins.ccpi_regularisation.functions import FGP_TV
from cil.optimisation.operators import Operator, CompositionOperator

# Additional packages
import numpy as np # conda install numpy
import matplotlib.pyplot as plt # conda install matplotlib
import dxchange # for readin hd5 meta data

#helper functions
import os
import helper_funcs as hf

# Script for model-based and NF reconstructions of the real data slice

if __name__ == "__main__":

    ctf_pm = np.load("/dtu-compute/Mathcrete_thesis/MgO_hydrate/CHR_4h_ctf_nofilter/MgO_insitu_water_35nm_bottom_011_pm_slice_1000.npy")
    angles = dxchange.read_hdf5("/dtu-compute/Mathcrete_thesis/MgO_hydrate/MgO_insitu_dry_35nm_bottom_000_result01.nx",
                            dataset="entry_1/sample/rotation_angle")
    print(f"Angles from nx file: {angles}")
    print(f"Data shape {ctf_pm.shape}")
    zoomb = [700,1200,200,700] # boundary
    zoomc = [1130,1430,1130,1430] # central
    zoomw = [450,950,1130,1630] # water
    zoomd = [1550,2050,1550,2050] # down right
    COR_NABU = 1234.7 # for 4h
    cor = -(2560//2 - COR_NABU)
    pixel_size = 1
    ag = AcquisitionGeometry.create_Parallel2D(rotation_axis_position=[cor,0],units='m')  \
            .set_panel(num_pixels=(ctf_pm.shape[1]),pixel_size=(pixel_size,pixel_size))\
            .set_angles(angles=angles)
    ag.set_labels(['angle','horizontal'])
    ig = ag.get_ImageGeometry()
    data_ctf = AcquisitionData(ctf_pm, geometry=ag, deep_copy=False)
    data_ctf.reorder('astra') # set order
    data_rr_ctf = hf.CIL_ring_remover(data_ctf,decNum=4,wname="db10",sigma=12)
    padsize = int(data_ctf.shape[0]*0.5)
    padded_sino = Padder.edge(pad_width={'horizontal': padsize})(data_rr_ctf)

    ## FBP ##
    recfbp = FBP(padded_sino, ig, backend='astra',filter='ram-lak').run(verbose=0).array
    fbp_b = recfbp[zoomb[0]:zoomb[1],zoomb[2]:zoomb[3]]
    fbp_c = recfbp[zoomc[0]:zoomc[1],zoomc[2]:zoomc[3]]
    fbp_w = recfbp[zoomw[0]:zoomw[1],zoomw[2]:zoomw[3]]
    fbp_d = recfbp[zoomd[0]:zoomd[1],zoomd[2]:zoomd[3]]
    
    ## CGLS ##
    extend = int(0.5*data_rr_ctf.shape[1])
    print(f"Extending image region by {extend}")
    y_pad = np.pad(data_rr_ctf.array,((0,0),(extend,extend))) 
    print(f"Using cor. {cor}")
    ag_full, ig_full, data_pad = hf.CIL_setup_cement(y_pad,pixel_size=1,angle_range=360,cor=cor)
    Radon_full = ProjectionOperator(ig_full, data_pad.geometry, device = "gpu")
    MaskOP = hf.MaskOperator(domain_geometry=data_pad.geometry,range_geometry=data_pad.geometry,r=[extend,extend+2560])
    AM = CompositionOperator(MaskOP,Radon_full) 
    LS_mask = LeastSquares(A=AM, b=data_pad)#padded_sino)#b=data_nf)
    try:
        cgls_roi = np.load("/dtu-compute/Mathcrete_thesis/mathcrete_dtu/real_data/cgls_mask_roi.npy")
        cgls_full = np.load("/dtu-compute/Mathcrete_thesis/mathcrete_dtu/real_data/cgls_mask_full.npy")
        print("Loaded CGLS Recon")
    except:
        zero_data = ig_full.allocate(0)
        cgls_mask = FISTA(f=LS_mask,g=None,initial=zero_data,update_objective_interval=1,max_iteration=400)
        cb1=callbacks.ProgressCallback()
        cgls_mask.run(300,callbacks=[cb1])
        mask_reco = (cgls_mask.solution).array
        np.save("/dtu-compute/Mathcrete_thesis/mathcrete_dtu/real_data/cgls_mask_full",mask_reco)
        cgls_roi = mask_reco[extend:extend+2560,extend:extend+2560]
        np.save("/dtu-compute/Mathcrete_thesis/mathcrete_dtu/real_data/cgls_mask_roi",cgls_roi)

    LS_b = cgls_roi[zoomb[0]:zoomb[1],zoomb[2]:zoomb[3]]
    LS_c = cgls_roi[zoomc[0]:zoomc[1],zoomc[2]:zoomc[3]]
    LS_w = cgls_roi[zoomw[0]:zoomw[1],zoomw[2]:zoomw[3]]
    LS_d = cgls_roi[zoomd[0]:zoomd[1],zoomd[2]:zoomd[3]]

    hf.plot_compare([recfbp,cgls_roi],title=["FBP","CGLS with mask operator"],
                savefig=True,
                savepath="/dtu-compute/Mathcrete_thesis/mathcrete_dtu/real_data/plots/model-based/fbp_cgls.png")
    plt.close
    plt.figure()
    plt.imshow(recfbp-cgls_roi,cmap="gray")
    plt.savefig("/dtu-compute/Mathcrete_thesis/mathcrete_dtu/real_data/plots/model-based/fbp_cgls_diff.png")
    hf.plot_compare([recfbp,cgls_roi],title=["FBP","CGLS with mask operator"],
                savefig=True,
                savepath="/dtu-compute/Mathcrete_thesis/mathcrete_dtu/real_data/plots/model-based/fbp_cgls.png")

    #### TV ####
    LS = LeastSquares(A=AM, b=padded_sino)
    alpha_tv = 5
    
    try: 
        TV_reco = np.load(f"/dtu-compute/Mathcrete_thesis/mathcrete_dtu/real_data/TV_alpha_{alpha_tv}.npy")
        print("Loaded TV recon")
    except:
        TV = FGP_TV(alpha=alpha_tv, nonnegativity=False, device='gpu')
    
        zero_data = ig.allocate(0)
        fista_TV = FISTA(initial=zero_data, f=LS, g=TV, update_objective_interval=2)
        cb1=callbacks.ProgressCallback()
        fista_TV.run(400,callbacks=[cb1])
        plt.plot(fista_TV.objective)
        plt.ylabel("Objective value")
        plt.show()
        plt.savefig("/dtu-compute/Mathcrete_thesis/mathcrete_dtu/real_data/ctf_tv_loss",bbox_inches='tight',dpi = 1000)
        TV_reco = (fista_TV.solution).array
        np.save(f"/dtu-compute/Mathcrete_thesis/mathcrete_dtu/real_data/TV_alpha_{alpha_tv}",TV_reco)

    TV_b = TV_reco[zoomb[0]:zoomb[1],zoomb[2]:zoomb[3]]
    TV_c = TV_reco[zoomc[0]:zoomc[1],zoomc[2]:zoomc[3]]
    TV_w = TV_reco[zoomw[0]:zoomw[1],zoomw[2]:zoomw[3]]
    TV_d = TV_reco[zoomd[0]:zoomd[1],zoomd[2]:zoomd[3]]
    ### MB ####
    alpha_mbs = [500,2000,5000]
    alpha_tvs = [t/100 for  t in alpha_mbs]
    mb_values = np.load("/dtu-compute/Mathcrete_thesis/mathcrete_dtu/real_data/real_mbs_6.npy")
    print(f"Mb values: {mb_values}")
    MB_b = []
    MB_c = []
    MB_w = []
    MB_d = []
    MBrecs = []
    mi = 1 # index to plot with other methods
    for alpha_mb in alpha_mbs:
        print(f"alpha mb {alpha_mb}, alpha tv smooth {alpha_tvs}")
        try:
            MBTV_reco =np.load(f"/dtu-compute/Mathcrete_thesis/mathcrete_dtu/real_data/mb_recons_6vals/MB_alphamb_{alpha_mb}_alphatvs_{alpha_tvs}.npy")
            print("Loaded MB+TV recon")
        except:
            initial = ImageData(recfbp,geometry=ig)
            MB = hf.Multibang(u=mb_values)
            epsilon = 1e-6
            #x0_mb = ig.allocate(fbp_data)
            Grad = GradientOperator(ig)
            cb1 = callbacks.ProgressCallback() # This is the progress bar 
            G = alpha_mb*MB
            F = LS_mask + alpha_tvs * OperatorCompositionFunction(SmoothMixedL21Norm(epsilon), Grad)
            fista_MBTV = FISTA(f=F,g=G,initial=initial,update_objective_interval=1,max_iteration=400)
            fista_MBTV.run(500,callbacks=[cb1])
            MBTV_reco = (fista_MBTV.solution).array
            np.save(f"/dtu-compute/Mathcrete_thesis/mathcrete_dtu/real_data/mb_recons_6vals/MB_alphamb_{alpha_mb}_alphatvs_{alpha_tvs}",MBTV_reco)
        MB_b.append(MBTV_reco[zoomb[0]:zoomb[1],zoomb[2]:zoomb[3]])
        MB_c.append(MBTV_reco[zoomc[0]:zoomc[1],zoomc[2]:zoomc[3]])
        MB_w.append(MBTV_reco[zoomw[0]:zoomw[1],zoomw[2]:zoomw[3]])
        MB_d.append(MBTV_reco[zoomd[0]:zoomd[1],zoomd[2]:zoomd[3]])
        MBrecs.append(MBTV_reco)
    print(f"SNR: FBP {hf.snr(recfbp)}, MB w. alpha {alpha_mbs[0]} {hf.snr(MBrecs[0])}, MB w. alpha {alpha_mbs[1]} {hf.snr(MBrecs[1])}, MB w. alpha {alpha_mbs[2]} {hf.snr(MBrecs[2])}")
    hf.plot_compare(MB_b,title=[r"$\alpha_{MB}=$"+f"{alpha_mbs[0]}",r"$\alpha_{MB}=$"+f"{alpha_mbs[1]}",r"$\alpha_{MB}=$"+f"{alpha_mbs[2]}"],
                    savefig=True,savepath="/dtu-compute/Mathcrete_thesis/mathcrete_dtu/real_data/plots/model-based/different_mb_alphas.png")
    # NF + TV
    try: 
        NF_reco = np.load("/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation/neural_field_saves/mgo_real/nf_recon_tv1e-5_prox200.npy")
        print("Loaded NF recon")
    except:
        import torch
        print(f"Missing nf recon")
        print("Reconstructing from network")
        assert torch.cuda.is_available()
        dev = torch.device("cuda:0")
        net = hf.real_nf_sim(300, 3,activation="SIREN").to(dev)
        # Be careful of what is here
        state_dict = torch.load("/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation/neural_field_saves/mgo_real/nf_recon/net_0")#_tv1e-5_prox200")
        net.load_state_dict(state_dict)
        N = TV_reco.shape[0]
        grid_mks = (2*np.arange(N) + 1)/(2*N) - 1/2
        c0, c1 = np.meshgrid(grid_mks, grid_mks)
        XY = torch.stack((torch.from_numpy(c0.flatten()),torch.from_numpy(c1.flatten())), axis = 1).float()
        F_np = np.zeros(N**2)

        nb = 10
        with torch.no_grad():
            for b in range(nb):
                F_np[b::nb] = net(XY[b::nb,:].to(dev)).cpu().detach().numpy().flatten()
            NF_reco = (F_np.reshape((N,N)))*np.mean(recfbp**2)**(1/2)
        np.save("/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation/neural_field_saves/mgo_real/nf_recon",NF_reco)
        print("NF reco saved")
    NF_b = NF_reco[zoomb[0]:zoomb[1],zoomb[2]:zoomb[3]]
    NF_c = NF_reco[zoomc[0]:zoomc[1],zoomc[2]:zoomc[3]]
    NF_w = NF_reco[zoomw[0]:zoomw[1],zoomw[2]:zoomw[3]]
    NF_d = NF_reco[zoomd[0]:zoomd[1],zoomd[2]:zoomd[3]]

    print(f"SNR: FBP {hf.snr(recfbp)}, with TV {hf.snr(TV_reco)}, with MB + TV {hf.snr(MBrecs[mi])}, with NF + TV {hf.snr(NF_reco)}")
    plt.rcParams.update({'font.size': 18})
    hf.plot_compare_big([fbp_d,TV_d,MB_d[mi],NF_d,
                         fbp_b,TV_b,MB_b[mi],NF_b,
                         fbp_c,TV_c,MB_c[mi],NF_c,
                         fbp_w,TV_w,MB_w[mi],NF_w],
    title=["FBP","TV","MB+TV","NF+TV"],
    rows=4,
    savefig=True, use_range=0,
    savepath="/dtu-compute/Mathcrete_thesis/mathcrete_dtu/real_data/plots/model-based/big_compare.png")
    hf.plot_compare_diff([fbp_b,NF_b,fbp_b-NF_b],title=["FBP","NF+TV","Difference"],savefig=True,
                         savepath="/dtu-compute/Mathcrete_thesis/mathcrete_dtu/real_data/plots/model-based/NF_FBP_diff.png")


