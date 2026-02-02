# CIL methods
try:
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
        from cil.optimisation.algorithms import PDHG, GD, SIRT
        from cil.optimisation.functions import BlockFunction, MixedL21Norm, L2NormSquared, SmoothMixedL21Norm
        from cil.optimisation.operators import BlockOperator, GradientOperator

        from cil.plugins.ccpi_regularisation.functions import FGP_TV
except:
        print("cil not available")

# Additional packages
import numpy as np # conda install numpy
import matplotlib.pyplot as plt # conda install matplotlib
import dxchange # for readin hd5 meta data
import matplotlib.patches as patches

#helper functions
import os
import helper_funcs as hf

# Script for comparing phase retreival methods

if __name__ == "__main__":

        zoomb = [700,1200,200,700] # boundary
        zoomc = [1130,1430,1130,1430] # central
        zoomw = [450,950,1130,1630] # water
        zoomd = [1550,2050,1550,2050] # down right
        raw = np.load("/dtu-compute/Mathcrete_thesis/mathcrete_dtu/real_data/time_4h_raw_slice_1000_dist1.npy")
        just_pag = np.load("/dtu-compute/Mathcrete_thesis/mathcrete_dtu/real_data/just_pag_4h_slice1000.npy")
        just_ctf = np.load("/dtu-compute/Mathcrete_thesis/mathcrete_dtu/real_data/just_ctf_4h_slice1000.npy")
        long_ctf = np.load("/dtu-compute/Mathcrete_thesis/mathcrete_dtu/real_data/long_ctf_4h_slice1000.npy")
        ctf_pm = np.load("/dtu-compute/Mathcrete_thesis/MgO_hydrate/CHR_4h_ctf_nofilter/MgO_insitu_water_35nm_bottom_011_pm_slice_1000.npy")
        pag_pm = np.load("/dtu-compute/Mathcrete_thesis/MgO_hydrate/CHR_4h_nofilter/slice_1000.npy")
        hf.plot_standard([raw,just_pag,just_ctf],title=["Raw measurements","Paganin","CTF"],savefig=True,
                                        savepath="/dtu-compute/Mathcrete_thesis/mathcrete_dtu/real_data/plots/raw_pag_ctf_sinos.png"
                                        )
        hf.plot_compare([just_pag,pag_pm],title=["Paganin","Paganin w. AP"],savefig=True,
                                        savepath="/dtu-compute/Mathcrete_thesis/mathcrete_dtu/real_data/plots/pag_pagap_sinos.png"
                                        )
        hf.plot_compare([just_ctf,ctf_pm,long_ctf],title=["CTF","CTF w. AP","More iterations"],savefig=True,
                                        savepath="/dtu-compute/Mathcrete_thesis/mathcrete_dtu/real_data/plots/ctf_ctfap_sinos.png"
                                        )
        hf.plot_standard([pag_pm,ctf_pm],title=["Paganin w. AP","CTF w. AP"],savefig=True,
                                        savepath="/dtu-compute/Mathcrete_thesis/mathcrete_dtu/real_data/plots/ap_ctf_pag_sinos.png"
                                        )
        detector = 2500
        x1 = [detector,detector]
        y1 = [0,2505]

        angles = dxchange.read_hdf5("/dtu-compute/Mathcrete_thesis/MgO_hydrate/MgO_insitu_dry_35nm_bottom_000_result01.nx",
                                dataset="entry_1/sample/rotation_angle")
        plt.close()
        plt.clf()
        plt.plot(angles,hf.normalize_img(ctf_pm)[0:2505,detector],label="CTF")
        plt.plot(angles,hf.normalize_img(pag_pm)[0:2505,detector],label=f"Paganin",linestyle="--")
        plt.xlabel(r'$\theta$')
        plt.ylabel(r'$\phi$')
        plt.legend(loc=3)
        plt.show()
        plt.savefig("/dtu-compute/Mathcrete_thesis/mathcrete_dtu/real_data/plots/ap_ctf_pag_sinos_lineplot.png",bbox_inches='tight',dpi = 1000)
        
        print(f"Angles from nx file: {angles}")
        print(f"Data shape {ctf_pm.shape}")
        COR_NABU = 1234.7 # precalculated for 4h time frne
        cor = -(2560//2 - COR_NABU) 
        pixel_size = 1 #35e-9 #in meters
        ag = AcquisitionGeometry.create_Parallel2D(rotation_axis_position=[cor,0],units='m')  \
                .set_panel(num_pixels=(ctf_pm.shape[1]),pixel_size=(pixel_size,pixel_size))\
                .set_angles(angles=angles)
        ag_wrongcor = AcquisitionGeometry.create_Parallel2D(rotation_axis_position=[0,0],units='m')  \
                .set_panel(num_pixels=(ctf_pm.shape[1]),pixel_size=(pixel_size,pixel_size))\
                .set_angles(angles=angles)
        ag.set_labels(['angle','horizontal'])
        ag_wrongcor.set_labels(['angle','horizontal'])
        data_jpag = AcquisitionData(just_pag, geometry=ag, deep_copy=False)
        data_jpag.reorder('astra') # set order
        data_jctf = AcquisitionData(just_ctf, geometry=ag, deep_copy=False)
        data_jctf.reorder('astra') # set order
        data_ctf = AcquisitionData(ctf_pm, geometry=ag, deep_copy=False)
        data_ctf.reorder('astra') # set order
        data_ctf_wrongcor = AcquisitionData(ctf_pm, geometry=ag_wrongcor, deep_copy=False)
        data_ctf_wrongcor.reorder('astra') # set order
        data_lctf = AcquisitionData(long_ctf, geometry=ag, deep_copy=False)
        data_lctf.reorder('astra') # set order
        data_pag = AcquisitionData(pag_pm, geometry=ag, deep_copy=False)
        data_pag.reorder('astra') # set order
        ig = ag.get_ImageGeometry()
        #data_nf = hf.CIL_ring_remover(data_nf,decNum=4,wname="db10",sigma=12)
        padsize = int(data_ctf.shape[0]*0.5)
        pad_ctf = Padder.edge(pad_width={'horizontal': padsize})(data_ctf)
        pad_ctf_wrongcor = Padder.edge(pad_width={'horizontal':padsize})(data_ctf_wrongcor)
        pad_pag = Padder.edge(pad_width={'horizontal': padsize})(data_pag)
        pad_jpag = Padder.edge(pad_width={'horizontal': padsize})(data_jpag)
        pad_jctf = Padder.edge(pad_width={'horizontal': padsize})(data_jctf)
        pad_lctf = Padder.edge(pad_width={'horizontal': padsize})(data_lctf)
        fbp_wrongcor = (FBP(pad_ctf_wrongcor, ig, backend='astra',filter='ram-lak').run(verbose=0).array)
        fbp_norr = (FBP(pad_ctf, ig, backend='astra',filter='ram-lak').run(verbose=0).array)
        fbp_ctf = hf.normalize_img(fbp_norr)
        fbp_pag = hf.normalize_img(FBP(pad_pag, ig, backend='astra',filter='ram-lak').run(verbose=0).array)
        fbp_jpag = hf.normalize_img(FBP(pad_jpag, ig, backend='astra',filter='ram-lak').run(verbose=0).array)
        fbp_jctf = hf.normalize_img(FBP(pad_jctf, ig, backend='astra',filter='ram-lak').run(verbose=0).array)
        fbp_lctf = hf.normalize_img(FBP(pad_lctf, ig, backend='astra',filter='ram-lak').run(verbose=0).array)
        print(f"SNRs (Before RR), Just CTF: {hf.snr(fbp_jctf)} CTF w. AP: {hf.snr(fbp_ctf)}, More iterations: {hf.snr(fbp_lctf)}")
        print(f"SNRs (Before RR), Just Paganin: {hf.snr(fbp_jpag)} Paganin w. AP: {hf.snr(fbp_pag)}")

        ##
        hf.plot_compare([fbp_wrongcor,fbp_norr],title=["No COR correction","With COR correction"],
                        savefig=True,
                        savepath="/dtu-compute/Mathcrete_thesis/mathcrete_dtu/real_data/plots/COR_correction_compare.png")
        hf.plot_compare([fbp_jpag,fbp_pag],title=["Paganin","Paganin with AP"],
                        savefig=True,
                        savepath="/dtu-compute/Mathcrete_thesis/mathcrete_dtu/real_data/plots/pag_pagap_FBP.png")
        hf.plot_compare([fbp_jpag[zoomb[0]:zoomb[1],zoomb[2]:zoomb[3]],fbp_pag[zoomb[0]:zoomb[1],zoomb[2]:zoomb[3]]],title=["Paganin","Paganin w. AP"],
                        savefig=True,
                        savepath="/dtu-compute/Mathcrete_thesis/mathcrete_dtu/real_data/plots/pag_pagap_FBP_boundary.png")
        hf.plot_compare([fbp_jctf,fbp_ctf,fbp_lctf],title=["CTF","CTF w. AP","More iterations"],
                        savefig=True,
                        savepath="/dtu-compute/Mathcrete_thesis/mathcrete_dtu/real_data/plots/ctf_ctfap_FBP.png")
        hf.plot_compare([fbp_jctf[zoomb[0]:zoomb[1],zoomb[2]:zoomb[3]],fbp_ctf[zoomb[0]:zoomb[1],zoomb[2]:zoomb[3]]],title=["CTF","CTF with AP"],
                        savefig=True,
                        savepath="/dtu-compute/Mathcrete_thesis/mathcrete_dtu/real_data/plots/ctf_ctfap_FBP_boundary.png")
        hf.plot_compare([fbp_jctf[zoomb[0]:zoomb[1],zoomb[2]:zoomb[3]],fbp_ctf[zoomb[0]:zoomb[1],zoomb[2]:zoomb[3]],fbp_lctf[zoomb[0]:zoomb[1],zoomb[2]:zoomb[3]]],
                        title=["CTF","CTF w. AP","More iterations"],
                        savefig=True,
                        savepath="/dtu-compute/Mathcrete_thesis/mathcrete_dtu/real_data/plots/ctf_ctf15_ctf60_fbp_boundary.png")
        hf.plot_compare([fbp_jctf[zoomc[0]:zoomc[1],zoomc[2]:zoomc[3]],fbp_ctf[zoomc[0]:zoomc[1],zoomc[2]:zoomc[3]],fbp_lctf[zoomc[0]:zoomc[1],zoomc[2]:zoomc[3]]],
                        title=["CTF","CTF w. AP","More iterations"],
                        savefig=True,
                        savepath="/dtu-compute/Mathcrete_thesis/mathcrete_dtu/real_data/plots/ctf_ctf15_ctf60_fbp_center.png")
        ###
        hf.plot_compare([fbp_pag,fbp_ctf],title=["Paganin","CTF"],
                        savefig=True,
                        savepath="/dtu-compute/Mathcrete_thesis/mathcrete_dtu/real_data/plots/ap_ctf_pag_FBP.png")
        hf.plot_compare([fbp_pag[zoomb[0]:zoomb[1],zoomb[2]:zoomb[3]],fbp_ctf[zoomb[0]:zoomb[1],zoomb[2]:zoomb[3]]],title=["Paganin","CTF"],
                        savefig=True,
                        savepath="/dtu-compute/Mathcrete_thesis/mathcrete_dtu/real_data/plots/ap_ctf_pag_FBP_boundary.png")
        hf.plot_compare([fbp_pag[zoomc[0]:zoomc[1],zoomc[2]:zoomc[3]],fbp_ctf[zoomc[0]:zoomc[1],zoomc[2]:zoomc[3]]],title=["Paganin","CTF"],
                        savefig=True,
                        savepath="/dtu-compute/Mathcrete_thesis/mathcrete_dtu/real_data/plots/ap_ctf_pag_FBP_center.png")
        # With RR
        d1 = data_ctf.copy()
        padsize = int(data_ctf.shape[0]*0.5)
        ## RR plot
        drrsmall = hf.CIL_ring_remover(d1,decNum=4,wname="db10",sigma=4)
        small_pad = Padder.edge(pad_width={'horizontal': padsize})(drrsmall)
        frrmsall = (FBP(small_pad, ig, backend='astra',filter='ram-lak').run(verbose=0).array)
        data_rr_ctf = hf.CIL_ring_remover(data_ctf,decNum=4,wname="db10",sigma=12)
        drrbig = hf.CIL_ring_remover(data_ctf,decNum=4,wname="db10",sigma=24)
        big_pad = Padder.edge(pad_width={'horizontal': padsize})(drrbig)
        frrbig = (FBP(big_pad, ig, backend='astra',filter='ram-lak').run(verbose=0).array)
        
        # For pag
        data_rr_pag = hf.CIL_ring_remover(data_pag,decNum=4,wname="db10",sigma=12)
        pad_ctf = Padder.edge(pad_width={'horizontal': padsize})(data_rr_ctf)
        pad_pag = Padder.edge(pad_width={'horizontal': padsize})(data_rr_pag)
        fbp_ctf = (FBP(pad_ctf, ig, backend='astra',filter='ram-lak').run(verbose=0).array)
        np.save("/dtu-compute/Mathcrete_thesis/mathcrete_dtu/real_data/fbp_ctf_4h_z1000",fbp_ctf)

        ## RR plot
        hf.plot_compare([fbp_norr[zoomb[0]:zoomb[1],zoomb[2]:zoomb[3]],frrmsall[zoomb[0]:zoomb[1],zoomb[2]:zoomb[3]],
                         fbp_ctf[zoomb[0]:zoomb[1],zoomb[2]:zoomb[3]],frrbig[zoomb[0]:zoomb[1],zoomb[2]:zoomb[3]]],
                         title=["No RR",r'$\sigma_{rr}=4$',r'$\sigma_{rr}=12$',r'$\sigma_{rr}=24$'],
                         savefig=True,savepath="/dtu-compute/Mathcrete_thesis/mathcrete_dtu/real_data/plots/ctf_RR_compare_b.png")
        hf.plot_compare([fbp_norr[zoomc[0]:zoomc[1],zoomc[2]:zoomc[3]],frrmsall[zoomc[0]:zoomc[1],zoomc[2]:zoomc[3]],
                         fbp_ctf[zoomc[0]:zoomc[1],zoomc[2]:zoomc[3]],frrbig[zoomc[0]:zoomc[1],zoomc[2]:zoomc[3]]],
                         title=["No RR",r'$\sigma_{rr}=4$',r'$\sigma_{rr}=12$',r'$\sigma_{rr}=24$'],
                         savefig=True,savepath="/dtu-compute/Mathcrete_thesis/mathcrete_dtu/real_data/plots/ctf_RR_compare_c.png")
        
        fbp_pag = hf.normalize_img(FBP(pad_pag, ig, backend='astra',filter='ram-lak').run(verbose=0).array)
        print(f"SNRs (After RR), Paganin: {hf.snr(fbp_pag)} CTF: {hf.snr(fbp_ctf)}")
        hf.plot_compare([fbp_pag,hf.normalize_img(fbp_ctf)],title=["Paganin","CTF"],
                        savefig=True,
                        savepath="/dtu-compute/Mathcrete_thesis/mathcrete_dtu/real_data/plots/ap_ctf_pag_FBPrr.png")
        hf.plot_compare([fbp_pag[zoomb[0]:zoomb[1],zoomb[2]:zoomb[3]],hf.normalize_img(fbp_ctf[zoomb[0]:zoomb[1],zoomb[2]:zoomb[3]])],
                        title=["Paganin","CTF"],
                        savefig=True,
                        savepath="/dtu-compute/Mathcrete_thesis/mathcrete_dtu/real_data/plots/ap_ctf_pag_FBPrr_boundary.png")
        hf.plot_compare([fbp_pag[zoomc[0]:zoomc[1],zoomc[2]:zoomc[3]],hf.normalize_img(fbp_ctf[zoomc[0]:zoomc[1],zoomc[2]:zoomc[3]])],title=["Paganin","CTF"],
                        savefig=True,
                        savepath="/dtu-compute/Mathcrete_thesis/mathcrete_dtu/real_data/plots/ap_ctf_pag_FBPrr_center.png")
        
        # ROIs with ctf
        fb = fbp_norr[zoomb[0]:zoomb[1],zoomb[2]:zoomb[3]]
        fc = fbp_norr[zoomc[0]:zoomc[1],zoomc[2]:zoomc[3]]
        fw = fbp_norr[zoomw[0]:zoomw[1],zoomw[2]:zoomw[3]]
        fd = fbp_norr[zoomd[0]:zoomd[1],zoomd[2]:zoomd[3]]
        hf.plot_compare([fbp_norr,fb,fc],title=["","",""],savefig=True,
                        savepath="/dtu-compute/Mathcrete_thesis/mathcrete_dtu/real_data/plots/ctf_rois1.png")
        hf.plot_compare([fbp_norr,fw,fd],title=["","",""],savefig=True,
                        savepath="/dtu-compute/Mathcrete_thesis/mathcrete_dtu/real_data/plots/ctf_rois2.png")
        
        plt.close()
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(fbp_norr,cmap="gray")
        ax.set_axis_off()
        rect = patches.Rectangle((zoomb[2], zoomb[0]), 
                                zoomb[1]-zoomb[0],
                                zoomb[3]-zoomb[2],
                                linewidth=1,
                                edgecolor='r', facecolor="none")
        ax.add_patch(rect)
        rect2 = patches.Rectangle((zoomc[0], zoomc[2]), 
                                zoomc[1]-zoomc[0],
                                zoomc[3]-zoomc[2],
                                linewidth=1,
                                edgecolor='b', facecolor="none")
        ax.add_patch(rect2)
        rect3 = patches.Rectangle((zoomw[2], zoomw[0]), 
                                zoomw[1]-zoomw[0],
                                zoomw[3]-zoomw[2],
                                linewidth=1,
                                edgecolor='g', facecolor="none")
        ax.add_patch(rect3)
        rect4 = patches.Rectangle((zoomd[0], zoomd[2]), 
                                zoomd[1]-zoomd[0],
                                zoomd[3]-zoomd[2],
                                linewidth=1,
                                edgecolor='orange', facecolor="none")
        ax.add_patch(rect4)
        plt.show()
        plt.savefig("/dtu-compute/Mathcrete_thesis/mathcrete_dtu/real_data/plots/CTF_wSQUARES.png", bbox_inches='tight',dpi = 1000)