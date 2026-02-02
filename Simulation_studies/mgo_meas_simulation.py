import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from cil.framework import AcquisitionGeometry, AcquisitionData, ImageData, ImageGeometry
from cil.plugins.astra import ProjectionOperator
from cil.recon import FBP
#os.chdir('/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation')
import helper_funcs as hf
from argparse import ArgumentParser
import dxchange # for readin hd5 meta data

# Script for Mgo Hydration phantom measurement creation

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--noise_level",type=float,default="0.01")
    args = parser.parse_args()

    # Hydrated mgo phantom
    gt = np.load("phantoms/full2560_2d_hann.npy")
    ############### ROI ####################
    phantom_shifted = np.pad(gt[350:-1,350:-1],((500,1500),(500,1500)))
    # plt.imshow(phantom_shifted,cmap="gray")
    # plt.show()
    n = phantom_shifted.shape[0]
    new_range = int((n-2560)//2)
    gt_roi = phantom_shifted[new_range:new_range+2560,new_range:new_range+2560]
    plt.close()
    plt.clf()
    plt.figure()
    plt.imshow(gt_roi,cmap="gray")
    plt.show()
    plt.savefig("plots/mgo/gt_roi.png",
                  bbox_inches='tight',dpi = 1000)

    ########################################
    n = gt.shape[0]
    pixel_size = 2/n
    ig = ImageGeometry(voxel_num_x=n, 
                    voxel_num_y=n,
                    voxel_size_x=pixel_size,
                    voxel_size_y=pixel_size)
    gt_data = ImageData(gt,geometry=ig)
    gt_roi_data = ImageData(gt_roi,geometry=ig)

    # plot data and regions
    zoom_p = [1700,2000,1200,1500] # particle
    zoom_b = [500,800,1000,1300] # boundary
    plt.close()
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(gt,cmap="gray")
    ax.set_axis_off()
    rect = patches.Rectangle((zoom_p[0], zoom_p[2]), 
                             zoom_p[1]-zoom_p[0],
                             zoom_p[3]-zoom_p[2],
                              linewidth=1,
                            edgecolor='r', facecolor="none")
    ax.add_patch(rect)
    rect2 = patches.Rectangle((zoom_b[0], zoom_b[2]), 
                             zoom_b[1]-zoom_b[0],
                             zoom_b[3]-zoom_b[2],
                              linewidth=1,
                            edgecolor='b', facecolor="none")
    ax.add_patch(rect2)
    plt.show()
    plt.savefig("plots/mgo/gt_full_wroi.png", bbox_inches='tight',dpi = 1000)
    #plt.colorbar()
    plt.close()
    plt.figure()
    fig, ax = plt.subplots(1)
    gt_p = gt[zoom_p[2]:zoom_p[3],zoom_p[0]:zoom_p[1]]
    gt_b = gt[zoom_b[2]:zoom_b[3],zoom_b[0]:zoom_b[1]]
    im = ax.imshow(gt_p,cmap="gray")
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, orientation='vertical')
    rect = patches.Rectangle((0, 0), 
                             gt_p.shape[0],
                             gt_p.shape[0]-1,
                              linewidth=10,
                            edgecolor='r', facecolor="none")
    ax.add_patch(rect)
    plt.show()
    plt.savefig("plots/mgo/gt_particle.png", bbox_inches='tight',dpi = 1000)
    plt.close()
    plt.figure()
    fig, ax = plt.subplots(1)
    im = ax.imshow(gt_b,cmap="gray")
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, orientation='vertical')
    rect = patches.Rectangle((0, 0), 
                             gt_p.shape[0],
                             gt_p.shape[0],
                              linewidth=10,
                            edgecolor='b', facecolor="none")
    ax.add_patch(rect)
    plt.show()
    plt.savefig("plots/mgo/gt_boundary.png", bbox_inches='tight',dpi = 1000)
    ##########################################################################3
    #Ideal sinograms
    angles = dxchange.read_hdf5("/dtu-compute/Mathcrete_thesis/MgO_hydrate/MgO_insitu_dry_35nm_bottom_000_result01.nx",
                            dataset="entry_1/sample/rotation_angle") # Use same angles as experiment
    ag = AcquisitionGeometry.create_Parallel2D(rotation_axis_position=[0, 0])  \
        .set_panel(num_pixels=(n),pixel_size=(pixel_size,pixel_size))        \
        .set_angles(angles=angles) 
    ag.set_labels(['angle','horizontal'])
    A = ProjectionOperator(ig, ag, "gpu")
    sino_ideal = (A.direct(gt_data))
    #sino_ideal_roi = (A.direct(gt_roi_data))
    sino_trunc = np.load("Simulated_sinogram/mgo/sinogram_clean_ROI.npy")
    sino_ideal_roi = ag.allocate()
    sino_ideal_roi.fill(sino_trunc)
    hf.plot_sino_gt([gt,sino_ideal.array],angles=angles,savefig=True,
                     savepath="plots/mgo/gt_ideal_sino.png",title=["",""])
    
    hf.plot_sino_gt([gt_roi,sino_ideal_roi.array],angles=angles,savefig=True,
                     savepath="plots/mgo/gt_ideal_sino_roi.png",title=["",""])

    print(f"2-norm of sino: {np.linalg.norm(sino_ideal.array)}")
    print(f"Inf-norm of sino: {np.max(abs(sino_ideal.array))}")
    np.random.seed(1)
    noise_level = args.noise_level
    noise = np.random.normal(0,1,size=((sino_ideal.array).shape))
    # Want 1 procent
    noise = noise*np.linalg.norm(sino_ideal.array)/np.linalg.norm(noise)*noise_level
    print(f"Relative norm of noise: {np.linalg.norm(noise)/np.linalg.norm(sino_ideal.array)}")
    noise2 = np.random.normal(0,1,size=((sino_ideal_roi.array).shape))
    # Want 1 procent
    noise2 = noise2*np.linalg.norm(sino_ideal_roi.array)/np.linalg.norm(noise2)*noise_level
    print(f"Relative norm of noise (ROI): {np.linalg.norm(noise2)/np.linalg.norm(sino_ideal_roi.array)}")


    stripe_out, field = hf.add_blurred_stripes(
        sino_ideal.array,
        num_stripes=150,
        axis='vertical',
        width_range=(1, 10),
        amplitude_range=np.array([-0.05, 0.05]),
        variation_scale=10,
        global_scale=0.3,
        blur_sigma=(1.5, 1.0),   # more blur along angles
        mode='additive',
        seed=111
    )
    stripe_out_roi, field = hf.add_blurred_stripes(
        sino_ideal_roi.array,
        num_stripes=150,
        axis='vertical',
        width_range=(1, 10),
        amplitude_range=np.array([-0.05, 0.05]),
        variation_scale=10,
        global_scale=0.3,
        blur_sigma=(1.5, 1.0),   # more blur along angles
        mode='additive',
        seed=111
    )

    sino_sn = stripe_out + noise
    sino_sn_roi = stripe_out_roi + noise2
    # plot sino
    plt.close()
    plt.clf()
    plt.figure()
    plt.imshow(sino_sn[1000:2000,1000:2000], cmap='gray')
    yticks = np.linspace(0,1000,4)
    xticks = np.linspace(0,1000,4)
    yticklabels = ["{:6.2f}".format(((angles[1000:2000])[int(i)]/180)*(np.pi)) for i in np.linspace(0,1000-1,4)]
    xticklabels = [1000,1333,1666,2000]#np.linspace(1000,2000,4)
    plt.yticks(yticks,yticklabels)
    plt.ylabel(r'$\theta$')
    plt.xticks(xticks,xticklabels)
    plt.show()
    plt.savefig("plots/mgo/sino_noise_stripe.png",
                  bbox_inches='tight',dpi = 1000)

    ## measurements
    sinogram_sn = ag.allocate()
    sinogram_sn.fill(sino_sn)
    sinogram_sn.reorder("astra")
    sinogram_sn_roi = ag.allocate()
    sinogram_sn_roi.fill(sino_sn_roi)
    sinogram_sn_roi.reorder("astra")
    delta = np.linalg.norm(sinogram_sn.array-sino_ideal.array) # for morozov
    print(f"delta = {delta}")
    delta_roi = np.linalg.norm(sinogram_sn_roi.array-sino_ideal_roi.array) # for morozov
    print(f"delta (ROI) = {delta_roi}")

    # Ring remover and FBP
    data_rr = hf.CIL_ring_remover(sinogram_sn,4,'db10',3)
    recfbp_sn_rr = hf.CIL_FBP(data_rr,ig).array
    data_rr_roi = hf.CIL_ring_remover(sinogram_sn_roi,4,'db10',3)
    recfbp_sn_rr_roi = hf.CIL_FBP(data_rr_roi,ig).array
    if noise_level == 0.01:
        np.save(f"/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation/Simulated_sinogram/mgo/sinogram_stripes_noise1",sinogram_sn.array)
        np.save(f"/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation/mgo_fbp_noise1",recfbp_sn_rr)
        np.save(f"/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation/Simulated_sinogram/mgo/sinogram_stripes_noise1_ROI",sinogram_sn_roi.array)
        np.save(f"/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation/mgo_fbp_noise1_ROI",recfbp_sn_rr_roi)
        hf.plot_compare([gt,recfbp_sn_rr],title=["Ground truth","FBP with rr"],savefig=True,savepath="plots/mgo/fbp_recon_noise1.png")
        hf.plot_compare([gt[zoom_b[2]:zoom_b[3],zoom_b[0]:zoom_b[1]],recfbp_sn_rr[zoom_b[2]:zoom_b[3],zoom_b[0]:zoom_b[1]]],title=["Ground truth","FBP with rr"],savefig=True,savepath="plots/mgo/bzoom_fbp_recon_noise1.png")
        hf.plot_compare([gt[zoom_p[2]:zoom_p[3],zoom_p[0]:zoom_p[1]],recfbp_sn_rr[zoom_p[2]:zoom_p[3],zoom_p[0]:zoom_p[1]]],title=["Ground truth","FBP with rr"],savefig=True,savepath="plots/mgo/pzoom_fbp_recon_noise1.png")
        # ROI
        hf.plot_compare([gt_roi,recfbp_sn_rr_roi],title=["Ground truth","FBP with rr"],savefig=True,savepath="plots/mgo/fbp_recon_noise1_ROI.png")
        hf.plot_standard([gt_roi,recfbp_sn_rr_roi],title=["Ground truth","FBP with rr"],savefig=True,savepath="plots/mgo/fbp_recon_noise1_ROI_diffcolorbar.png")

    
    