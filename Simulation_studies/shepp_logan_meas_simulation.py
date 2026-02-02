import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from cil.framework import AcquisitionGeometry, AcquisitionData, ImageData, ImageGeometry
from cil.plugins.astra import ProjectionOperator
from cil.recon import FBP
from scipy.ndimage import gaussian_filter
#os.chdir('/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation')
import helper_funcs as hf
from argparse import ArgumentParser

# Script to create half-smoothed Shepp-Logan phantom and simulate measurements

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--sigma",type=int, help="Level of smoothing in phantoms",default="10")
    args = parser.parse_args()

    # Smoothed shepp logan
    gt_ns = np.load("phantoms/shepp_logan_2560.npy")
    n = gt_ns.shape[0]
    smoothed = gaussian_filter(gt_ns[:(n//2),:],sigma=args.sigma)
    gt = gt_ns.copy()
    gt[:(n//2),:] = smoothed
    zooms = [1130,1430]
    hf.plot_compare([gt_ns,gt],title=["Standard Shepp Logan","Smoothed Shepp Logan"],savefig=True,
                    savepath="plots/new_shepp_logan_smoothed/phantoms.png")
    hf.plot_compare([gt_ns[zooms[0]:zooms[1],zooms[0]:zooms[1]],gt[zooms[0]:zooms[1],zooms[0]:zooms[1]]],
                    title=["Standard Shepp Logan","Smoothed Shepp Logan"],
                    savefig=True, savepath="plots/new_shepp_logan_smoothed/phantoms_zoomed.png")
    np.save("phantoms/new_shepp_logan_smoothed.npy",gt)

    ## ROI plot
    plt.close()
    plt.figure()
    fig, axs = plt.subplots(nrows=1, ncols=n, figsize=(8, 8))
    im = axs[0].imshow(gt,cmap="gray")
    axs[0].set_axis_off()
    rect = patches.Rectangle((zooms[0], zooms[0]), 
                             zooms[1]-zooms[0],
                             zooms[1]-zooms[0],
                              linewidth=1,
                            edgecolor='r', facecolor="none")
    axs[0].add_patch(rect)
    gt_z = gt[zooms[0]:zooms[1],zooms[0]:zooms[1]]
    axs[1].imshow(gt_z,cmap="gray")
    axs[1].set_axis_off()
    fig.colorbar(im, ax=axs[1], orientation='vertical')
    rect2 = patches.Rectangle((0, 0), 
                             gt_z.shape[0],
                             gt_z.shape[0]-1,
                              linewidth=10,
                            edgecolor='r', facecolor="none")
    axs[1].add_patch(rect2)
    plt.show()
    plt.savefig("plots/new_shepp_logan_smoothed/gt_center_Red.png", bbox_inches='tight',dpi = 1000)

    ###
    pixel_size = 2/n
    print(f"Using pixel size {pixel_size}")
    ig = ImageGeometry(voxel_num_x=n, 
                    voxel_num_y=n,
                    voxel_size_x=pixel_size,
                    voxel_size_y=pixel_size)
    gt_data = ImageData(gt,geometry=ig)

    #Ideal sinogram
    angles = np.linspace(0, 180, n, endpoint=False)
    ag = AcquisitionGeometry.create_Parallel2D(rotation_axis_position=[0, 0])  \
        .set_panel(num_pixels=(n),pixel_size=(pixel_size,pixel_size))        \
        .set_angles(angles=angles) 
    ag.set_labels(['angle','horizontal'])
    A = ProjectionOperator(ig, ag, "gpu")
    sino_ideal = (A.direct(gt_data)).array
    hf.plot_sino_gt([gt,sino_ideal],title=["",""],angles=angles,savefig=True,
                     savepath="plots/new_shepp_logan_smoothed/gt_ideal_sino.png")

    noise_levels = [0.001,0.01,0.05,0.1]
    num_angles = [2560,1280,320,40]
    
    rec_noise = []
    rec_noise_zoom = []
    title_noise = []
    sino_noise = []
    rec_angles = []
    rec_angles_zoom = []
    title_angles = []
    sino_angles = []

    rrmse_angles = []
    sssim_angles = []
    rrmse_noise = []
    sssim_angles = []

    for na in num_angles:
        angles = np.linspace(0, 180, na, endpoint=False)
        ag = AcquisitionGeometry.create_Parallel2D(rotation_axis_position=[0, 0])  \
                .set_panel(num_pixels=(n),pixel_size=(pixel_size,pixel_size))        \
                .set_angles(angles=angles) 
        ag.set_labels(['angle','horizontal'])
        A = ProjectionOperator(ig, ag, "gpu")
        sinogram = A.direct(gt_data)
        sinogram.reorder('astra')
        noise = np.random.normal(0,1,size=(sinogram.array.shape))
        noise = noise*np.linalg.norm(sinogram.array)/np.linalg.norm(noise)*noise_levels[1]
        print(f"Relative norm of noise: {np.linalg.norm(noise)/np.linalg.norm(sinogram.array)}")
        sino_out = sinogram.array + noise
        sino_n = ag.allocate()
        sino_n.fill(sino_out)
        delta = np.linalg.norm(A.direct(ImageData(gt,geometry=ig)).array-sino_out)
        print(f"delta = {delta}")
        print(f"Norm of clean data = {np.linalg.norm(sinogram.array)}")
        print(f"delta/norm(data): {delta/np.linalg.norm(sinogram.array)}")
        sino_angles.append(sino_n.array)
        title_angles.append(f"No. angles: {na}")
        recfbp = FBP(sino_n, ig, backend='astra',filter='ram-lak').run(verbose=0)
        rec_angles.append(recfbp.array)
        np.save(f"fbp_saves/fbp_nl_{2}_na_{na}",recfbp.array)
        rec_angles_zoom.append(recfbp.array[zooms[0]:zooms[1],zooms[0]:zooms[1]])
        #show2D(sinogram_noisy.array[:,0:100])
        np.save(f"Simulated_sinogram/new_shepp_logan_smoothed/sino_nl_{2}_na_{na}",sino_n.array)
        if na == num_angles[1]: # Vary noise
            for level, nl in enumerate(noise_levels):
                noise = np.random.normal(0,1,size=(sinogram.array.shape))
                noise = noise*np.linalg.norm(sinogram.array)/np.linalg.norm(noise)*nl
                print(f"Relative norm of noise: {np.linalg.norm(noise)/np.linalg.norm(sinogram.array)}")
                sino_out = sinogram.array + noise
                sinogram_noisy = ag.allocate()
                sinogram_noisy.fill(sino_out)
                delta = np.linalg.norm(A.direct(ImageData(gt,geometry=ig)).array-sino_out)
                print(f"delta = {delta}")
                print(f"Norm of clean data = {np.linalg.norm(sinogram.array)}")
                print(f"delta/norm(data): {delta/np.linalg.norm(sinogram.array)}")
                sino_noise.append(sinogram_noisy.array)
                title_noise.append(f"Noise level: {nl}")
                np.save(f"Simulated_sinogram/new_shepp_logan_smoothed/sino_nl_{level+1}_na_{na}",sinogram_noisy.array)
                recfbp = FBP(sinogram_noisy, ig, backend='astra',filter='ram-lak').run(verbose=0)
                rec_noise.append(recfbp.array)
                np.save(f"fbp_saves/fbp_nl_{level+1}_na_{na}",recfbp.array)
                rec_noise_zoom.append(recfbp.array[zooms[0]:zooms[1],zooms[0]:zooms[1]])

    hf.plot_compare(rec_noise,title=title_noise,savefig=True,
                savepath="plots/new_shepp_logan_smoothed/noisy_fbps.png")
    hf.plot_compare(rec_noise_zoom,title=title_noise,savefig=True,
                savepath="plots/new_shepp_logan_smoothed/noisy_fbps_zoom.png")
    hf.plot_compare(rec_angles,title=title_angles,savefig=True,
                savepath="plots/new_shepp_logan_smoothed/angle_fbps.png")
    hf.plot_compare(rec_angles_zoom,title=title_angles,savefig=True,
                savepath="plots/new_shepp_logan_smoothed/angles_fbps_zoom.png")
