import numpy as np
import matplotlib.pyplot as plt
import os
#os.chdir('/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation')
import helper_funcs as hf
from argparse import ArgumentParser

# Script for creating partialy hydrated MgO phantom

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--filter", type=str, help="Filter to use in FBP recon",default="hann")
    args = parser.parse_args()
    sinogram = np.load("/dtu-compute/Mathcrete_thesis/MgO_hydrate/CHR_4h_nofilter/slice_1000.npy")
    COR_NABU = 1234.7 # for time frame 4h
    cor = -(2560//2 - COR_NABU)
    ag, ig, data = hf.CIL_setup_cement(sinogram,pixel_size=1,cor=cor)
    data_rr = hf.CIL_ring_remover(data,decNum=4,wname='db10',sigma=6)
    recfbp = hf.CIL_FBP(data_rr,ig,filter=args.filter)
    slice = hf.normalize_img(recfbp.array)
    center = (1000,1000)
    radius = 910
    h, w = slice.shape[:2]
    mask = hf.create_circular_mask(h, w,radius=radius,center=center)
    masked_img = slice.copy()
    masked_img[~mask] = 0
    slice = masked_img[(center[0]-radius):(center[0]+radius), (center[1]-radius):(center[1]+radius)]
    h, w = slice.shape[:2]
    mask = hf.create_circular_mask(h, w)
    phantom = slice.copy()

    # Manual rough segmentation
    seg0 = 0
    seg1 = 0.25
    seg2 = 0.4
    seg3 = 0.56
    seg4 = 0.68
    plt.figure()
    plt.hist(phantom[mask].reshape(-1),bins = 2000)
    plt.axvline(seg0, color='k', linestyle='dashed', linewidth=1)
    plt.axvline(seg1, color='k', linestyle='dashed', linewidth=1)
    plt.axvline(seg2, color='k', linestyle='dashed', linewidth=1)
    plt.axvline(seg3, color='k', linestyle='dashed', linewidth=1)
    plt.axvline(seg4, color='k', linestyle='dashed', linewidth=1)
    plt.show()
    plt.savefig(f'/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation/plots/phantom_creation/phantom_{args.filter}_histogram.png', dpi = 1000)

    phantom[(phantom <= seg0) & (phantom > phantom.min())] = seg0/2
    phantom[(seg0 < phantom) & (phantom <= seg1)] = (seg0+seg1)/2
    phantom[(seg1 < phantom) & (phantom <= seg2)] = (seg2+seg1)/2
    phantom[(phantom > seg2) & (phantom <= seg3)] = (seg2+seg3)/2 
    phantom[(phantom > seg3) & (phantom <= seg4)] = (seg3+seg4)/2 
    phantom[phantom > seg4] = 1 
    print(f"MB values {[seg0,(seg0+seg1)/2,(seg2+seg1)/2,(seg2+seg3)/2,(seg3+seg4)/2,1]}")
    np.save("mgo_mb_values",[seg0,(seg0+seg1)/2,(seg2+seg1)/2,(seg2+seg3)/2,(seg3+seg4)/2,1])
    hf.mshow(phantom)
    plt.savefig(f'/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation/plots/phantom_creation/phantom_{args.filter}_before_further_masking.png', dpi = 1000)

    # Background is far from constant (for paganin) so unfortunately I need to include manual masking of this
    phantom[1410:1600,:700] = np.max(phantom[1410:1600,:700])
    phantom[1150:1250,:435] = np.max(phantom[1150:1250,:435])
    phantom[1200:1250,:450] = np.max(phantom[1200:1250,:450])
    phantom[1250:1670,:560] = np.max(phantom[1250:1670,:560])
    (phantom[400:,:245]) = np.max(phantom[400:,:245])
    phantom[900:,:300]= np.max(phantom[900:,:300])
    phantom[1000:,:355] = np.max(phantom[1000:,:355])
    phantom[1350:1650,:600] = np.max(phantom[1350:1650,:600])
    phantom = hf.circle_mask(phantom)

    # Save fig and phantom 
    padded = np.pad(phantom,(370,370))
    print(f"Shape of final phantom: {padded.shape}")
    hf.mshow(padded)
    plt.savefig(f'/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation/plots/phantom_creation/phantom_{args.filter}_full.png', dpi = 1000)
    hf.mshow(padded[1130:1430,1130:1430])
    plt.savefig(f'/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation/plots/phantom_creation/phantom_{args.filter}_center.png', dpi = 1000)
    hf.mshow(padded[1200:1500,1700:2000])
    plt.savefig(f'/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation/plots/phantom_creation/phantom_{args.filter}_particle.png', dpi = 1000)
    hf.mshow(padded[1000:1500,500:1000])
    plt.savefig(f'/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation/plots/phantom_creation/phantom_{args.filter}_boundary.png', dpi = 1000)
    np.save(f"/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation/phantoms/full2560_2d_{args.filter}",padded)