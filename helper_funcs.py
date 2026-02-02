import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from cil.framework import AcquisitionGeometry, AcquisitionData, ImageGeometry
from cil.processors import Padder, RingRemover
from cil.recon import FBP
from cil.optimisation.functions import Function
from cil.optimisation.operators import LinearOperator
import dxchange # for readin hd5 meta data

## helper functions and classes for 

##################### Misc ##############################

def morozov_check(sol,operator,ymeas,delta,tol=1e-6):
    # To check MDP parameter choice rule
    res = np.linalg.norm(operator.direct(sol).array-ymeas)
    check = abs(res - delta) < tol
    return res, check

def create_circular_mask(h, w, center=None, radius=None):
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def circle_mask(slice,center=None, radius=None):
    h, w = slice.shape[:2]
    mask = create_circular_mask(h, w,radius=radius,center=center)
    masked_img = slice.copy()
    masked_img[~mask] = 0
    return masked_img

##################### Plotting ##############################

def mirror_flip(array):
    array = np.flip(array)
    array = np.flip(array,axis=1)
    return array

def normalize_img(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def snr(f,noise_mask=None):
    f = normalize_img(f)
    if not noise_mask:
        noise_mask = create_circular_mask(2560,2560,center=(1000,1050
                                                ),radius=1350) # manually created from 4 hour scan
    mu = np.mean(f[noise_mask==0])
    std = np.std(f[noise_mask==1])
    return mu /std

def mshow(a):
    fig, axs = plt.subplots(1, 1, figsize=(8, 8))
    im = axs.imshow(a, cmap='gray')
    fig.colorbar(im)
    plt.show()

def plot_standard(a,title,savefig=False,savepath=None,size=0):
    plt.close()
    plt.clf()
    n = len(a)
    if size > 0:
        fig_size = size
    else:
        fig_sizes = [4,8,16,16]
        fig_size = fig_sizes[n-1]
    shrinks = [0.7,0.31,0.2,0.15]
    
    fig, axs = plt.subplots(nrows=1, ncols=n, figsize=(fig_size, fig_size))
    for i in range(n):
        temp = axs[i].imshow(a[i], cmap='gray')
        axs[i].set_title(title[i])
        axs[i].axis("off")
        fig.colorbar(temp, ax=axs[i], shrink=shrinks[n-1])

    if savefig and savepath:
        plt.savefig(savepath, bbox_inches='tight',dpi = 1000)
    plt.show()

def plot_compare(a,title,savefig=False,savepath=None,use_range=0):
    plt.close()
    plt.clf()
    n = len(a)
    if use_range == -1:
        use_range = n-1
    fig_sizes = [4,8,16,16]
    fig, axs = plt.subplots(nrows=1, ncols=n, figsize=(fig_sizes[n-1], fig_sizes[n-1]))
    vmin, vmax = a[use_range].min(), a[use_range].max()
    for i in range(n):
        if n == 1:
            temp = axs.imshow(a[i], cmap='gray', vmin = vmin, vmax = vmax)
            axs.set_title(title[i])
            axs.axis("off")
            break
        if i == use_range:
            temp = axs[i].imshow(a[i], cmap='gray', vmin = vmin, vmax = vmax)
        else:
            axs[i].imshow(a[i], cmap='gray', vmin = vmin, vmax = vmax)
        axs[i].set_title(title[i])
        axs[i].axis("off")
    fig.subplots_adjust(right=0.8)

    shrinks = [0.7,0.31,0.2,0.15]
    fig.colorbar(temp, ax=axs, shrink=shrinks[n-1])
    if savefig and savepath:
        plt.savefig(savepath, bbox_inches='tight',dpi = 1000)
    plt.show()

def plot_compare_diff(a,title,savefig=False,savepath=None,use_range=0):
    plt.close()
    plt.clf()
    n = len(a)
    fig_sizes = [4,8,16,16]
    fig, axs = plt.subplots(nrows=1, ncols=n, figsize=(fig_sizes[n-1], fig_sizes[n-1]))
    vmin, vmax = a[use_range].min(), a[use_range].max()
    for i in range(n):
        if i == n-1: 
            temp = axs[i].imshow(a[i], cmap='gray') 
        else:
            axs[i].imshow(a[i], cmap='gray', vmin = vmin, vmax = vmax)
        axs[i].set_title(title[i])
        axs[i].axis("off")
    fig.subplots_adjust(right=0.8)

    shrinks = [0.7,0.31,0.2,0.2]
    fig.colorbar(temp, ax=axs, shrink=shrinks[n-1])
    if savefig and savepath:
        plt.savefig(savepath, bbox_inches='tight',dpi = 1000)
    plt.show()

def plot_compare_mrows(a,title,savefig=False,savepath=None,use_range=0,rows=4):
    plt.close()
    plt.clf()
    n = len(a) // rows
    print(f"columns = {n}")
    if use_range == -1:
        use_range = n-1
    fig_sizes = [4,8,6,4]
    shrinks = [0.7,0.31,0.2,0.6]
    print(f"Using shrink {shrinks[n]}")
    fig, axs = plt.subplots(nrows=rows, ncols=n, figsize=(fig_sizes[n-1], fig_sizes[n-1]))
    fig.subplots_adjust(hspace=0)
    vmin, vmax = a[use_range].min(), a[use_range].max()
    for r in range(rows):
        
        for i in range(n):
            ax = axs[r, i]
            last_img = ax.imshow(a[r*n+i], cmap='gray', vmin=vmin, vmax=vmax)
            ax.set_title(title[r*n+i])  # titles only on first row
            ax.axis("off")

    fig.subplots_adjust(right=0.88) # make room for colorbar
    cbar_ax = fig.add_axes([0.90, 0.12, 0.015, 0.75]) # [left, bottom, width, height] 
    fig.colorbar(last_img, cax=cbar_ax,shrink=shrinks[n])
    if savefig and savepath:
        plt.savefig(savepath, bbox_inches='tight', dpi=1000)

    plt.show()


def plot_compare_big(a,title,rows,row_titles=None,savefig=False,savepath=None,use_range=0):
    plt.close()
    plt.clf()
    n = len(a) // rows
    print(f"columns = {n}")
    if use_range == -1:
        use_range = n-1
    fig_sizes = [4,8,16,18]
    shrinks = [0.7,0.31,0.2,0.15]
    fig, axs = plt.subplots(nrows=rows, ncols=n, figsize=(fig_sizes[n-1], fig_sizes[n-1]))
    if rows < 5:
        fig.subplots_adjust(hspace=-0.1)
    else:
        print("adjusting vspace")
        fig.subplots_adjust(hspace=0.05)
        fig.subplots_adjust(wspace=-0.1)
    vmin, vmax = a[use_range].min(), a[use_range].max()
    if rows == 3: 
        row_indices = [0.75,0.5,0.25]
    if rows == 4: 
        row_indices = [0.775,0.6,0.4,0.225]
    if rows == 5:
        row_indices = [0.8,0.65,0.5,0.35,0.2]
    for r in range(rows):
        
        for i in range(n):

            ax = axs[r, i]
            last_img = ax.imshow(a[r*n+i], cmap='gray', vmin=vmin, vmax=vmax)
            ax.set_title(title[i] if r == 0 else "")  # titles only on first row
            ax.axis("off")
        if row_titles is not None:
            if n == 3:
                fig.text( 0.1, # x-position (left margin) 
                 row_indices[r], # y-position (center of row) 
                 row_titles[r], va='center', ha='center', rotation='vertical', fontsize=22)
            else:
                fig.text( 0.125, # x-position (left margin) 
                 row_indices[r], # y-position (center of row) 
                 row_titles[r], va='center', ha='center', rotation='vertical', fontsize=22)

    fig.subplots_adjust(right=0.88) # make room for colorbar 
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])#[0.90, 0.1, 0.02, 0.8]) # [left, bottom, width, height] 
    fig.colorbar(last_img, cax=cbar_ax)
    if savefig and savepath:
        plt.savefig(savepath, bbox_inches='tight', dpi=1000)

    plt.show()

def plot_sino_gt(a,title,angles,savefig=False,savepath=None,size=0):
    plt.close()
    plt.clf()
    n = len(a)
    if size > 0:
        fig_size = size
    else:
        fig_sizes = [4,8,16,16]
        fig_size = fig_sizes[n-1]
    shrinks = [0.7,0.31,0.2,0.15]
    ticks = np.linspace(0,2506,4)
    yticklabels = ["{:6.2f}".format((angles[int(i)]/180)*(np.pi)) for i in np.linspace(0,len(angles)-1,4)]
    #xticklabels = []"{:6.2f}".format(angles(i)) for i in range(4)]
    fig, axs = plt.subplots(nrows=1, ncols=n, figsize=(fig_size, fig_size))

    axs[0].imshow(a[0], cmap='gray')
    axs[0].set_title("Ground truth")
    #axs[1].set_yticks([])
    axs[1].imshow(a[1], cmap='gray')
    #axs[i].set_title(title[i])
    axs[1].set_yticks(ticks)
    axs[1].set_yticklabels(yticklabels)
    axs[1].yaxis.set_label_position("right")
    axs[1].yaxis.tick_right()
    axs[1].set_ylabel(r'$\theta$')
    #axs[1].set_xlabel("Horizontal")
    axs[1].set_title("Clean sinogram")
    if savefig and savepath:
        plt.savefig(savepath, bbox_inches='tight',dpi = 1000)
    plt.show()

def plot_roi(a,x,y,wx,wy,mirror_flip=False):
    fig, ax = plt.subplots(1)
    if mirror_flip:
        a = mirror_flip(a)
    ax.imshow(a,cmap="gray")
    rect = patches.Rectangle((x, y), wx, wy, linewidth=1,
                            edgecolor='r', facecolor="none")

    ax.add_patch(rect)
    plt.show()

def plot_hline(a,xs,y):
    plt.plot(xs,[y,y],color="blue",linewidth=3)
    plt.imshow(a,cmap="gray")
    plt.show()

    plt.plot(a[y,xs[0]:xs[1]])
    #plt.legend(loc=1)
    plt.show()

##################### CIL setup for Mgo hydration data ##############################

def CIL_setup_cement(sinogram,factor_multiply=False,cor=0,pixel_size=1,verbose=False,angle_range=360):
    n = sinogram.shape[1] 
    nrj_ev = 29.60*1000
    wavelength=12398.4e-10/nrj_ev
    pixel_size = pixel_size
    factor =  wavelength/(2*np.pi)*1/pixel_size 
    if factor_multiply:
        sinogram = sinogram*factor # potentially more realistic values closer to the refraction index
    
    #Setup geometries
    ig = ImageGeometry(voxel_num_x=n, 
                    voxel_num_y=n,
                    voxel_size_x=pixel_size,
                    voxel_size_y=pixel_size)
    if angle_range == 360:
        angles = dxchange.read_hdf5("/dtu-compute/Mathcrete_thesis/MgO_hydrate/MgO_insitu_dry_35nm_bottom_000_result01.nx",
                                    dataset="entry_1/sample/rotation_angle") # Use same angles as experiment
    elif angle_range == 180:
        angles = np.linspace(0, 180, sinogram.shape[0], endpoint=False)

    if verbose:
        print(f"Number of pixels: {n}")
        print(f"Number of angles: {len(angles)}")
    ag = AcquisitionGeometry.create_Parallel2D(rotation_axis_position=[cor, 0],units="m")  \
         .set_panel(num_pixels=(n),pixel_size=(pixel_size,pixel_size))        \
         .set_angles(angles=angles) 
    ag.set_labels(['angle','horizontal'])
    data_acq = AcquisitionData(sinogram, geometry=ag, deep_copy=False)
    data_acq.reorder('astra')
    return ag, ig, data_acq 

def CIL_FBP(data,ig,padding=True,pad_fraction=0.5,filter="ram-lak"):
    if padding:
        padsize = int(data.shape[0]*pad_fraction)
        data = Padder.edge(pad_width={'horizontal': padsize})(data)
    recfbp = FBP(data, ig, backend='astra',filter=filter).run(verbose=0)
    return recfbp

def CIL_ring_remover(data,decNum=4,wname="db10",sigma=4):
    rr_proc = RingRemover(decNum=decNum,  wname=wname,   sigma=sigma,  info=True)
    rr_proc.set_input(data)
    return rr_proc.get_output()

##################### Metrics ##############################
from skimage.metrics import structural_similarity

def multibang_metric(gt, rec, mb_values):
    #rec = rec.flatten()
    idx = np.abs(rec[..., None] - mb_values).argmin(axis=-1) # Map each element to its closest allowed value result = allowed[idx]
    result = mb_values[idx]
    return np.sum(abs(gt-result)<1e-2)/(gt.shape[0]**2), result

def compute_metrics(gt,rec,get_values=False,print_out=True,list=False,mb_values=[0]):
    #mse = mean_squared_error(rec, gt)
    if len(mb_values) > 1:
        if list:
            rrmse = []
            ssim = []
            mb_score = []
            for i in range(len(rec)):
                rrmse.append((np.mean( (rec[i] - gt)**2)/np.mean(gt**2))**(1/2))
                ssim.append(structural_similarity(rec[i], gt, data_range=(rec[i]).max() - (rec[i]).min()))
                score, _ = (multibang_metric(gt,rec[i],mb_values))
                mb_score.append(score)
        else:
            rrmse = (np.mean( (rec - gt)**2)/np.mean(gt**2))**(1/2)
            ssim = structural_similarity(rec, gt, data_range=rec.max() - rec.min())
            mb_score, _ = multibang_metric(gt,rec,mb_values)
        if print_out:
            print(f"RRMSE: {rrmse}")
            print(f"SSIM: {ssim}")
            print(f"MB score: {mb_score}")
        if get_values:
            return rrmse, ssim, mb_score
    else:
        if list:
            rrmse = []
            ssim = []
            for i in range(len(rec)):
                rrmse.append((np.mean( (rec[i] - gt)**2)/np.mean(gt**2))**(1/2))
                ssim.append(structural_similarity(rec[i], gt, data_range=(gt).max() - (gt).min()))
        else:
            rrmse = (np.mean( (rec - gt)**2)/np.mean(gt**2))**(1/2)
            ssim = structural_similarity(rec, gt, data_range=(gt).max() - (gt).min())
        if print_out:
            print(f"RRMSE: {rrmse}")
            print(f"SSIM: {ssim}")
        if get_values:
            return rrmse, ssim

################## Stripe simulation #########################################

from scipy.ndimage import gaussian_filter

def add_blurred_stripes(
    img,
    num_stripes=8,
    axis='vertical',
    width_range=(2, 12),            # in pixels along the stripe direction (detector bins)
    amplitude_range=(-0.08, 0.12),  # contrast; small values look realistic
    variation_scale=45.0,           # smoothness of amplitude variation across angles (rows)
    global_scale=1.0,               # scales overall artifact strength
    blur_sigma=(3.0, 1.0),          # (sigma_rows, sigma_cols) for anisotropic blur
    mode='additive',                # 'additive' or 'multiplicative'
    seed=None
):
    """
    Add blurred, non-constant stripe artifacts to a sinogram-like image.
    
    Parameters
    ----------
    img : 2D ndarray
        Input sinogram (rows: projection angles, cols: detector bins).
    num_stripes : int
        Number of stripes to inject.
    axis : str
        'vertical' (typical for sinograms) or 'horizontal'.
    width_range : tuple(float, float)
        Min/max Gaussian stripe width in pixels along detector axis.
    amplitude_range : tuple(float, float)
        Min/max stripe amplitude (contrast). Use small magnitudes for realism.
    variation_scale : float
        Controls how smoothly stripe amplitude varies across the orthogonal axis
        (larger => smoother).
    global_scale : float
        Global scaling of the stripe field.
    blur_sigma : tuple(float, float)
        Gaussian blur sigmas (rows, cols). Increase first value to smear along angles.
    mode : str
        'additive' (img + field) or 'multiplicative' (img * (1 + field)).
    seed : int or None
        RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    # Ensure we operate on float copy
    img = np.asarray(img, dtype=np.float32)
    h, w = img.shape

    # Stripe orientation
    vertical = (axis == 'vertical')
    n_det = w if vertical else h
    n_ang = h if vertical else w

    # Choose stripe centers along detector axis
    centers = rng.uniform(0, n_det, size=num_stripes)

    # Choose widths and base amplitudes
    widths = rng.uniform(width_range[0], width_range[1], size=num_stripes)
    base_ampl = rng.uniform(amplitude_range[0], amplitude_range[1], size=num_stripes)

    # Slow amplitude variation across angles (orthogonal axis)
    # Create smooth variation field per stripe
    # Start with random noise and smooth it
    noise = rng.normal(0, 1, size=(num_stripes, n_ang))
    # Smooth each stripe's variation by Gaussian along angles
    var_sigma = max(1.0, variation_scale / 10.0)  # translate scale to sigma
    for i in range(num_stripes):
        noise[i] = gaussian_filter(noise[i], sigma=var_sigma, mode='reflect')
        # normalize to unit range and center
        nmin, nmax = noise[i].min(), noise[i].max()
        if nmax > nmin:
            noise[i] = (noise[i] - nmin) / (nmax - nmin)     # [0,1]
            noise[i] = (noise[i] - 0.5) * 2.0                # [-1,1]
        else:
            noise[i] = np.zeros_like(noise[i])

    # Construct stripe field
    field = np.zeros((h, w), dtype=np.float32)

    # Precompute detector coordinate array
    det_coords = np.arange(n_det, dtype=np.float32)

    for i in range(num_stripes):
        c = centers[i]
        sigma = max(1e-3, widths[i] / 2.355)  # convert FWHM-ish to sigma; gentle heuristic
        # 1D Gaussian profile along detector axis
        profile = np.exp(-0.5 * ((det_coords - c) / sigma) ** 2).astype(np.float32)

        # Normalize profile to unit peak
        if profile.max() > 0:
            profile /= profile.max()

        # Amplitude variation across angles for this stripe
        ampl_row = base_ampl[i] * (1.0 + 1.0 * noise[i]) #(1.0 + 0.6 * noise[i])  # 60% variation depth
        #ampl_row = base_ampl[i] * noise[i] # allow zero values
        
        if vertical:
            # Broadcast: angles (rows) x detector (cols)
            stripe = ampl_row[:, None] * profile[None, :]
            field += stripe
        else:
            stripe = ampl_row[None, :] * profile[:, None]
            field += stripe

    # Global scale and anisotropic blur to mimic phase retrieval
    field *= global_scale
    field = gaussian_filter(field, sigma=blur_sigma, mode='reflect')

    # Apply to image
    if mode == 'additive':
        out = img + field * np.max(np.abs(img))  # scale relative to image dynamic range
    elif mode == 'multiplicative':
        out = img * (1.0 + field)
    else:
        raise ValueError("mode must be 'additive' or 'multiplicative'")

    return out, field

################################## Multibang #############################################
class Multibang(Function):
    
    def __init__(self,**kwargs):
        
        utemp = np.sort(kwargs.get('u'))
        self.u = utemp 
    
    def convex_conjugate(self, x):
        return 0
    
    def __call__(self,x):
        
        u = self.u
        Max = u.max()
        utemp = np.concatenate(([-np.inf],u,[np.inf]))
        su = utemp.shape
        sx = x.shape
        ru = utemp.reshape(np.concatenate(([1,1],su)))
        t = x.as_array()
        rx = t.reshape(np.concatenate((sx,[1])))
        uval = np.sum(ru<=rx,axis=2)
        uval = uval-(x==Max)
        uL = utemp[uval-1]
        uR = utemp[uval]
        m = (uR-t)*(t-uL)
        m[m==np.nan] = np.inf
        return np.sum(m)
    
    def proximal(self,x, t, out=None):
        y = x*0
        u = self.u
        k = u.size
        xm = np.concatenate(([-np.inf],(1-t)*u[1:k]+t*u[0:(k-1)]))
        xm = xm.reshape(np.array([1,1,k]))
        xp = np.concatenate(((1-t)*u[0:k-1]+t*u[1:k],[np.inf]))
        xp = xp.reshape(np.array([1,1,k]))
        ax = x.as_array()
        sx = x.shape
        rx = ax.reshape(np.concatenate((sx,[1])))
        uvalm = np.sum(xm<rx,axis=2)
        uvalp = np.sum(xp<rx,axis=2)
        temp = uvalp<uvalm
        arr = np.zeros(sx)
        arr[temp] = u[uvalp[temp]]
        arr[~temp] = 1/(1-2*t)*(ax[~temp]-t*(u[uvalp[~temp]-1]+u[uvalp[~temp]]))
        
        y.fill(arr)
        
        if out is None:
            return y
        else:
            out.fill(y)

##################### Neural Fields ##############################
try:
    import torch
    class real_nf_sim(torch.nn.Module):
        def __init__(self, width, depth, encoding_width = None,encoding_std = 20,activation="SILU"):
            super().__init__()
            self.w = width
            self.d = depth
            if encoding_width is None:
                self.ew = 2*self.w
            else:
                self.ew = encoding_width

            #self.input = torch.nn.Linear(self.ew+2, self.w)
            self.input = torch.nn.Linear(self.ew, self.w)

            self.inner_layers = torch.nn.ModuleList( [torch.nn.Linear(self.w, self.w) for i in range(self.d)])
            if activation == "SILU":
                print("Using SILU activation")
                self.act = torch.nn.SiLU()
            elif activation == "SIREN":
                print("Using SIREN activation")
                self.act = lambda x: torch.sin(2*np.pi*x)/(2*np.pi)
            else:
                print("Using no activation")

            print(f"Using network with depth {depth}, width {width}, enc. width {self.ew}, enc. std {encoding_std} and residual connection")
            self.output = torch.nn.Linear(self.w, 1)
            self.Enc  = torch.nn.Parameter(torch.randn(2, self.ew)*encoding_std)#20
            self.Enc.requires_grad = False
            self.enc_bias = torch.nn.Parameter(torch.rand(self.ew)/4)
            self.enc_bias.requires_grad = False

        def forward(self, x):
            x = x@self.Enc + self.enc_bias[None,:]
            #x = torch.cat((x,y), dim = 1)
            x = torch.sin(2*np.pi*x)
            x = self.act(self.input(x))
            for l in self.inner_layers:
                x = self.act(l(x)) + x
            x = self.output(x)
            return x

    class real_nf_sim_nopos(torch.nn.Module):
        def __init__(self, width, depth,activation="SILU"):
            super().__init__()
            self.w = width
            self.d = depth

            #self.input = torch.nn.Linear(self.ew+2, self.w)
            self.input = torch.nn.Linear(2, self.w)
            print(f"Using network with depth {depth}, width {width} and residual connection (no pos. encoding)")
            self.inner_layers = torch.nn.ModuleList( [torch.nn.Linear(self.w, self.w) for i in range(self.d)])
            if activation == "SILU":
                print("Using SILU activation")
                self.act = torch.nn.SiLU()
            elif activation == "SIREN":
                print("Using SIREN activation")
                self.act = lambda x: torch.sin(2*np.pi*x)/(2*np.pi)
            else:
                print("Using no activation")

        
            self.output = torch.nn.Linear(self.w, 1)

        def forward(self, x):
            x = self.act(self.input(x))
            for l in self.inner_layers:
                x = self.act(l(x)) + x
            x = self.output(x)
            return x

except:
    print("torch not available")

##################### Mask operator ##############################

class MaskOperator(LinearOperator):
    def __init__(self, domain_geometry, range_geometry,r=[0,0], order='C'):
        """
        Custom linear operator to mask sinogram with zeroes.

        Parameters:
        domain_geometry (ImageGeometry): The geometry of the input space.
        range_geometry (ImageGeometry): The geometry of the output space.
        range: range to mask (detector axis)
        order (str): The order of flattening and reshaping operations.
                     'C' for row-major (C-style),
                     'F' for column-major (Fortran-style).
        """
        super(MaskOperator, self).__init__(domain_geometry=domain_geometry, 
                                               range_geometry=range_geometry)
        #self.A = A
        self.order = order # not sure I need this
        self.r = r

    def direct(self, x, out=None):
        array = x.as_array()
        masked = np.zeros_like(array)
        masked[:,int(self.r[0]):int(self.r[1])] = array[:,int(self.r[0]):int(self.r[1])]
        if out is None:
            result = self.range_geometry().allocate()
            result.fill(masked)
            return result
        else:
            out.fill(masked)

    def adjoint(self, y, out=None):
        array = y.as_array()
        masked = np.zeros_like(array)
        masked[:,int(self.r[0]):int(self.r[1])] = array[:,int(self.r[0]):int(self.r[1])]
        if out is None:
            result = self.range_geometry().allocate()
            result.fill(masked)
            return result
        else:
            out.fill(masked)