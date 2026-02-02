import matplotlib.pyplot as plt
import numpy as np
import torch
import copy 

from cil.framework import  AcquisitionGeometry, ImageData, AcquisitionData
from cil.plugins.astra.operators import ProjectionOperator

import os
os.chdir('/dtu-compute/Mathcrete_thesis/mathcrete_dtu/Simulation')
import helper_funcs as hf
import dxchange 
import argparse
import time

# Script for training neural field for CT reconstruction

def reconstruct(image_op, measurements, \
                net, coordinates, \
                nits = 10**3, \
                nprint=200 ,lr = 1e-3
                ):
    torch.cuda.empty_cache()
    coords = coordinates.to(dev)
    meas = measurements.to(dev)

    N  = meas.shape[1]
    LOSS_list = []
    opt = torch.optim.Adam(net.parameters(), lr = lr)
    for it in range(nits):
        if it == nits//2:
            print("Allowing encoding weights to change")
            net.Enc.requires_grad = True
            net.enc_bias.requires_grad = True
            opt = torch.optim.Adam(net.parameters(), lr = lr)

        opt.zero_grad()
        F = net(coords).reshape((N,N))

        exp_meas =  image_op.apply(F)
        loss = torch.mean( (exp_meas - meas)**2)

        #print(it, df.item(), loss.item(), reg.item())
        if (it%nprint == 0) and nprint > 0 :
            print(f"Iteration: {it}", f"Loss: {loss.item()}")
        LOSS_list.append(loss.item())
        loss.backward()
        opt.step()
    return LOSS_list

def reconstruct_tv(image_op, measurements, \
                net, coordinates, \
                gamma, nits = 10**3, \
                nprint=200, lr = 1e-3
                ):
    torch.cuda.empty_cache()
    coords = coordinates.to(dev)
    meas = measurements.to(dev)
    N  = meas.shape[1]
    LOSS_list = []
    opt = torch.optim.Adam(net.parameters(), lr = lr)

    for it in range(nits):
        
        if it == nits//2:
            print("Allowing encoding weights to change")
            net.Enc.requires_grad = True
            net.enc_bias.requires_grad = True
            opt = torch.optim.Adam(net.parameters(), lr = lr)

        opt.zero_grad()
        F = net(coords).reshape((N,N))

        exp_meas =  image_op.apply(F)
        df = torch.mean( (exp_meas - meas)**2)
        
        Fx = torch.roll(F, 1, 0)
        Fy = torch.roll(F, 1, 1)

        reg = gamma*N*torch.mean( torch.sqrt( (Fx - F)**2 + (Fy - F)**2 +1e-6)-1e-3)#+ 1e-8) - 1e-5)
        loss = 0.5*df + reg

        if (it%nprint == 0) and nprint > 0 :
            print(f"Iteration: {it}", f"Loss: {loss.item()}")

        LOSS_list.append(loss.item())
        loss.backward()
        opt.step()
    return LOSS_list

def prox_reconstruct(image_op, measurements, \
                net, coordinates, \
                global_its = 200, \
                nprint = 1,
                nmetrics = 1,
                innter_its = 10, nb = 10, lr = 1e-3, \
                gt=None, print_inner=True):

    N = measurements.array.shape[1]
    dev = net.Enc.get_device()

    ag = image_op.sinogram_geometry
    ig = ag.get_ImageGeometry()

    inds = np.arange(coordinates.shape[0])
    binds = np.arange(nb)

    opt = torch.optim.Adam(net.parameters(), lr = lr)
    
    LOSS_list = []
    # When ground truth is present
    RRMSE_list = []
    SSIM_list = []

    for global_it in range(global_its):
        torch.cuda.empty_cache()
        if global_it == global_its//2:
            print("Allowing encoding weights to change")
            net.Enc.requires_grad = True
            net.enc_bias.requires_grad = True
            opt = torch.optim.Adam(net.parameters(), lr = lr)
            
        F_np = np.zeros(N**2)
        with torch.no_grad():
            for b in range(nb):
                F_np[b::nb] = net(coordinates[b::nb,:].to(dev)).cpu().detach().numpy().flatten()
            F_np = F_np.reshape((N,N))
        
        if nmetrics > 0:
            RRMSE_true, SSIM_true = hf.compute_metrics(gt,hf.circle_mask(F_np),
                                        get_values=True,print_out=False)
            RRMSE_list.append(RRMSE_true)
            SSIM_list.append(SSIM_true)
            if (global_it%nmetrics == 0):
                print(f"global_it={global_it}", f"Global RRMSE, SSIM: {RRMSE_true:.4f}, {SSIM_true:.4f}")

        F_cil = ImageData(F_np, geometry = ig)
        res_cil = image_op.direct(F_cil) - measurements

        data_fidelity = np.mean(res_cil.array)**2
        LOSS_list.append(data_fidelity)
        if (global_it%nprint == 0) and nprint > 0 :
            print(f"global_it={global_it}", f"Global loss (data fidelity): {data_fidelity}")

        grad = image_op.adjoint( res_cil )
        agrad =  image_op.direct(grad)

        global_Lr = np.mean( grad.array**2)/np.mean(agrad.array**2)
        print("     Global LR: {}".format(global_Lr))
        
        F_np -= global_Lr*grad.array
        f_torch = torch.from_numpy(F_np).flatten().to(dev)
        
        for it in range(innter_its):
            np.random.shuffle(inds)
            np.random.shuffle(binds)
            loss_sum = 0
            for b in binds:
                opt.zero_grad()
                f = net(coordinates[inds[b::nb], :].to(dev)).flatten()
                loss = torch.mean( (f-f_torch[inds[b::nb]])**2)
                loss_sum += loss.item()
                loss.backward()
                opt.step()
            if print_inner:
                print(f"Inner it: {it}: loss: {loss_sum}")

    if nmetrics > 0:
        return LOSS_list, RRMSE_list, SSIM_list
    else: 
        return LOSS_list

        

def prox_reconstruct_tv(image_op, measurements, \
                net, coordinates, gamma, \
                global_its = 200, innter_its = 10, nb = 20, lr = 1e-3, \
                nprint = 1, nmetrics=1, gt=None, print_inner=True):

    N = measurements.array.shape[1]
    dev = net.Enc.get_device()

    ag = image_op.sinogram_geometry
    ig = ag.get_ImageGeometry()

    # inds = np.arange(coordinates.shape[0])
    # binds = np.arange(nb)
    inds = torch.arange(coordinates.shape[0], device=dev) 
    binds = torch.arange(nb, device=dev)

    opt = torch.optim.Adam(net.parameters(), lr = lr)

    LOSS_list = []
    # When ground truth is present
    RRMSE_list = []
    SSIM_list = []
    coordinates = coordinates.to(dev)
    for global_it in range(global_its):

        torch.cuda.empty_cache()
        if global_it == global_its//2:
            print("Allowing encoding weights to change")
            net.Enc.requires_grad = True
            net.enc_bias.requires_grad = True
            opt = torch.optim.Adam(net.parameters(), lr = lr)
        
        F_np = np.zeros(N**2)
        with torch.no_grad():
            for b in range(nb):
                F_np[b::nb] = net(coordinates[b::nb,:].to(dev)).cpu().detach().numpy().flatten()
            F_np = F_np.reshape((N,N))

        if nmetrics > 0:
            RRMSE_true, SSIM_true = hf.compute_metrics(gt,hf.circle_mask(F_np),
                                        get_values=True,print_out=False)
            RRMSE_list.append(RRMSE_true)
            SSIM_list.append(SSIM_true)
            if (global_it%nmetrics == 0):
                print(f"global_it={global_it}", f"Global RRMSE, SSIM: {RRMSE_true:.4f}, {SSIM_true:.4f}")

        F_cil = ImageData(F_np, geometry = ig)

        res_cil = image_op.direct(F_cil) - measurements

        data_fidelity = np.mean(res_cil.array)**2
        LOSS_list.append(data_fidelity)
        if (global_it%nprint == 0) and nprint > 0 :
            print(f"global_it={global_it}", f"Global loss (data fidelity): {data_fidelity}")

        grad = image_op.adjoint( res_cil )
        agrad =  image_op.direct(grad)

        global_Lr = np.mean( grad.array**2)/np.mean(agrad.array**2)
        print("     Global LR: {}".format(global_Lr))
        
        F_np -= global_Lr*grad.array
        f_torch = torch.from_numpy(F_np).flatten().to(dev)

        for it in range(innter_its):
            # np.random.shuffle(inds)
            # np.random.shuffle(binds)
            perm_inds = inds[torch.randperm(len(inds), device=dev)] 
            perm_binds = binds[torch.randperm(len(binds), device=dev)]
            loss_sum = 0
            for b in perm_binds:
                opt.zero_grad()
                f = net(coordinates[perm_inds[b::nb], :].to(dev)).flatten()
                df = torch.mean( (f- f_torch[perm_inds[b::nb]])**2)

                fx = net(coordinates[perm_inds[b::nb] - 1, :].to(dev)).flatten() 
                fy = net(coordinates[perm_inds[b::nb] - N, :].to(dev)).flatten() 

                reg = global_Lr*gamma*N*torch.mean( torch.sqrt( (fx - f)**2 + (fy - f)**2 + 1e-6)-1e-3)#+ 1e-8) - 1e-5)
                loss = 0.5*df + reg
                loss_sum += loss.item()
                loss.backward()
                opt.step()
            if print_inner:
                print(f"Inner it: {it}: loss: {loss_sum}")
                

        
    if nmetrics > 0:
        return LOSS_list, RRMSE_list, SSIM_list
    else: 
        return LOSS_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural field reconstruction training with ProxNF")
    parser.add_argument("--folder", type=str, default="nf_recon_tv", help="Folder to store results")
    parser.add_argument("--path_meas", type=str, default="Simulated_sinogram/sinogram_noisy_and_stripes.npy", help="Path to measurements")
    parser.add_argument("--path_gt", type=str, default="phantoms/full2560_2d_hann.npy", help="Path to ground truth")
    parser.add_argument("--nb_prox", type=int, default=20, help="Number of batches in ProxNF")
    parser.add_argument("--angle_range", type=int, default=360, help="Angle range")
    parser.add_argument("--sub_nits", type=int, default=int(2*10**3), help="Number of iterations for subsample training")
    parser.add_argument("--prox_nits", type=int, default=200, help="Number of global proxNF iterations")
    parser.add_argument("--ssfs", nargs="+", type=int, default=[10, 5, 2], help="Subsample factors")
    parser.add_argument("--lr_prox",type=float,default=1e-4,help="Learning rate for prox training")
    parser.add_argument("--lr_sub",type=float,default=1e-3,help="Learning rate for subsample training)")
    parser.add_argument("--norm_output",type=bool,default=False,help="Normalize output of NF after training")
    parser.add_argument("--encoding_std",type=int,default=20,help="Std in fourier feature postional encoding of network")
    parser.add_argument("--width",type=int,default=200,help="Depth of network")
    parser.add_argument("--depth",type=int,default=2,help="Depth of network")
    parser.add_argument("--tv",type=bool,default=False, help="To include TV")
    parser.add_argument("--morozov",type=bool,default=False,help="Find TV param by morozov")
    parser.add_argument("--mupdate",type=float,default=1.1)
    parser.add_argument("--tv_reg",type=float,default=1,help="TV parameter")
    parser.add_argument("--load_base",type=bool,default=False)
    parser.add_argument("--skip_ss",type=bool,default=False)
    parser.add_argument("--base_path",type=str,default=None)
    args = parser.parse_args()

    assert torch.cuda.is_available()
    dev = torch.device("cuda:0")
    
    morozov_update = args.mupdate
    if args.tv:
        print(f"Using TV with reg. parameter {args.tv_reg}")
    print("Using learning params:")
    print(f"nb_prox: {args.nb_prox}, lr_sub: {args.lr_sub}, lr_prox: {args.lr_prox} ")
    #F_gt = plt.imread('MgO_insitu_water_35nm_bottom.png')
    F_gt = np.load(args.path_gt)
    N = F_gt.shape[0]

    grid_mks = (2*np.arange(N) + 1)/(2*N) - 1/2
    c0, c1 = np.meshgrid(grid_mks, grid_mks)

    #F_gt -= 0.50000763
    F_norm = np.mean(F_gt**2)**(1/2)
    F_gt /= F_norm

    ssfs = args.ssfs #[10, 5, 2]

    # network size
    width = args.width
    depth = args.depth
    pos_std = args.encoding_std 
    #net = real_nf(width, depth).to(dev)
    net = hf.real_nf_sim(width, depth,activation="SIREN",encoding_std = pos_std).to(dev)
    if args.load_base:
        print("Loading base network:"+args.base_path)
        state_dict = torch.load(args.base_path)
        net.load_state_dict(state_dict)
    else:
        print("Loading no base network")
        
    # Ring remover
    #Meas = np.load('measurements.npy')
    print(f"Loading sinogram from {args.path_meas}")
    sinogram = np.load(args.path_meas)

    no_angles = sinogram.shape[0]
    if args.angle_range == 360: # cement phantom
        angles = dxchange.read_hdf5("/dtu-compute/Mathcrete_thesis/MgO_hydrate/MgO_insitu_dry_35nm_bottom_000_result01.nx",
                                dataset="entry_1/sample/rotation_angle") # Use same angles as experiment
        mb_values = np.array([0,0.25/2,(0.25+0.4)/2,(0.56+0.4)/2,(0.56+0.68)/2,1]) / F_norm
        _, _, data_no_rr = hf.CIL_setup_cement(sinogram,pixel_size=2/N,angle_range=args.angle_range)
        data = hf.CIL_ring_remover(data_no_rr,decNum=4,wname="db10",sigma=4)
        
    elif args.angle_range == 180: # shepp logan
        angles = np.linspace(0, 180, sinogram.shape[0], endpoint=False)
        mb_values = np.array([0,0.1,0.2,0.3,0.4,1]) / F_norm
        _, _, data = hf.CIL_setup_cement(sinogram,pixel_size=2/N,angle_range=args.angle_range)  

    Meas = data.array
    Meas /= F_norm # added by luke
    #angles = (np.arange(180)).astype(np.float32)


    for s, ssf in enumerate(ssfs):
        if args.skip_ss:
            print("Skipping subsample training")
            break
        print("Subsample Factor {}".format(ssf))
        n = N//ssf
        f_gt = 0*F_gt[::ssf,::ssf]/(ssf**2)

        for i in range(ssf):
            for j in range(ssf):
                f_gt += F_gt[i::ssf,j::ssf]/(ssf**2)

        f_norm = np.mean(f_gt**2)**(1/2)
        f_gt/= f_norm

        grid_mks = (2*np.arange(n) + 1)/(2*n) - 1/2
        c0, c1 = np.meshgrid(grid_mks, grid_mks)
        XY = torch.stack((torch.from_numpy(c0.flatten()),torch.from_numpy(c1.flatten())), axis = 1).float()
        
        ma = (len(angles)//ssf)*ssf
        angles_small = angles[:ma:ssf]
        meas = 0*Meas[:ma:ssf,::ssf]
        
        for j in range(ssf):
            for i in range(ssf):
                meas += Meas[i:ma:ssf,j::ssf]/(ssf**2)
        meas_norm = np.mean(meas**2)**(1/2)
        meas_torch = torch.from_numpy(meas)

        ag = AcquisitionGeometry.create_Parallel2D()\
                                .set_angles(angles_small)\
                                .set_panel(n, pixel_size=(2/N)*ssf)
        ig = ag.get_ImageGeometry()
        A = ProjectionOperator(ig, ag, device='gpu')
        delta = np.linalg.norm(A.direct(ig.allocate(f_gt)).array-meas) # for morozov

        class XrayTorchOp(torch.autograd.Function):

            @staticmethod
            def forward(ctx, input):
                ctx.save_for_backward(input)
                X = ImageData(input.cpu().numpy(), geometry = ig)
                y = A.direct(X).array
                return torch.from_numpy(y).to(input.get_device())
            @staticmethod
            def backward(ctx, grad_output):
                Y = AcquisitionData(grad_output.cpu().numpy(), geometry = ag)
                x = A.adjoint(Y).array
                return  torch.from_numpy(x).to(grad_output.get_device())
        
        # Try loading presaved network - otherwise we train a new one
        try:
            state_dict = torch.load("neural_field_saves/"+args.folder+ "/net_{}".format(ssf))
            net.load_state_dict(state_dict)
                        
        except:
            print("Retraining Net {}".format(ssf))
            net.Enc.requires_grad = False
            net.enc_bias.requires_grad = False
            Time = time.time()
            if s == 0: # Do more cheap training iterations for first subsample level
                sub_no_iter = 2*args.sub_nits
            else: 
                sub_no_iter = args.sub_nits
            if args.tv:
                gamma_factor = meas.shape[0]*meas.shape[1]/(n**2)
                print(f"Gamma factor (M/N**2): {gamma_factor}")
                alpha_tv = args.tv_reg
                if s == 0 and args.morozov:
                    alpha_old = alpha_tv
                    max_miter = 10
                    mrep = 0
                    print(f"Finding alpha with Morozov, delta = {delta}")
                    while mrep < max_miter:
                        old_net = copy.deepcopy(net)
                        if mrep != 0:
                            sub_no_iter = args.sub_nits
                            
                        _ = reconstruct_tv(XrayTorchOp, meas_torch, net, XY, gamma=gamma_factor*alpha_tv,
                                                   nits = 2*sub_no_iter,lr=args.lr_sub)
                        with torch.no_grad():
                            f_temp = net(XY.to(dev)).reshape((n,n))
                            f_temp_np = f_temp.cpu().detach().numpy()
                        res, mcheck = hf.morozov_check(sol=ig.allocate(f_temp_np),operator=A,ymeas=meas,delta=delta)
                        print(f"Alpha (TV): {alpha_tv}")
                        print(f"Residual: {res}")
                        print(f"Morozov principle satisfied: {mcheck}")
                        if not mcheck or mrep == max_miter-1:
                            alpha_tv = alpha_old
                            print(f"Alpha found as {alpha_tv}")
                            net = copy.deepcopy(net)
                            break
                        else:
                            alpha_old = alpha_tv
                            alpha_tv = morozov_update*alpha_tv
                            mrep += 1

                LOSS_list_ssf = reconstruct_tv(XrayTorchOp, meas_torch, net, XY, gamma=gamma_factor*alpha_tv,
                                                   nits = sub_no_iter,lr=args.lr_sub)
            else:
                LOSS_list_ssf = reconstruct(XrayTorchOp, meas_torch, net, XY, nits = sub_no_iter, 
                                                lr=args.lr_sub)

            print("Subsample Training Time: ", time.time() - Time)
            plt.close()
            plt.clf()
            plt.semilogy(range(sub_no_iter),LOSS_list_ssf)
            plt.title("Training loss curve (Subsampling)")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")          
            plt.savefig("plots/"+args.folder+f"/losscurve_ssf{ssf}.png")

            torch.save(net.state_dict(), "neural_field_saves/"+args.folder + "/net_{}".format(ssf)) 
                      
        subsample_batch = False
        if subsample_batch: ### Do subsample reconstruction in batches
            nb = 10
            f = np.zeros(n**2)
            with torch.no_grad():
                for b in range(nb):
                    f[b::nb] = net(XY[b::nb,:].to(dev)).cpu().detach().numpy().flatten()
            f_np = f.reshape((n,n))
        else:
            with torch.no_grad():
                f = net(XY.to(dev)).reshape((n,n))
                f_np = f.cpu().detach().numpy()
        
        if args.norm_output:
            f_np_norm = np.mean(f_np**2)**(1/2)
            f_np/= f_np_norm


        hf.plot_compare_diff([f_gt,f_np,f_np-f_gt],title=["GT","Neural field rep.","Difference"], \
                             savefig=True,savepath="plots/"+args.folder+"/subsample_recon_{}.png".format(ssf),use_range=0)
        # plt.close()
        # plt.clf()
        # fig, a = plt.subplots(1,3)
        # a[0].imshow(f_gt, vmin = f_gt.min(), vmax = f_gt.max(),cmap="gray")
        # a[1].imshow(f_np, vmin = f_gt.min(), vmax = f_gt.max(),cmap="gray")
        # a2 = a[2].imshow(f_np - f_gt,cmap="gray")
        # plt.colorbar(a2)
        # plt.savefig('plots/'+ args.folder + '/subsample_recon_{}0.png'.format(ssf)) #dpi = 1000)
        # del fig
        # del a
        
        rrmse_sub = np.mean( (f_np - f_gt)**2)**(1/2)/(np.mean(  f_gt**2)**(1/2))
        print(" Subsample Error: {}".format(rrmse_sub))

        rrmse2 = 0
        for i in range(ssf):
            for j in range(ssf):
                rrmse2 += np.mean( (f_np - F_gt[i::ssf, j::ssf])**2)
        rrmse = rrmse2**(1/2)/ssf
        print(" Global Error: {}".format(rrmse))

    ag = AcquisitionGeometry.create_Parallel2D()\
                            .set_angles(angles)\
                            .set_panel(N, pixel_size=(2/N)*1)
    ig = ag.get_ImageGeometry()
    A = ProjectionOperator(ig, ag, device='gpu')
    meas_cil = AcquisitionData(Meas, geometry = ag)

    grid_mks = (2*np.arange(N) + 1)/(2*N) - 1/2
    c0, c1 = np.meshgrid(grid_mks, grid_mks)
    XY = torch.stack((torch.from_numpy(c0.flatten()),torch.from_numpy(c1.flatten())), axis = 1).float()

    print("Prox reconstruct")
            # Try loading presaved network - otherwise we train a new one
    try:
        state_dict = torch.load("neural_field_saves/"+args.folder+ "/net_0")
        print("Loaded net_0")
        net.load_state_dict(state_dict)
        net.Enc.requires_grad = False
        net.enc_bias.requires_grad = False
    except: 
        print("no net_0 found - training from scratch")
        net.Enc.requires_grad = False
        net.enc_bias.requires_grad = False
    finally:

        if args.tv:
            gamma_factor = Meas.shape[0]*Meas.shape[1]/(N**2)
            print(f"Gamma factor (M/N**2): {gamma_factor}")
            Time = time.time()
            LOSS_list, RRMSE_list, SSIM_list = prox_reconstruct_tv(A, meas_cil, net, XY,gamma=gamma_factor*args.tv_reg,global_its=args.prox_nits,lr=args.lr_prox, gt = F_gt, \
                                                               nb = args.nb_prox)
            print("Time for ProxNF training: ", time.time() - Time)
        else:
            Time = time.time()        
            LOSS_list, RRMSE_list, SSIM_list = prox_reconstruct(A, meas_cil, net, XY,global_its=args.prox_nits,lr=args.lr_prox, gt = F_gt)
            print("Time for ProxNF training: ", time.time() - Time)
    
    plt.close()
    plt.clf()
    plt.semilogy(range(args.prox_nits),LOSS_list)
    plt.title("Training loss curve (ProxNF)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")          
    plt.savefig("plots/"+args.folder+"/losscurve.png")

    plt.close()
    plt.clf()
    plt.plot(range(args.prox_nits),RRMSE_list)
    plt.title("Training RRMSE curve")
    plt.xlabel("Epochs")
    plt.ylabel("RRMSE")          
    plt.savefig("plots/"+args.folder+"/RRMSEcurve.png")

    plt.close()
    plt.clf()
    plt.plot(range(args.prox_nits),SSIM_list)
    plt.title("Training SSIM curve")
    plt.xlabel("Epochs")
    plt.ylabel("RRMSE")          
    plt.savefig("plots/"+args.folder+"/SSIMcurve.png")

    #######################################
    print("Doing Global Reconstruction")
    F_np = np.zeros(N**2)

    nb = 10
    with torch.no_grad():
        for b in range(nb):
            F_np[b::nb] = net(XY[b::nb,:].to(dev)).cpu().detach().numpy().flatten()
        F_np = F_np.reshape((N,N))
    
    if args.norm_output:
        F_np_norm = np.mean(F_np**2)**(1/2)
        F_np/= F_np_norm

    # plt.close()
    # plt.clf()
    # fig, a = plt.subplots(1,3)
    # a[0].imshow(F_gt, vmin = F_gt.min(), vmax = F_gt.max())
    # a[1].imshow(F_np, vmin = F_gt.min(), vmax = F_gt.max())
    # a2 = a[2].imshow(F_np - F_gt)
    # plt.colorbar(a2)
    # plt.savefig('plots/' + args.folder + '/recon_00.png') #dpi = 1000)
    zooms = [1130,1430]
    hf.plot_compare([F_gt,F_np,F_np-F_gt],title=["GT","Neural field rep.","Difference"],savefig=True,savepath="plots/"+args.folder+"/net_0",use_range=0)
    hf.plot_compare([F_gt[zooms[0]:zooms[1],zooms[0]:zooms[1]],F_np[zooms[0]:zooms[1],zooms[0]:zooms[1]],(F_np-F_gt)[zooms[0]:zooms[1],zooms[0]:zooms[1]]],title=["GT","Neural field rep.","Difference"],savefig=True,savepath="plots/"+args.folder+"/net_0_zoom_middle",use_range=0)
    torch.save(net.state_dict(), "neural_field_saves/" + args.folder + "/net_0")
    
    # rrmse = np.mean((F_np - F_gt)**2)**(1/2)
    # print("RRMSE Error: {}".format(rrmse))
    hf.compute_metrics(F_gt,hf.circle_mask(F_np),mb_values=mb_values)

