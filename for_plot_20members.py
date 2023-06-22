# generate csv file for 20 menbers 
import numpy as np
import glob, sys
import csv

# set up mesh
import matplotlib.tri as tri
from dolfin import *
from qgm2_parameters import *
from scipy import interpolate

mesh = RectangleMesh(Point(0.0, 0.0), Point(L, L), ngrid, ngrid)
space = FunctionSpace(mesh, "CG", 1)
test, trial = TestFunction(space), TrialFunction(space)
zero_bcs = DirichletBC(space, 0.0, "on_boundary")

# some general reusable variables
x_vec, y_vec = np.meshgrid(np.linspace(0, L, ngrid+1), np.linspace(0, L, ngrid+1))
input_points = np.array([x_vec[:].flatten(), y_vec[:].flatten()]).T
output_points = mesh.coordinates()
triang = tri.Triangulation(output_points[:, 0], output_points[:, 1])

# for putting stuff onto mesh
dof_2_vert = dof_to_vertex_map(space)

# for moving stuff onto grid
vert_2_dof = vertex_to_dof_map(space)



def grid_to_mesh(grid_data, mesh_func):
    """Takes grid data and interpolates it onto a pre-defined finite element mesh"""
    for layer in range(len(mesh_func)):
        data_in = grid_data[layer,:,:]

        # interpolator = interpolate.LinearNDInterpolator(input_points, data_in[:].flatten())
        interpolator = interpolate.NearestNDInterpolator(input_points, data_in[:].flatten())  # much faster
        mesh_func[layer].vector()[:] = interpolator(output_points[dof_2_vert])
        
    return mesh_func

def mesh_to_grid(mesh_func, ny=ngrid+1, nx=ngrid+1):
    """Takes finite element function and pull out data onto regular grid
    
       This subtroutine exploits the fact that the mesh is structued and regular, so just do it
       with vertex_to_dof map. If not regular mesh, will need to write a probing routine,
       or cheat by pulling out vertices then constructing an interpolator.
    """
    layers = len(mesh_func)
    grid_data = np.zeros((layers, ny, nx))
    for l in range(layers):
        grid_data[l, :, :] = np.reshape(mesh_func[l].vector()[vert_2_dof], (ngrid+1, ngrid+1))
        
    return grid_data


def cal_div(u,v):
    if u.shape[0] >512:
        u=u[:-1,:-1]
        v=v[:-1,:-1]
        
    uq_p1=np.zeros((3,513,513))
    uq_p1[0,:-1,:-1]=u[:,:]
    vq_p1=np.zeros((3,513,513))
    vq_p1[0,:-1,:-1]=v[:,:]
    
    uq_p_mesh=grid_to_mesh(uq_p1,[Function(space, name = "uq_p_%i" % (i + 1)) for i in range(layers)])
    vq_p_mesh=grid_to_mesh(vq_p1,[Function(space, name = "vq_p_%i" % (i + 1)) for i in range(layers)])
    
    div_uq00 = [Function(space, name = "div_uq_%i" % (i + 1)) for i in range(layers)]
    for l in range(layers):  
        Lt = (div(as_vector([uq_p_mesh[l], vq_p_mesh[l]])) - trial) * test * dx
        solve(lhs(Lt) == rhs(Lt), div_uq00[l])
        
    return mesh_to_grid(div_uq00)[0,:-1,:-1] # return [512,512]


def cal_mix_norm2(f_input):
    Nx = 512
    Ny = 512
    x = np.linspace(0, 2*np.pi, Nx, endpoint=False)  # python FFT assumes has no end point
    y = np.linspace(0, 2*np.pi, Ny, endpoint=False)
    xx, yy = np.meshgrid(x, y)
    
    f=f_input

    f_h = np.fft.fft2(f)

    kx = np.concatenate([np.arange(Nx//2+1), np.arange(-Nx//2+1, 0)])  # [0, 1, 2, .. N, -(N-1)... -1 ] convention
    ky = np.concatenate([np.arange(Ny//2+1), np.arange(-Ny//2+1, 0)])

    kxx, kyy = np.meshgrid(kx ,ky)
    k2 = kxx**2 + kyy**2  # create |k|^2 = kx^2 + ky^2 (this is all real so it's fine)

    p = .5# H^-p norm (because can't take power to neg integer)

    f_h_int = np.zeros(kxx.shape, dtype=complex)
    for i in range(Nx):
        for j in range(Ny):
            if (i == 0) & (j == 0):
                f_h_int[i, j] = 0  # don't do anything to the zero mode
            else:
                f_h_int[i, j] =np.abs(f_h[i, j] / (Nx * Ny))**2 / (k2[i, j])**p  # normalisation
    return np.real(np.sum(f_h_int) * Nx*Ny)



# sum up data
rni1=30
rni2=50
nre='00' # defaul setting
perc=6400 # data used
nl = 0 # noise level

dir_res='../data/preds/00/'#
fw = open('../data/for_plot/l2_%s_paper_1.csv'%(nre), 'w')
writer = csv.writer(fw)
for nlayer2 in range(1): # Only surface
    for fn in ['empb','em','eb' ]:#   
        for var in ['psi','q','rel']:#
            vn='div'
            print(nre,vn,fn,var)
            for rni in range(rni1,rni2):
                div_uq_grid_o=np.load('../data/training/00/div_uq_%s_80.npy'%(fn))[nlayer2,:-1,:-1]
                pdd = np.load(dir_res+'%s/Pr_%s_%s_%s_%s_%s.npy'%(nlayer2,vn,fn,nre,var,rni))
                if pdd.shape[0]>512:
                    pdd=pdd[:-1,:-1]
                tl=np.sum(np.power((div_uq_grid_o-pdd),2))
                tt=cal_mix_norm2(div_uq_grid_o-pdd)
                writer.writerow([tl,np.sqrt(np.divide(tl,np.sum(np.power(div_uq_grid_o,2)))),
                                 tt,np.sqrt(np.divide(tt,cal_mix_norm2(div_uq_grid_o))),
                                     fn,var,vn,perc,rni,nl])
            
            for vn in ['grad','uvq']:
                print(nre,vn,fn,var)
                for rni in range(rni1,rni2):
                    div_uq_grid_o=np.load('../data/training/00/div_uq_%s_80.npy'%(fn))[nlayer2,:-1,:-1]
                    ff = np.load(dir_res+'%s/Pr_%s_%s_%s_%s_%s.npz'%(nlayer2,vn,fn,nre,var,rni))
                    uq_p=ff[ff.files[0]]
                    vq_p=ff[ff.files[1]]
                    div_grid=cal_div(uq_p,vq_p)
                    if vn=='grad':
                        tl=np.sum(np.power((div_uq_grid_o+div_grid),2)) 
                        tt=cal_mix_norm2(div_uq_grid_o+div_grid)
                    else:
                        tl=np.sum(np.power((div_uq_grid_o-div_grid),2))
                        tt=cal_mix_norm2(div_uq_grid_o-div_grid)
                    writer.writerow([tl,np.sqrt(np.divide(tl,np.sum(np.power(div_uq_grid_o,2)))),
                                     tt,np.sqrt(np.divide(tt,cal_mix_norm2(div_uq_grid_o))),
                                     fn,var,vn,perc,rni,nl])
fw.close()