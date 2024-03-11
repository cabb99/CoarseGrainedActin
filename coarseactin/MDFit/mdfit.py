import numpy as np
from numba import jit, prange, vectorize, float64
import math
from scipy.special import erf

class MDFit:
    def __init__(self, coordinates, sigma, experimental_map,voxel_size,padding=3):
        self.coordinates=coordinates
        self.experimental_map=experimental_map
        self.sigma=np.array([1,1,1])
        self.padding=padding
        self.n_voxels=np.array(experimental_map.shape)
        self.voxel_size=np.array(voxel_size)
        self.voxel_limits=[np.arange(-padding,n_voxels[0]+1+padding)*voxel_size[0],
                           np.arange(-padding,n_voxels[1]+1+padding)*voxel_size[1],
                           np.arange(-padding,n_voxels[2]+1+padding)*voxel_size[2]]

    def fold_padding(self,volume_map):
        p=self.padding
        vp=volume_map.copy()
        if p>0 and len(volume_map.shape)==3:
            vp[-2*p:-p, :, :] += vp[:p, :, :]
            vp[:, -2*p:-p, :] += vp[:, :p, :]
            vp[:, :, -2*p:-p] += vp[:, :, :p]
            vp[p:2*p, :, :]   += vp[-p:, :, :]
            vp[:, p:2*p, :]   += vp[:, -p:, :]
            vp[:, :, p:2*p]   += vp[:, :, -p:]
            vp=vp[p:-p, p:-p, p:-p]
        elif p>0 and len(volume_map.shape)==4:
            vp[:, -2*p:-p, :, :] += vp[:, :p, :, :]
            vp[:, :, -2*p:-p, :] += vp[:, :, :p, :]
            vp[:, :, :, -2*p:-p] += vp[:, :, :, :p]
            vp[:, p:2*p, :, :]   += vp[:, -p:, :, :]
            vp[:, :, p:2*p, :]   += vp[:, :, -p:, :]
            vp[:, :, :, p:2*p]   += vp[:, :, :, -p:]
            vp=vp[:,p:-p, p:-p, p:-p]
        return vp

    def sim_map(self):
        sigma=self.sigma*np.sqrt(2)
        phix=(1+erf((self.voxel_limits[0]-self.coordinates[:,None,0])/sigma[0]))/2
        phiy=(1+erf((self.voxel_limits[1]-self.coordinates[:,None,1])/sigma[1]))/2
        phiz=(1+erf((self.voxel_limits[2]-self.coordinates[:,None,2])/sigma[2]))/2

        dphix=(phix[:,1:]-phix[:,:-1])
        dphiy=(phiy[:,1:]-phiy[:,:-1])
        dphiz=(phiz[:,1:]-phiz[:,:-1])
        
        smap=(dphix[:,:,None,None]*dphiy[:,None,:,None]*dphiz[:,None,None,:]).sum(axis=0)
        
        return self.fold_padding(smap)

    def corr_coef(self):
        simulation_map=self.sim_map()
        return (simulation_map*self.experimental_map).sum()/np.sqrt((simulation_map**2).sum()*(self.experimental_map**2).sum())

    def dcorr_coef_numerical(self, delta=1e-5):
        num_derivatives = np.zeros(self.coordinates.shape)
        original_corr_coef = self.corr_coef()
    
        for i in range(self.coordinates.shape[0]):
            for j in range(self.coordinates.shape[1]):
                # Perturb coordinates positively
                self.coordinates[i, j] += delta
                positive_corr_coef = self.corr_coef()
                
                # Perturb coordinates negatively
                self.coordinates[i, j] -= 2*delta
                negative_corr_coef = self.corr_coef()
                
                # Compute numerical derivative
                num_derivatives[i, j] = (positive_corr_coef - negative_corr_coef) / (2*delta)
                
                # Reset coordinates to original value
                self.coordinates[i, j] += delta
                
        return num_derivatives

    def fit(self,numerical=False):
        f=1
        for i in range(1000):
            if numerical:
                dx=self.dcorr_coef_numerical()
            else:
                dx=self.dcorr_coef()
            f=.1/np.abs(dx).max()
            self.coordinates=self.coordinates+f*dx
            for j in range(3): 
                self.coordinates[:, j][self.coordinates[:, j] < 0] += self.voxel_size[j] * self.n_voxels[j]
                self.coordinates[:, j][self.coordinates[:, j] >= self.voxel_size[j] * self.n_voxels[j]] -= self.voxel_size[j] * self.n_voxels[j]
            if i%10==0:
                print(i,self.corr_coef())

    def dsim_map_numerical(self, delta=1e-5):
        num_particles = self.coordinates.shape[0]
        sim_map_shape = self.sim_map().shape
        derivatives = {
            'dx': np.zeros((num_particles,) + sim_map_shape),
            'dy': np.zeros((num_particles,) + sim_map_shape),
            'dz': np.zeros((num_particles,) + sim_map_shape)
        }
        
        original_sim_map = self.sim_map()
        
        for i in range(num_particles):
            for j, direction in enumerate(['dx', 'dy', 'dz']):
                original_coordinate = self.coordinates[i, j]
        
                # Perturb coordinate in the positive direction
                self.coordinates[i, j] = original_coordinate + delta
                positive_sim_map = self.sim_map()
        
                # Perturb coordinate in the negative direction
                self.coordinates[i, j] = original_coordinate - delta
                negative_sim_map = self.sim_map()
        
                # Compute numerical derivative for this particle and direction
                derivatives[direction][i] = (positive_sim_map - negative_sim_map) / (2 * delta)
        
                # Reset coordinate to original value
                self.coordinates[i, j] = original_coordinate
        
        return derivatives

    @staticmethod
    def outer_mult(x,y,z):
        return x[:,:,None,None]*y[:,None,:,None]*z[:,None,None,:]
    
    def dsim_map(self):
        sigma=self.sigma*np.sqrt(2)

        x_mu_sigma=(self.voxel_limits[0]-self.coordinates[:,None,0])/sigma[0]
        y_mu_sigma=(self.voxel_limits[1]-self.coordinates[:,None,1])/sigma[1]
        z_mu_sigma=(self.voxel_limits[2]-self.coordinates[:,None,2])/sigma[2]
        
        phix=(1+erf(x_mu_sigma))/2
        phiy=(1+erf(y_mu_sigma))/2
        phiz=(1+erf(z_mu_sigma))/2
        
        dphix_dx= -np.exp(-x_mu_sigma**2) / np.sqrt(np.pi) / sigma[0]
        dphiy_dy= -np.exp(-y_mu_sigma**2) / np.sqrt(np.pi) / sigma[1]
        dphiz_dz= -np.exp(-z_mu_sigma**2) / np.sqrt(np.pi) / sigma[2]
        
        dphix_ds= x_mu_sigma*dphix_dx*np.sqrt(2)
        dphiy_ds= y_mu_sigma*dphiy_dy*np.sqrt(2)
        dphiz_ds= z_mu_sigma*dphiz_dz*np.sqrt(2)

        dphix=(phix[:,1:]-phix[:,:-1])
        dphiy=(phiy[:,1:]-phiy[:,:-1])
        dphiz=(phiz[:,1:]-phiz[:,:-1])
        
        ddphix_dx=dphix_dx[:,1:]-dphix_dx[:,:-1]
        ddphiy_dy=dphiy_dy[:,1:]-dphiy_dy[:,:-1]
        ddphiz_dz=dphiz_dz[:,1:]-dphiz_dz[:,:-1]
        
        ddphix_ds=dphix_ds[:,1:]-dphix_ds[:,:-1]
        ddphiy_ds=dphiy_ds[:,1:]-dphiy_ds[:,:-1]
        ddphiz_ds=dphiz_ds[:,1:]-dphiz_ds[:,:-1]

        dsim={}

        dsim['dx']=self.outer_mult(ddphix_dx,dphiy,dphiz)
        dsim['dy']=self.outer_mult(dphix,ddphiy_dy,dphiz)
        dsim['dz']=self.outer_mult(dphix,dphiy,ddphiz_dz)
        dsim['dsx']=self.outer_mult(ddphix_ds,dphiy,dphiz)
        dsim['dsy']=self.outer_mult(dphix,ddphiy_ds,dphiz)
        dsim['dsz']=self.outer_mult(dphix,dphiy,ddphiz_ds)

        for key in dsim:
            dsim[key]=self.fold_padding(dsim[key])
        return dsim

    def dcorr_coef_numpy(self):
        dsim=self.dsim_map()
        dsim=np.array([dsim['dx'],dsim['dy'],dsim['dz'],dsim['dsx'],dsim['dsy'],dsim['dsz']]).transpose(1,0,2,3,4)
        sim=self.sim_map()
        exp=self.experimental_map
        
        #
        num1=np.sum(dsim*exp[None,None,:,:,:],axis=(2,3,4))
        den1=np.sqrt(np.sum(sim**2)) * np.sqrt(np.sum(exp**2))
        
        num2 = np.sum(dsim*sim[None,None,:,:,:],axis=(2,3,4))* np.sum(sim * exp)
        den2 = np.sum(sim**2) * den1
        
        # Final equation
        return ((num1 / den1) - (num2 / den2))[:,:3]
    
    def dcorr_coef(self):
        return dcorr_v3(self.coordinates,self.n_voxels,self.voxel_size,self.sigma, self.experimental_map,self.padding,5)
    def test(self):
        assert np.allclose(self.dsim_map()['dx'],self.dsim_map_numerical()['dx'])
        assert np.allclose(self.dsim_map()['dy'],self.dsim_map_numerical()['dy'])
        assert np.allclose(self.dsim_map()['dz'],self.dsim_map_numerical()['dz'])
        assert np.allclose(self.dcorr_coef(),self.dcorr_coef_numerical())

@vectorize([float64(float64)], nopython=True)
def numba_erf(x):
    return math.erf(x)

@jit(nopython=True)
def substract_and_fold(arr,p):
    darr=arr[:,1:]-arr[:,:-1]
    darr[:, -2*p:-p] += darr[:, :p]
    darr[:, p:2*p]   += darr[:, -p:]
    return darr[:,p:-p]

@jit(nopython=True, parallel=True)
def dcorr_v3(coordinates, n_voxels ,voxel_size ,sigma, experimental_map, padding, multiplier):
    n_dim = coordinates.shape[0]
    i_dim = n_voxels[0]
    j_dim = n_voxels[1]
    k_dim = n_voxels[2]

    voxel_limits_x=np.arange(-padding,n_voxels[0]+1+padding)*voxel_size[0]
    voxel_limits_y=np.arange(-padding,n_voxels[1]+1+padding)*voxel_size[1]
    voxel_limits_z=np.arange(-padding,n_voxels[2]+1+padding)*voxel_size[2]
    
    min_coords = (coordinates - multiplier * sigma)
    max_coords = (coordinates + multiplier * sigma)
    
    limits=np.zeros((coordinates.shape[0],6),dtype=np.int64)
    
    limits[:,0]=np.searchsorted(voxel_limits_x,min_coords[:,0])-1
    limits[:,1]=np.searchsorted(voxel_limits_x,max_coords[:,0])+1
    limits[:,2]=np.searchsorted(voxel_limits_y,min_coords[:,1])-1
    limits[:,3]=np.searchsorted(voxel_limits_y,max_coords[:,1])+1
    limits[:,4]=np.searchsorted(voxel_limits_z,min_coords[:,2])-1
    limits[:,5]=np.searchsorted(voxel_limits_z,max_coords[:,2])+1
    
    sigma=sigma*np.sqrt(2) #(3,)
    x_mu_sigma=np.zeros((n_dim,voxel_limits_x.shape[0]))
    y_mu_sigma=np.zeros((n_dim,voxel_limits_y.shape[0]))
    z_mu_sigma=np.zeros((n_dim,voxel_limits_z.shape[0]))
    for n in prange(n_dim):
        x_mu_sigma[n,:]=(voxel_limits_x-coordinates[n,0])/sigma[0] #(n,x+1+2*p)
        y_mu_sigma[n,:]=(voxel_limits_y-coordinates[n,1])/sigma[1] #(n,x+1+2*p)
        z_mu_sigma[n,:]=(voxel_limits_z-coordinates[n,2])/sigma[2] #(n,x+1+2*p)

    phix=(1+numba_erf(x_mu_sigma))/2 #(n,x+1+2*p)
    phiy=(1+numba_erf(y_mu_sigma))/2 #(n,y+1+2*p)
    phiz=(1+numba_erf(z_mu_sigma))/2 #(n,z+1+2*p)
    
    dphix_dx= -np.exp(-x_mu_sigma**2) / np.sqrt(np.pi) / sigma[0] #(n,x+1+2*p)
    dphiy_dy= -np.exp(-y_mu_sigma**2) / np.sqrt(np.pi) / sigma[1] #(n,y+1+2*p)
    dphiz_dz= -np.exp(-z_mu_sigma**2) / np.sqrt(np.pi) / sigma[2] #(n,z+1+2*p)
    
    dphix_ds= x_mu_sigma*dphix_dx*np.sqrt(2) #(n,x+1+2*p)
    dphiy_ds= y_mu_sigma*dphiy_dy*np.sqrt(2) #(n,y+1+2*p)
    dphiz_ds= z_mu_sigma*dphiz_dz*np.sqrt(2) #(n,z+1+2*p)
    
    dphix=substract_and_fold(phix, padding) #(n,x)
    dphiy=substract_and_fold(phiy, padding) #(n,y)
    dphiz=substract_and_fold(phiz, padding) #(n,z)
    
    ddphix_dx=substract_and_fold(dphix_dx, padding) #(n,x)
    ddphiy_dy=substract_and_fold(dphiy_dy, padding) #(n,y)
    ddphiz_dz=substract_and_fold(dphiz_dz, padding) #(n,z)
    
    ddphix_ds=substract_and_fold(dphix_ds, padding) #(n,x)
    ddphiy_ds=substract_and_fold(dphiy_ds, padding) #(n,y)
    ddphiz_ds=substract_and_fold(dphiz_ds, padding) #(n,z)
    
    exp=experimental_map #(x,y,z)
    
    #Calculate sim
    sim=np.zeros((i_dim,j_dim,k_dim), dtype=np.float64) #(x,y,z)
    for n in prange(n_dim):
        i_min,i_max,j_min,j_max,k_min,k_max=limits[n]
        for i in range(i_min,i_max+1):
            i=(i-padding)%i_dim
            for j in range(j_min,j_max+1):
                j=(j-padding)%j_dim
                for k in range(k_min,k_max+1):
                    k=(k-padding)%k_dim
                    sim[i,j,k]+=dphix[n,i]*dphiy[n,j]*dphiz[n,k]
    
    #Calculate derivatives
    num1 = np.zeros((n_dim,6), dtype=np.float64)
    num2 = np.zeros((n_dim,6), dtype=np.float64)
    for n in prange(n_dim):
        i_min,i_max,j_min,j_max,k_min,k_max=limits[n]
        for i in range(i_min,i_max+1):
            i=(i-padding)%i_dim
            for j in range(j_min,j_max+1):
                j=(j-padding)%j_dim
                for k in range(k_min,k_max+1):
                    k=(k-padding)%k_dim
                    exp_val=exp[i,j,k]
                    sim_val=sim[i,j,k]
                    num1[n,0]+=ddphix_dx[n,i]*dphiy[n,j]*dphiz[n,k]*exp_val
                    num1[n,1]+=dphix[n,i]*ddphiy_dy[n,j]*dphiz[n,k]*exp_val
                    num1[n,2]+=dphix[n,i]*dphiy[n,j]*ddphiz_dz[n,k]*exp_val
                    num1[n,3]+=ddphix_ds[n,i]*dphiy[n,j]*dphiz[n,k]*exp_val
                    num1[n,4]+=dphix[n,i]*ddphiy_ds[n,j]*dphiz[n,k]*exp_val
                    num1[n,5]+=dphix[n,i]*dphiy[n,j]*ddphiz_ds[n,k]*exp_val
                    num2[n,0]+=ddphix_dx[n,i]*dphiy[n,j]*dphiz[n,k]*sim_val
                    num2[n,1]+=dphix[n,i]*ddphiy_dy[n,j]*dphiz[n,k]*sim_val
                    num2[n,2]+=dphix[n,i]*dphiy[n,j]*ddphiz_dz[n,k]*sim_val
                    num2[n,3]+=ddphix_ds[n,i]*dphiy[n,j]*dphiz[n,k]*sim_val
                    num2[n,4]+=dphix[n,i]*ddphiy_ds[n,j]*dphiz[n,k]*sim_val
                    num2[n,5]+=dphix[n,i]*dphiy[n,j]*ddphiz_ds[n,k]*sim_val
    
    num2*=np.sum(sim * exp) #(n,6)
    den1=np.sqrt(np.sum(sim**2)) * np.sqrt(np.sum(exp**2)) #(,)
    den2=np.sum(sim**2) * den1 #(,)
    
    result=((num1 / den1) - (num2 / den2)) #(n,6)
    return result

nx,ny,nz=70,60,50
coordinates=np.random.rand(10,3)*(nx,ny,nz)
experimental_map=np.random.rand(nx,ny,nz)
self=MDFit(coordinates,experimental_map,n_voxels=[nx,ny,nz],voxel_size=[1,1,1],padding=4)
assert np.allclose(self.dcorr_coef_numpy(),self.dcorr_coef_numerical())
assert np.allclose(self.dcorr_coef()[:,:3],self.dcorr_coef_numpy())
assert np.allclose(dcorr_v3(self.coordinates,self.n_voxels,self.voxel_size,self.sigma, self.experimental_map,self.padding,5)[:,:3],self.dcorr_coef_numpy())



