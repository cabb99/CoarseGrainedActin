import numpy as np
from numba import jit, prange, vectorize, float64
import math
from scipy.special import erf

class MDFit:
    def __init__(self, experimental_map, voxel_size=None, origin=None, dtype=np.float64):
        self.dtype=dtype
        
        # Convert inputs to numpy arrays with specified dtype, handling defaults
        self.experimental_map = np.asarray(experimental_map, dtype=self.dtype)
        if voxel_size is None:
            self.voxel_size = np.array([1, 1, 1], dtype=self.dtype)
        else:
            self.voxel_size = np.asarray(voxel_size, dtype=self.dtype)

        if origin is None:
            self.origin = np.array([0, 0, 0], dtype=self.dtype)
        else:
            self.origin = np.asarray(origin, dtype=self.dtype)

        # Validate the dimensions of inputs
        if self.voxel_size.ndim != 1 or self.voxel_size.size != 3:
            raise ValueError("voxel_size must be a one-dimensional numpy array of size 3.")
        if self.experimental_map.ndim != 3:
            raise ValueError("experimental_map must be a 3D numpy array.")

        
        self.n_voxels = np.array(self.experimental_map.shape)[::-1]
        self.padding = None
        self.voxel_limits = None
        self.coordinates = None
        self.sigma = None
        self.epsilon = None
        self.force = None

    @classmethod
    def from_mrc(cls, mrc_file,cutoff_min=None,cutoff_max=None):
        import mrcfile
        with mrcfile.open(mrc_file) as mrc:
            data = mrc.data
            voxel_size=mrc.voxel_size
            header=mrc.header
            
        if cutoff_min:
            data[data<cutoff_min]=0
        if cutoff_max:
            data[data>cutoff_max]=cutoff_max
        return cls(experimental_map=data.transpose(header['mapc']-1,header['mapr']-1,header['maps']-1),
                   voxel_size=np.array([voxel_size['x'], voxel_size['y'], voxel_size['z']]), 
                   origin=np.array([header['origin']['x'], header['origin']['y'], header['origin']['z']]))
        
    def save_mrc(self, mrc_file, experimental=False):
        import mrcfile
        import datetime
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        data_description = [f"Data generated on {current_date}", "Simulated density map."]

        map_data = self.experimental_map if experimental else self.simulation_map()
        # Assuming mrc is your MRC file object after opening it with mrcfile
        with mrcfile.new(mrc_file, overwrite=True) as mrc:
            mrc.set_data(map_data.astype(np.float32))
            mrc.voxel_size = tuple(self.voxel_size)

            # Setting header values directly can be error-prone due to potential key mismatches or incorrect assignments;
            # instead, use the attributes provided by the mrcfile library when possible.
            mrc.header.mapc, mrc.header.mapr, mrc.header.maps = 1, 2, 3
            mrc.header.mode = 2  # Mode 2 is commonly used for 32-bit floating point data
            mrc.header.mx, mrc.header.my, mrc.header.mz = map_data.shape
            mrc.header.nx, mrc.header.ny, mrc.header.nz = map_data.shape
            mrc.header.nxstart, mrc.header.nystart, mrc.header.nzstart = 0, 0, 0
            mrc.header.cellb = (90.0, 90.0, 90.0)
            
            # Calculate cell dimensions based on voxel size and shape
            mrc.header.cella = (map_data.shape[0] * self.voxel_size[0],
                                map_data.shape[1] * self.voxel_size[1],
                                map_data.shape[2] * self.voxel_size[2])
            

            # Origin
            mrc.header.origin = tuple(self.origin)
            mrc.header.nlabl = len(data_description)
            for i in range(mrc.header.nlabl):
                mrc.header.label[i] = data_description[i].encode('ascii')[:80]

            mrc.update_header_stats()
            mrc.flush()

    def add_force(self,system):
        import openmm
        n_particles=system.getNumParticles()
        
        force = openmm.CustomCompoundBondForce(1,"-k(i,0)*x1-k(i,1)*y1-k(i,2)*z1")
        force_array=np.zeros((n_particles,3))
        force_vectors = openmm.Discrete2DFunction(force_array.shape[0], force_array.shape[1], force_array.T.flatten())
        force.addTabulatedFunction('k', force_vectors)
        force.addPerBondParameter('i')
        for i in range(n_particles):
            force.addBond([i], [i])
        force.setUsesPeriodicBoundaryConditions(True)
        system.addForce(force)
        self.force=force
        return force
    
    def periodic_vectors(self):
        return [self.voxel_size[2]*self.n_voxels[2]/10,0,0],\
               [0,self.voxel_size[1]*self.n_voxels[1]/10,0],\
               [0,0,self.voxel_size[0]*self.n_voxels[0]/10]
    
    def update_force(self,simulation, force=None, force_array=None):
        if force is None:
            force=self.force
        if force_array is None:
            force_array=3200*self.dcorr_coef()[:,:3]
        tabulated_function = self.force.getTabulatedFunction(0)
        params = tabulated_function.getFunctionParameters()
        params[2] = force_array.T.ravel()
        tabulated_function.setFunctionParameters(*params)
        force.updateParametersInContext(simulation.context)
            

    def set_coordinates(self, coordinates, sigma=None, epsilon=None):
        coordinates = np.asarray(coordinates, dtype=self.dtype)
        if sigma is not None:
            sigma = np.asarray(sigma, dtype=self.dtype)
        if epsilon is not None:
            epsilon = np.asarray(epsilon, dtype=self.dtype)

        # Validate shapes
        if coordinates.shape[1] != 3:
            raise ValueError("coordinates must have shape (n, 3)")
        
        if sigma is not None and (sigma.shape[0] != coordinates.shape[0] or sigma.shape[1] != 3):
            raise ValueError("sigma must have shape (n, 3)")
        elif sigma is None and self.sigma is None:
            sigma = np.ones((coordinates.shape[0], 3))
        elif sigma is None and self.sigma is not None:
            sigma = self.sigma
        
        
        if epsilon is not None and epsilon.shape != (coordinates.shape[0],):
            raise ValueError("epsilon must have shape (n,)")
        elif epsilon is None and self.epsilon is None:
            epsilon = np.ones(coordinates.shape[0])
        elif epsilon is None and self.epsilon is not None:
            epsilon = self.epsilon

        self.coordinates = coordinates - self.origin
        self.sigma = sigma
        self.epsilon = epsilon
        self.setup_map(sigma)
        self.fix_bounds()
        
    def setup_map(self, sigma):
        # Assumes sigma has been validated and is available
        self.padding = int(np.ceil(5*np.max(sigma.max(axis=0)/self.voxel_size)))
        self.voxel_limits = [np.arange(-self.padding, self.n_voxels[i] + self.padding + 1) * self.voxel_size[i] 
                             for i in range(3)]


    def fix_bounds(self):
        #Fix the coordinates to be within the bounds of the experimental map
        max_bounds = self.n_voxels * self.voxel_size
        self.coordinates = np.mod(self.coordinates, max_bounds)
    
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

    def simulation_map(self):
        sigma=self.sigma*np.sqrt(2)
        phix=(1+erf((self.voxel_limits[0]-self.coordinates[:,None,0])/sigma[:,None,0]))/2
        phiy=(1+erf((self.voxel_limits[1]-self.coordinates[:,None,1])/sigma[:,None,1]))/2
        phiz=(1+erf((self.voxel_limits[2]-self.coordinates[:,None,2])/sigma[:,None,2]))/2

        dphix=(phix[:,1:]-phix[:,:-1])
        dphiy=(phiy[:,1:]-phiy[:,:-1])
        dphiz=(phiz[:,1:]-phiz[:,:-1])
        
        smap=(dphix[:,None,None,:]*dphiy[:,None,:,None]*dphiz[:,:,None,None]).sum(axis=0)
        
        return self.fold_padding(smap)

    def corr_coef(self):
        simulation_map=self.simulation_map()
        return (simulation_map*self.experimental_map).sum()/np.sqrt((simulation_map**2).sum()*(self.experimental_map**2).sum())

    def dcorr_coef_numerical(self, delta=1e-5):
        num_derivatives = np.zeros((self.coordinates.shape[0],7))
    
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
            for j in range(self.sigma.shape[1]):
                # Perturb coordinates positively
                self.sigma[i, j] += delta
                positive_corr_coef = self.corr_coef()
                
                # Perturb coordinates negatively
                self.sigma[i, j] -= 2*delta
                negative_corr_coef = self.corr_coef()
                
                # Compute numerical derivative
                num_derivatives[i, j+3] = (positive_corr_coef - negative_corr_coef) / (2*delta)
                
                # Reset coordinates to original value
                self.sigma[i, j] += delta
            self.epsilon[i] += delta
            positive_corr_coef = self.corr_coef()
            self.epsilon[i] -= 2*delta 
            negative_corr_coef = self.corr_coef()
            num_derivatives[i, 6] = (positive_corr_coef - negative_corr_coef) / (2*delta)
                
        return num_derivatives

    def fit(self,numerical=False):
        f=1
        for i in range(1000):
            if numerical:
                dx=self.dcorr_coef_numerical()
            else:
                dx=self.dcorr_coef()
            dx=dx[:,:3]
            f=.1/np.abs(dx).max()
            self.coordinates=self.coordinates+f*dx
            for j in range(3): 
                self.coordinates[:, j][self.coordinates[:, j] < 0] += self.voxel_size[j] * self.n_voxels[j]
                self.coordinates[:, j][self.coordinates[:, j] >= self.voxel_size[j] * self.n_voxels[j]] -= self.voxel_size[j] * self.n_voxels[j]
            if i%10==0:
                print(i,self.corr_coef())

    def dsim_map_numerical(self, delta=1e-5):
        num_particles = self.coordinates.shape[0]
        sim_map_shape = self.simulation_map().shape
        derivatives = {
            'dx': np.zeros((num_particles,) + sim_map_shape),
            'dy': np.zeros((num_particles,) + sim_map_shape),
            'dz': np.zeros((num_particles,) + sim_map_shape)
        }
        
        for i in range(num_particles):
            for j, direction in enumerate(['dx', 'dy', 'dz']):
                original_coordinate = self.coordinates[i, j]
        
                # Perturb coordinate in the positive direction
                self.coordinates[i, j] = original_coordinate + delta
                positive_sim_map = self.simulation_map()
        
                # Perturb coordinate in the negative direction
                self.coordinates[i, j] = original_coordinate - delta
                negative_sim_map = self.simulation_map()
        
                # Compute numerical derivative for this particle and direction
                derivatives[direction][i] = (positive_sim_map - negative_sim_map) / (2 * delta)
        
                # Reset coordinate to original value
                self.coordinates[i, j] = original_coordinate
        
        return derivatives

    @staticmethod
    def outer_mult(x,y,z):
        return x[:,None,None,:]*y[:,None,:,None]*z[:,:,None,None]
    
    def dsim_map(self):
        sigma=self.sigma*np.sqrt(2)

        x_mu_sigma=(self.voxel_limits[0]-self.coordinates[:,None,0])/sigma[:,None,0]
        y_mu_sigma=(self.voxel_limits[1]-self.coordinates[:,None,1])/sigma[:,None,1]
        z_mu_sigma=(self.voxel_limits[2]-self.coordinates[:,None,2])/sigma[:,None,2]
        
        phix=(1+erf(x_mu_sigma))/2
        phiy=(1+erf(y_mu_sigma))/2
        phiz=(1+erf(z_mu_sigma))/2
        
        dphix_dx= -np.exp(-x_mu_sigma**2) / np.sqrt(np.pi) / sigma[:,None,0]
        dphiy_dy= -np.exp(-y_mu_sigma**2) / np.sqrt(np.pi) / sigma[:,None,1]
        dphiz_dz= -np.exp(-z_mu_sigma**2) / np.sqrt(np.pi) / sigma[:,None,2]
        
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
        dsim['eps']=self.outer_mult(dphix,dphiy,dphiz)

        for key in dsim:
            dsim[key]=self.fold_padding(dsim[key])
        return dsim

    def dcorr_coef_numpy(self):
        dsim=self.dsim_map()
        dsim=np.array([dsim['dx'],dsim['dy'],dsim['dz'],dsim['dsx'],dsim['dsy'],dsim['dsz'],dsim['eps']]).transpose(1,0,2,3,4)
        sim=self.simulation_map()
        exp=self.experimental_map
        
        #
        num1=np.sum(dsim*exp[None,None,:,:,:],axis=(2,3,4))
        den1=np.sqrt(np.sum(sim**2)) * np.sqrt(np.sum(exp**2))
        
        num2 = np.sum(dsim*sim[None,None,:,:,:],axis=(2,3,4))* np.sum(sim * exp)
        den2 = np.sum(sim**2) * den1
        
        # Final equation
        return ((num1 / den1) - (num2 / den2))
    
    def dcorr_coef(self):
        return dcorr_v3(self.coordinates,self.n_voxels,self.voxel_size,self.sigma, self.epsilon, self.experimental_map,self.padding,5)
    
    def test(self):
        assert np.allclose(self.dsim_map()['dx'],self.dsim_map_numerical()['dx'])
        assert np.allclose(self.dsim_map()['dy'],self.dsim_map_numerical()['dy'])
        assert np.allclose(self.dsim_map()['dz'],self.dsim_map_numerical()['dz'])
        assert np.allclose(self.dcorr_coef()[:,:3],self.dcorr_coef_numerical()[:,:3])
        assert np.allclose(self.dcorr_coef_numpy()[:,:3],self.dcorr_coef_numerical()[:,:3])
        assert np.allclose(self.dcorr_coef()[:,:3],self.dcorr_coef_numpy()[:,:3])
        assert np.allclose(self.dcorr_coef()[:,3:],self.dcorr_coef_numpy()[:,3:],atol=1e-5)
        assert np.allclose(dcorr_v3(self.coordinates,self.n_voxels,self.voxel_size,self.sigma,self.epsilon,self.experimental_map,self.padding,5),self.dcorr_coef_numpy(),atol=1e-5)


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
def dcorr_v3(coordinates, n_voxels ,voxel_size ,sigma, epsilon, experimental_map, padding, multiplier):
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
        x_mu_sigma[n,:]=(voxel_limits_x-coordinates[n,0])/sigma[n,0] #(n,x+1+2*p)
        y_mu_sigma[n,:]=(voxel_limits_y-coordinates[n,1])/sigma[n,1] #(n,x+1+2*p)
        z_mu_sigma[n,:]=(voxel_limits_z-coordinates[n,2])/sigma[n,2] #(n,x+1+2*p)

    phix=(1+numba_erf(x_mu_sigma))/2 #(n,x+1+2*p)
    phiy=(1+numba_erf(y_mu_sigma))/2 #(n,y+1+2*p)
    phiz=(1+numba_erf(z_mu_sigma))/2 #(n,z+1+2*p)
    
    
    dphix_dx= np.zeros((n_dim,voxel_limits_x.shape[0])) #(n,x+1+2*p)
    dphiy_dy= np.zeros((n_dim,voxel_limits_y.shape[0])) #(n,y+1+2*p)
    dphiz_dz= np.zeros((n_dim,voxel_limits_z.shape[0])) #(n,z+1+2*p)
    for n in prange(n_dim):
        dphix_dx[n,:]= -np.exp(-x_mu_sigma[n,:]**2) / np.sqrt(np.pi) / sigma[n,0] #(n,x+1+2*p)
        dphiy_dy[n,:]= -np.exp(-y_mu_sigma[n,:]**2) / np.sqrt(np.pi) / sigma[n,1] #(n,y+1+2*p)
        dphiz_dz[n,:]= -np.exp(-z_mu_sigma[n,:]**2) / np.sqrt(np.pi) / sigma[n,2] #(n,z+1+2*p)
    
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
    
    exp=experimental_map #(z,y,x)
    
    #Calculate sim
    sim=np.zeros((k_dim,j_dim,i_dim), dtype=np.float64) #(z,y,x)
    for n in prange(n_dim):
        eps_n=epsilon[n]
        i_min,i_max,j_min,j_max,k_min,k_max=limits[n]
        for k in range(k_min,k_max+1):
            k=(k-padding)%k_dim
            for j in range(j_min,j_max+1):
                j=(j-padding)%j_dim
                for i in range(i_min,i_max+1):
                    i=(i-padding)%i_dim
                    sim[k,j,i]+=eps_n*dphix[n,i]*dphiy[n,j]*dphiz[n,k]
    
    #Calculate derivatives
    num1 = np.zeros((n_dim,7), dtype=np.float64)
    num2 = np.zeros((n_dim,7), dtype=np.float64)
    for n in prange(n_dim):
        eps_n=epsilon[n]
        i_min,i_max,j_min,j_max,k_min,k_max=limits[n]
        for k in range(k_min,k_max+1):
            k=(k-padding)%k_dim
            for j in range(j_min,j_max+1):
                j=(j-padding)%j_dim
                for i in range(i_min,i_max+1):
                    i=(i-padding)%i_dim
                    exp_val=exp[k,j,i]
                    sim_val=sim[k,j,i]
                    num1[n,0]+=eps_n*ddphix_dx[n,i]*dphiy[n,j]*dphiz[n,k]*exp_val
                    num1[n,1]+=eps_n*dphix[n,i]*ddphiy_dy[n,j]*dphiz[n,k]*exp_val
                    num1[n,2]+=eps_n*dphix[n,i]*dphiy[n,j]*ddphiz_dz[n,k]*exp_val
                    num1[n,3]+=eps_n*ddphix_ds[n,i]*dphiy[n,j]*dphiz[n,k]*exp_val
                    num1[n,4]+=eps_n*dphix[n,i]*ddphiy_ds[n,j]*dphiz[n,k]*exp_val
                    num1[n,5]+=eps_n*dphix[n,i]*dphiy[n,j]*ddphiz_ds[n,k]*exp_val
                    num1[n,6]+=dphix[n,i]*dphiy[n,j]*dphiz[n,k]*exp_val
                    num2[n,0]+=eps_n*ddphix_dx[n,i]*dphiy[n,j]*dphiz[n,k]*sim_val
                    num2[n,1]+=eps_n*dphix[n,i]*ddphiy_dy[n,j]*dphiz[n,k]*sim_val
                    num2[n,2]+=eps_n*dphix[n,i]*dphiy[n,j]*ddphiz_dz[n,k]*sim_val
                    num2[n,3]+=eps_n*ddphix_ds[n,i]*dphiy[n,j]*dphiz[n,k]*sim_val
                    num2[n,4]+=eps_n*dphix[n,i]*ddphiy_ds[n,j]*dphiz[n,k]*sim_val
                    num2[n,5]+=eps_n*dphix[n,i]*dphiy[n,j]*ddphiz_ds[n,k]*sim_val
                    num2[n,6]+=dphix[n,i]*dphiy[n,j]*dphiz[n,k]*sim_val
    
    num2*=np.sum(sim * exp) #(n,7)
    den1=np.sqrt(np.sum(sim**2)) * np.sqrt(np.sum(exp**2)) #(,)
    den2=np.sum(sim**2) * den1 #(,)
    
    result=((num1 / den1) - (num2 / den2)) #(n,7)
    return result


if __name__=='__main__':
    nx,ny,nz=70,60,50
    coordinates=np.random.rand(10,3)*(nx,ny,nz)
    sigma=np.ones(coordinates.shape)
    epsilon=np.ones(coordinates.shape[0])
    experimental_map=np.random.rand(nz,ny,nx)
    self=MDFit(experimental_map,voxel_size=[1,1,1])
    self.set_coordinates(coordinates,sigma,epsilon)
    r1=self.dcorr_coef()
    r2=self.dcorr_coef_numerical()
    self.test()


