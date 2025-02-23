import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
from functools import partial
import optax
from jaxopt import OptaxSolver

class streamframe:
    """
    Input units must be kpc, kpc/Myr.
    sim_coords is Nx6
    origin_particle is length 6 array
    Output units are deg, mas/yr, km/s
    """
    def __init__(self, sim_coords, origin_particle):
        self.sim_coords = sim_coords
        self.origin_particle = origin_particle
        self.origin_L_sim = jnp.cross(origin_particle[:3], origin_particle[3:])
        
        self.xs_hat = origin_particle[:3]/jnp.sqrt(jnp.sum(origin_particle[:3]**2))
        self.zs_hat = self.origin_L_sim / jnp.sqrt(jnp.sum(self.origin_L_sim**2))
        self.ys_hat = jnp.cross(self.zs_hat, self.xs_hat)
        
        ## This can all be done in a single matmul, but keeping seperated for now...
        ## Get positions and stream-aligned cartesian frame
        self.xs = jnp.sum(sim_coords[:,:3]*self.xs_hat[None,:],axis=1)
        self.ys = jnp.sum(sim_coords[:,:3]*self.ys_hat[None,:],axis=1)
        self.zs = jnp.sum(sim_coords[:,:3]*self.zs_hat[None,:],axis=1)
        self.rs = jnp.sqrt(self.xs**2 + self.ys**2 + self.zs**2)
        
        self.xyz_s = jnp.vstack([self.xs,self.ys,self.zs]).T
        
        
        ## Get velocities in stream-aligned cartesian frame
        ## Make all velocities relative to the origin particle
        DeltaV = sim_coords[:,3:] - origin_particle[None,3:]
        self.vxs = jnp.sum(DeltaV*self.xs_hat[None,:],axis=1)
        self.vys = jnp.sum(DeltaV*self.ys_hat[None,:],axis=1)
        self.vzs = jnp.sum(DeltaV*self.zs_hat[None,:],axis=1)
        
        self.vxyz_s = jnp.vstack([self.vxs,self.vys,self.vzs]).T

        self.coordsfrom_Lvec = self.get_frame_from_Lvec()
        
    @partial(jax.jit,static_argnums=(0))
    def get_frame_from_Lvec(self):
        """
        Compute stream-frame using the angular momentum vector of a single particle
        phi1 be in the local orbital plane of the reference particle, phi2 will be along the L-vector
        """
        phi1 = jnp.arctan2(self.ys,self.xs)
        phi2 = jnp.arcsin(self.zs/self.rs)
        dist = self.rs
        
        rs_hat = self.xyz_s/dist[:,None]
        phi1_hat = -jnp.sin(phi1[:,None])*jnp.cos(phi2[:,None])*self.xs_hat[None,:] + jnp.cos(phi1[:,None])*jnp.cos(phi2[:,None])*self.ys_hat[None,:]
        phi2_hat = -jnp.cos(phi1[:,None])*jnp.sin(phi2[:,None])*self.xs_hat[None,:] - jnp.sin(phi1[:,None])*jnp.sin(phi2[:,None])*self.ys_hat[None,:] + jnp.cos(phi2[:,None])*self.zs_hat[None,:]
        
        
        vr = jnp.sum(self.vxyz_s*rs_hat,axis=1) 
        vr = vr ##- vr.mean() #TODO: should subtract origin particle velocity, not 
        vphi1 = jnp.sum(self.vxyz_s*phi1_hat,axis=1)
        vphi2 = jnp.sum(self.vxyz_s*phi2_hat,axis=1)
        
        pm_phi1 = vphi1/dist #cosphi2*phi1hat
        pm_phi2 = vphi2/dist
        
        rad_per_Myr_to_mas_per_yr = (2.0626481e8)*(1e-6)
        kpc_per_Myr_to_km_per_s = 977.79222
        stream_frame_coords = {'phi1':jnp.rad2deg(phi1), 
                               'phi2':jnp.rad2deg(phi2), 
                               'pm_phi1':pm_phi1*rad_per_Myr_to_mas_per_yr, 
                               'pm_phi2':pm_phi2*rad_per_Myr_to_mas_per_yr, 
                               'r':dist, 
                               'vr':vr*kpc_per_Myr_to_km_per_s}
        return stream_frame_coords

    @partial(jax.jit,static_argnums=(0))
    def compute_rotated_phi12(self, theta_rot, phi1, phi2):
        """
        Rotated the phi12 frame by theta_rot
        theta_rot and phi12 in deg
        """
        theta_rot = jnp.deg2rad(theta_rot)
        phi12_vec = jnp.vstack([phi1, phi2]).T # N x 2
        
        rot_mat = jnp.array([[jnp.cos(theta_rot), -jnp.sin(theta_rot)],
                                [jnp.sin(theta_rot), jnp.cos(theta_rot)]])

        rotated_phi12_vec = jnp.einsum('ij,kj->ki',rot_mat,phi12_vec)
        return dict(phi1=rotated_phi12_vec[:,0], phi2=rotated_phi12_vec[:,1], rot_mat=rot_mat)

    @partial(jax.jit,static_argnums=(0))
    def compute_rotated_coords(self, theta_rot):
        """
        rotate the phi12 and pm12 frame by theta_rot
        theta_rot in deg
        """
        theta_rot = jnp.deg2rad(theta_rot)
        phi1, phi2 = self.coordsfrom_Lvec['phi1'], self.coordsfrom_Lvec['phi2']
        phi12_vec = jnp.vstack([phi1, phi2]).T # N x 2

        pm1, pm2 = self.coordsfrom_Lvec['pm_phi1'], self.coordsfrom_Lvec['pm_phi2']
        pm12_vec = jnp.vstack([pm1, pm2]).T
        
        rot_mat = jnp.array([[jnp.cos(theta_rot), -jnp.sin(theta_rot)],
                                [jnp.sin(theta_rot), jnp.cos(theta_rot)]])

        rotated_phi12_vec = jnp.einsum('ij,kj->ki',rot_mat,phi12_vec)
        rotated_pm_vec = jnp.einsum('ij,kj->ki',rot_mat,pm12_vec)
        return dict(phi1=rotated_phi12_vec[:,0], 
                    phi2=rotated_phi12_vec[:,1], 
                    pm_phi1=rotated_pm_vec[:,0],
                    pm_phi2=rotated_pm_vec[:,1],
                    r= self.coordsfrom_Lvec['r'],
                    vr=self.coordsfrom_Lvec['vr'])


    @partial(jax.jit,static_argnums=(0,2))
    def coordsfrom_fit(self, lr=1e-3, maxiter=2_000):
        """
        Estimate a stream frame by minimizing the squared phi2 values.
        lr: learning rate for optimizer
        maxiter: max number of iterations for the optimizer
        """

        @jax.jit
        def cost_func(theta_rot, phi1, phi2):
            """
            Define a cost-function that encourages a flat stream along phi2 of zero
            theta_rot in deg
            """

            rotated_coords = self.compute_rotated_phi12(theta_rot, phi1, phi2)
            return jnp.sum(rotated_coords['phi2']**2)

       
        def solve(lr, maxiter):
        
            opt = optax.adam(lr)
            solver = OptaxSolver(opt=opt, fun=cost_func, maxiter=maxiter)

            res = solver.run(0.0, 
                            phi1=self.coordsfrom_Lvec['phi1'], 
                            phi2=self.coordsfrom_Lvec['phi2'])
            return res.params

        optimized_rot = solve(lr, maxiter)

        return self.compute_rotated_coords(optimized_rot)


