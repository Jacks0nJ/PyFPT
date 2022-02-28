"""
Created on Wed Feb 17 15:45:15 2021
    This is to act as the class for a stochastic inflationary model with a 
    scalar field and Hubble parameter. This will thus define the potential,
    have field values and a number of e-foldings. It propagates both 
    
    This can also generate the relevent noise term at a particular point in
    the simulation.
    
    For calculating the conformal time tau, I'm currently using the de Sitter
    approximation. This of course could be taken to a higher approximation in
    the future.
    
    As a general rule, 'vec' refers to any 1 or 2D array, but 'matrix' or 'mat'
    refers only to a 3D array.
    
@author: Joe Jackson

"""
import numpy as np
import stochastic_runge_kutta4methods as rk4
import inflation_functions_e_foldings as cosfuncs
import chaotic_inflation_cython42 as cython_code
#import scipy as sp
#M_pl = 2.435363*10**18 old value
M_pl = 1.0
PI = np.pi#so can just use PI as needed


class Stochastic_Inflation:
    def __init__(self, V, V_p, V_pp, end_cond, a_i, reflection_cond = None,\
                 mu=None):
        self._V = V
        self._V_p = V_p
        self._V_pp = V_pp
        self._end = end_cond
        self._N = 0.0#No e-foldings propagated on initiation
        self._a_i = np.copy(a_i)#This is when inflation starts
        self._reflection_cond = reflection_cond
        if mu != None:
            self._mu = mu
        

        
        
    def V(self, phi):
        V_value = self._V(phi)
        return V_value
        
    def V_prime(self, phi):
        Vp_value = self._V_p(phi)
        return Vp_value
    
    
    def V_pprime(self, phi):
        Vpp_value = self._V_pp(phi)
        return Vpp_value
    
    #Remember we are working with N as the time varible
    def phi_accel_N_sr(self, phi, H, N):
        if H>0:
            _phi_dN = self.phi_dN_sr(phi, H)
        else:
            print('H is: ', H)
            print(r'$\phi$ is:', phi)
            raise ValueError(r'H is: ', H, r'phi is:', phi,\
                             'Non positive Hubble parameter') 
        return _phi_dN
    
    #Remember we are working with N as the time varible
    def phi_accel_N_sr_ind(self, phi, N):
        H = self.hubble_param_sr(phi)
        _phi_dN = self.phi_dN_sr(phi, H)
        return _phi_dN
    
    #Remember we are working with N as the time varible
    def phi_accel_N(self, phi_vec, H, N):
        phi = phi_vec[0]
        phi_N_dif = phi_vec[1]
        if H>0:
            accel = cosfuncs.phi_accel_with_N(self._V_p,phi,phi_N_dif,H)
        else:
            print('H is: ', H)
            print(r'$\phi$ is:', phi)
            raise ValueError(r'H is: ', H, r'phi is:', phi,\
                             'Non positive Hubble parameter') 
        return accel
    
    
    #Just phi as a variable, H is a constraint eqn and is not propagated
    def phi_accel_N_ind(self, phi_vec, N):
        phi = phi_vec[0]
        phi_N_dif = phi_vec[1]
        H = cosfuncs.hubble_param(self._V,phi,phi_N_dif)
        if H>0:
            accel = cosfuncs.phi_accel_with_N(self._V_p,phi,phi_N_dif,H)
        else:
            raise ValueError('Non positive Hubble parameter') 
        return accel
    
    #Remember we are working with N as the time varible
    def hubble_accel_N_sr(self, phi, H):
        _H_dN = cosfuncs.hubble_param_dN(phi,H)
        return _H_dN
    
    #Remember we are working with N as the time varible
    def hubble_accel_N(self, phi_vec, H):
        _H_dN = cosfuncs.hubble_param_dN(phi_vec[1],H)
        return _H_dN
    
    #Function to define all of the accelerations
    def all_accel_N_sr(self, matrix, N):
        accel_vec = np.zeros([1,2])
        accel_vec[0,0] = self.phi_accel_N_sr(matrix[0,0], matrix[0,1], N)
        accel_vec[0,1] = self.hubble_accel_N_sr(accel_vec[0,0], matrix[0,1])
        return accel_vec
    
    #Function to define all of the accelerations
    def all_accel_N(self, matrix, N):
        #Only simulating background field
        if matrix.shape[1] == 1:
            #print(matrix)

            accel_vec = self.phi_accel_N_ind(matrix, N)
        #If we are including H as a variable
        elif matrix.shape[1] == 2:
            H = matrix[1,1]
            accel_vec = np.zeros(2)
            accel_vec[0] = self.phi_accel_N(matrix[:,0], H, N)
            #Remember that H is in the 'velocity' position
            accel_vec[1] = self.hubble_accel_N(matrix[:,0], H)
        #If including different k-modes
        elif matrix.shape[1] > 2:
            #Define phi vector, H etc
            phi_vec = matrix[:,0]
            H = matrix[1,1]
            ks = self._k_modes#Shorthand
            vk_matrix = matrix[:,2:matrix.shape[1]]
            #Pre allocate the acceleration vector
            accel_vec = np.zeros(matrix.shape[1], dtype=complex)
            #We already know the background dynamics and their acceleration
            accel_vec[0] = self.phi_accel_N(phi_vec, H, N)
            #Remember that H is in the 'velocity' position
            accel_vec[1] = self.hubble_accel_N(matrix[:,0], H)
            #Iteratively add the different k modes accelerations
            for k_idx in range(matrix.shape[1]-2):
                #If they exit the deep subhorizon
                k_mode = ks[k_idx]
                accel_vec[2+k_idx] = \
                self.vk_mode_accel_N(phi_vec, vk_matrix[:,k_idx], k_mode, N, H)
        #Any other state is an error 
        else:
             raise ValueError('Unknown \'accelration\' term error')
        
        return accel_vec
    
    
    #This needs to be improved, due to the end surface problem.
    def end_of_inflation_condition(self, matrices, N):
        cond = False
        if matrices[0,0] <= self._phi_end:
            cond = True
        
        return cond
    
        #This needs to be improved, due to the end surface problem.
    def end_of_inflation_condition_epsilon(self, matrices, N):
        phi_dN = matrices[1,0]
        _epsilon = self.hubble_flow_param_one(phi_dN)
        #Default to false
        cond = False
        if _epsilon >= 1:
            cond = True
        
        return cond
    
    #This is incomplete - the phi_dN noise needs to be added
    def noise_sr(self, matrix, N):
        H = matrix[0,1]
        noise = np.zeros([matrix.shape[0], matrix.shape[1]])
        #Including only phi noise term amplitude
        noise[0,0] = H/(2*PI)
        #Now including the noise
        noise = noise*np.random.normal(size=(matrix.shape[0],\
                                             matrix.shape[1]))
        return noise
    
    #This only assumes phi as the propagated variable, returns the amplitude
    def noise_amplitude_sr_ind(self, phi, N):
        H = self.hubble_param_sr(phi)
        #Including only phi noise term
        noise = H/(2*PI)
        return noise

    #This is incomplete - the phi_dN noise needs to be added
    def noise_amplitude_phase_space(self, matrix, N):
        H = matrix[1,1]
        noise = np.zeros([matrix.shape[0], matrix.shape[1]])
        #Including only phi noise term amplitude
        noise[0,0] = H/(2*PI)
        noise[1,0] = H/(2*PI)#Place holder, fix
        return noise


    #This simulates just the background field and H, in the slow roll approx
    def full_sim_stochastic_sr(self, phi_i, N_f, tol, error_type, dN, dN_min):
        #Combining the two variables into one matrix to propagate in rk4 method
        if phi_i.shape[0] == 1:#Checking just single value input
            H_i = self.hubble_param_sr(phi_i)
            initial_matrix = np.zeros([1,2])
            initial_matrix[0,0] = phi_i
            initial_matrix[0,1] = H_i
        else:
            ValueError('Incorrect data input for slow roll stochastic model')
            
        end = self.end_of_inflation_condition
        matrix, N_vec = \
            rk4.full_simulation_stochastic(initial_matrix,\
                                           self.all_accel_N_sr, self.noise_sr,\
                                               self._N,N_f, tol, error_type, end, dt = dN)
        return matrix, N_vec
    
    
    #This simulates just the background field and H, in the slow roll approx.
    #It uses basic Euler step to match the noise implementation and fixed step
    def full_sim_stochastic_sr_ind(self, phi_i, N_f, dN, dN_min=None,\
                                   efficient=False,store = False, steps='RK4'):
        #This includes a not having a reflection conditions
        if efficient==False:
            matrix, N_vec = \
                rk4.full_simulation(phi_i, self.phi_accel_N_sr_ind, N_f,\
                                        dt=dN, dt_min = dN_min, t_i = self._N,\
                                        end_cond=self._end,\
                                        reflect_cond = self._reflection_cond,
                                        noise_vec=self.noise_amplitude_sr_ind,\
                                        store_steps=store,step_method = steps)
            return matrix, N_vec
        #If using the 'efficent code'
        elif efficient==True:
            
            phi_final, N_final = \
                rk4.simulation_stochastic_efficent(phi_i,\
                                                  self.phi_accel_N_sr_ind,\
                                                  self.noise_amplitude_sr_ind,\
                                                  self._N, dN, self._end)
            return phi_final, N_final
    
    
    #This simulates just the background field and H
    def full_sim_stochastic(self, phi_vec_i, H_vec_i, N_f, tol, error_type,\
                            dN, dN_min):
        #Combining the two variables into one matrix to propagate in rk4 method
        if phi_vec_i.ndim == 2 and H_vec_i.ndim == 2:#if a 2d array was input
            initial_matrix = np.hstack((phi_vec_i, H_vec_i))
        elif phi_vec_i.ndim == 1 and H_vec_i.ndim == 1:#Both 1d input
            initial_matrix = np.transpose(np.vstack((phi_vec_i, H_vec_i)))
        elif phi_vec_i.ndim == 1 and H_vec_i.ndim == 2:#If they are different
            initial_matrix = np.transpose(np.vstack((phi_vec_i, H_vec_i[:,0])))
        else:
            initial_matrix = np.transpose(np.vstack((phi_vec_i[:,0], H_vec_i)))
            
        end = self.end_of_inflation_condition
        matrix, N_vec = \
            rk4.full_simulation_stochastic(initial_matrix, self.all_accel_N,\
                                           self.noise_phase_space, self._N,\
                                               N_f, tol, error_type, dN_min,\
                                                   end, False, dN)

        return matrix, N_vec
    

        
    
    #This simulates just the background field and H
    def full_sim(self, phi_vec_i, H_vec_i, N_f, tol, error_type, dN_min):
        #Combining the two variables into one matrix to propagate in rk4 method
        initial_matrix = np.hstack((phi_vec_i, H_vec_i))
        matrix, N_vec = \
            rk4.full_simulation_basic(initial_matrix, self.all_accel_N,\
                                self._N, N_f, tol, error_type, dN_min, 0)
        
        #Update the phi and N of the cosmology
        self._phi_vec = np.copy(matrix[-1,:,0])
        self._H_vec = np.copy(matrix[-1,:,1])
        self._N = np.copy(N_vec[-1,:,:])
        
        return matrix, N_vec
    
    #This simulates just the background field dynamics, H is constrain eqn
    def full_sim_ind(self, N_f, tol, error_type, dN_min):
        phi_matrix, N_vec = \
            rk4.full_simulation_basic(self._phi_vec, self.phi_accel_N_ind,\
                                self._N, N_f, tol, error_type, dN_min, True)
        
        #Update the phi and N of the cosmology
        self._phi_vec = np.copy(phi_matrix[-1,:,:])
        self._N = np.copy(N_vec[-1,:,:])
        
        return phi_matrix, N_vec
    
    def one_adeptive_step(self, tol):
        dN = rk4.time_step_size_estimator(self._phi_vec, self.phi_accel_N_ind,\
                                          self._N, 1000*tol)
        phi_matrix, dN = rk4.adeptive_step(self._phi_vec, dN,\
                                           self.phi_accel_N_ind, self._N, tol)
        
        self._phi_vec = np.copy(phi_matrix)
        self._N = self._N+dN
        
        return phi_matrix, self._N
    
    def power_spectrum_stochastic_sr_ind(self, phi_m, N, num_its, h_frac, dN):
        phi_r = (1+h_frac)*phi_m
        #Left of this point
        phi_l = (1-h_frac)*phi_m
        
        _phi_range = np.array([phi_r, phi_m, phi_l])
        #Pre allocate memory
        different_Ns_vec = np.zeros([num_its,3])
        #Pre allocate memory
        delta_N_squared_vec = np.zeros(3)
        for j, _phi in enumerate(_phi_range):
            for i in range(num_its):
                sim_matrix, N_matrix = \
                    self.full_sim_stochastic_sr_ind(_phi_range[j], 1.5*N, dN)
                #Store the N distribution
                different_Ns_vec[i,j] = N_matrix[-1,0,0]
                
            delta_N_squared_vec[j] =\
                cosfuncs.delta_N_squared(different_Ns_vec[:,j])
            print('Power spectrum calculation is ' + str(j+1) + '/3 complete.')
            
        ps = np.gradient(delta_N_squared_vec,\
                         np.mean(different_Ns_vec, axis=0))
                
        return ps, different_Ns_vec
    
    def power_spectrum_stochastic_sr(self, N, num_its, h_frac, tol, dN):
        #Point of interest
        phi_m = M_pl*np.sqrt(np.array([4*N+2]))
        #Right of this point
        phi_r = (1+h_frac)*phi_m
        #Left of this point
        phi_l = (1-h_frac)*phi_m
        '''
        This needs to be fixed
        '''
        
        _phi_range = np.array([phi_r, phi_m, phi_l])[:,0]
        #Pre allocate memory
        N_mean_vec = np.zeros(3)
        #Pre allocate memory
        delta_N_squared_vec = np.zeros(3)
        j=0
        for _phi in _phi_range:
            Ns = np.zeros(num_its)
            for i in range(num_its):
                sim_matrix, N_matrix = \
                    self.full_sim_stochastic_sr(_phi_range[j], 2*N, tol,\
                                             'fractional', dN, 10**(-8))
                #Store the N distribution
                Ns[i] = N_matrix[-1,0,0]
                
            N_mean_vec[j] = cosfuncs.average_N(Ns)
            delta_N_squared_vec[j] = cosfuncs.delta_N_squared(Ns)
            j += 1
            print('Power spectrum calculation is ' + str(j) + '/3 complete.')
            
        ps = np.gradient(delta_N_squared_vec, N_mean_vec)
                
        return ps, N_mean_vec
        
    def power_spectrum_stochastic(self, N, num_its, h_frac, tol, dN):
        #Point of interest
        phi_m = M_pl*np.sqrt(np.array([4*N+2]))
        #Right of this point
        phi_r = (1+h_frac)*phi_m
        #Left of this point
        phi_l = (1-h_frac)*phi_m
        '''
        This needs to be fixed
        '''
        
        _phi_range = np.array([phi_r, phi_m, phi_l])[:,0]
        #Pre allocate memory
        N_mean_vec = np.zeros(3)
        #Pre allocate memory
        delta_N_squared_vec = np.zeros(3)
        j=0
        for _phi in _phi_range:
            Ns = np.zeros(num_its)
            for i in range(num_its):
                #Setting the intial conditions
                _initial_phi = np.array([_phi_range[j],\
                                         self.phi_dN_sr_ind(_phi_range[j])])
                print(_initial_phi)
                _H = cosfuncs.hubble_param(self._V, _initial_phi[0],\
                                           _initial_phi[1])
                _H_vec = np.array([0*_H, _H])
                sim_matrix, N_matrix = \
                    self.full_sim_stochastic(_initial_phi,_H_vec, 2*N, tol,\
                                             'fractional', dN, 10**(-8))
                #Store the N distribution
                Ns[i] = N_matrix[-1,0,0]
                
            N_mean_vec[j] = cosfuncs.average_N(Ns)
            delta_N_squared_vec[j] = cosfuncs.delta_N_squared(Ns)
            j += 1
            print('Power spectrum calculation is ' + str(j) + '/3 complete.')
            
        ps = np.gradient(delta_N_squared_vec, N_mean_vec)
                
        return ps, N_mean_vec
    #Uses the cython code to calculate the power spectrum, and does this for
    #a specified range of phi values. Requires a double as arguments
    def cython_power_spectrum_stochastic(self, phi_values, num_sims, dN):

        num_samples = len(phi_values)
        #Pre allocate memory
        N_mean_vec = np.zeros(num_samples)
        delta_N_squared_vec = np.zeros(num_samples)
        Ns_distribution = np.zeros((num_sims, num_samples))
        print('starting')
        for j, _phi in enumerate(phi_values):
            Ns_distribution[:,j] =\
                np.array(cython_code.many_simulations(_phi, 2**0.5,\
                                                      0.0, 200,  dN, num_sims))   
            N_mean_vec[j] = np.mean(Ns_distribution[:,j])
            delta_N_squared_vec [j] = np.var(Ns_distribution[:,j])
            print('Data point '+str(j+1)+'/'+str(num_samples)+' complete')
        ps = np.gradient(delta_N_squared_vec, N_mean_vec)

        return ps, Ns_distribution

    
    
    '''
    Using the previous definitions of these functions
    '''
    
    
    
    def hubble_param_ind(self, phi, phi_dN):
        H = cosfuncs.hubble_param(self._V, phi, phi_dN)
        return H
    
    def hubble_param_sr(self, phi):
        H = cosfuncs.hubble_param_sr(self._V, phi)
        return H
    
    def phi_dN_sr(self, phi, H):
        phi_dN = cosfuncs.phi_dN_sr(self._V_p, phi, H)
        return phi_dN
    
    def phi_dN_sr_ind(self, phi):
        phi_dN = cosfuncs.phi_dN_sr_ind(self._V, self._V_p, phi)
        return phi_dN
    
    def hubble_param_dN(self, phi_dN, H):
        H_dN = cosfuncs.hubble_param_dN(phi_dN, H)
        return H_dN
    
    def density_ind(self, phi, phi_dN):
        rho = cosfuncs.density_ind(self._V, phi, phi_dN)
        return rho
    
    def density(self, phi, phi_dN, H):
        rho = cosfuncs.density(self._V, phi, phi_dN, H)
        return rho
    
    def pressure_ind(self, phi, phi_dN):
        p = cosfuncs.pressure_ind(self._V, phi, phi_dN)
        return p
    
    def pressure(self, phi, phi_dN, H):
        p = cosfuncs.pressure(self._V, phi, phi_dN, H)
        return p
    
    def eos_param_ind(self, phi, phi_dN):
        w = cosfuncs.eos_param_ind(self._V, phi, phi_dN)
        return w
    
    def eos_param(self, phi, phi_dN, H):
        w = cosfuncs.eos_param(self._V, phi, phi_dN, H)
        return w
    
    def hubble_flow_param_one(self, phi_dN):
        epsilon_1 = cosfuncs.hubble_flow_param_one(phi_dN)
        return epsilon_1
    
    def hubble_flow_param_two_ind(self, phi, phi_dN):
        epsilon_2 = \
            cosfuncs.hubble_flow_param_two_ind(self._V, self._V_p, phi, phi_dN)
        return epsilon_2
    
    def hubble_flow_param_two(self, phi, phi_dN, H):
        epsilon_2 = cosfuncs.hubble_flow_param_two(self._V, self._V_p, phi,\
                                                   phi_dN, H)
        return epsilon_2
    
    def f_param_ind(self, phi, phi_dN):
        f = cosfuncs.f_param_ind(self._V, self._V_p,phi, phi_dN)
        return f
    
    def f_param(self, phi, phi_dN, H):
        f = cosfuncs.f_param(self._V_p, phi, phi_dN, H)
        return f
    
    def mu_param_ind(self, phi, phi_dN):
        mu = cosfuncs.mu_param_ind(self._V, self._V_pp,phi, phi_dN)
        return mu
    
    def mu_param(self, phi, H):
        mu = cosfuncs.mu_param(self._V_pp,phi,H)
        return mu
    
    def eta_param(self,  phi, phi_dN, H):
        eta = cosfuncs.eta_param(self._V_p,phi,phi_dN,H)
        return eta
    
    def z_prime_prime_ind(self, phi, phi_dN, N):
        f = cosfuncs.f_param_ind(self._V,self._V_p,phi,phi_dN)
        e = cosfuncs.hubble_flow_param_one(phi_dN)
        mu = cosfuncs.mu_param_ind(self._V, self._V_pp, phi, phi_dN)
        H = cosfuncs.hubble_param(self._V, phi, phi_dN)
        a = self._a_i*np.exp(N)
        _z_pp_by_z= (2+5*e-3*mu-12*f*e+2*e**2)*(a*H)**2
        return _z_pp_by_z
    
    def z_prime_prime(self, phi, phi_dN, N, H):
        f = cosfuncs.f_param(self._V_p, phi, phi_dN, H)
        e = cosfuncs.hubble_flow_param_one(phi_dN)
        mu = cosfuncs.mu_param(self._V_pp, phi, H)
        H = cosfuncs.hubble_param(self._V, phi, phi_dN)
        a = self._a_i*np.exp(N)
        _z_pp_by_z= (2+5*e-3*mu-12*f*e+2*e**2)*(a*H)**2
        return _z_pp_by_z
    
    def z_prime_prime_by_aH_ind(self, phi, phi_dN):
        f = cosfuncs.f_param_ind(self._V,self._V_p,phi,phi_dN)
        e = cosfuncs.hubble_flow_param_one(phi_dN)
        mu = cosfuncs.mu_param_ind(self._V, self._V_pp, phi, phi_dN)
        _z_pp_by_aH = 2+5*e-3*mu-12*f*e+2*e**2
        return _z_pp_by_aH

    def z_prime_prime_by_aH(self, phi, phi_dN, H):
        f = cosfuncs.f_param(self._V_p, phi, phi_dN, H)
        e = cosfuncs.hubble_flow_param_one(phi_dN)
        mu = cosfuncs.mu_param(self._V_pp, phi, H)
        _z_pp_by_aH = 2+5*e-3*mu-12*f*e+2*e**2
        return _z_pp_by_aH
    
    def conformal_time_ind(self, phi, phi_dN, N):
        #This is the de Sitter approx
        tau = cosfuncs.conformal_time_ind(self._V, phi, phi_dN, N, self._a_i)
        return tau
    
    def conformal_time(self, N, H):
        #This is the de Sitter approx
        a = self._a_i*np.exp(N)
        tau = cosfuncs.conformal_time(H, a)
        return tau
    
    #This simply gives a Bunch Davies vacuum for some k-mode
    def bunch_davies_vacuum(self, tau, k):
        bdv = cosfuncs.bunch_davies_vacuum(tau, k)
        return bdv
        
    def sr_power_spectrum_potential(self, phi):
        P = cosfuncs.sr_power_spectrum_potential(self._V, self._V_p, phi)
        return P
    
    def sr_power_spectrum_next_order(self, phi, phi_dN, H):
        _ps = cosfuncs.sr_power_spectrum_next_order(self._V, self._V_p, phi,\
                                                    phi_dN, H)
        return _ps
    
    def sr_spectral_index(self,phi,phi_dN,H):
       n_s = cosfuncs.sr_spectral_index(self._V_p,phi,phi_dN,H)
       return n_s
   
    ##Now using the stochastic functions of cosfuncs
    
    
    def classicality_criterion(self, phi_int):
        eta = cosfuncs.classicality_criterion(self._V, self._V_p,\
                                           self._V_pp, phi_int)
        return eta
    
    
    def mean_N_sto_limit(self, phi_int, phi_end):
        N_mean = cosfuncs.mean_N_sto_limit(self._V, self._V_p,\
                                           self._V_pp,phi_int, phi_end)
        return N_mean
    
    def delta_N_squared_sto_limit(self, phi_int, phi_end):
        d_N_squared = \
            cosfuncs.delta_N_squared_sto_limit(self._V, self._V_p, self._V_pp,\
                                               phi_int, phi_end)
        return d_N_squared
    
    def skewness_N_sto_limit(self, phi_int, phi_end):
        skew_value = \
            cosfuncs.skewness_N_sto_limit(self._V, self._V_p, self._V_pp,\
                                               phi_int, phi_end)
        return skew_value
    
    def kurtosis_N_sto_limit(self, phi_int, phi_end, fisher = True):
        kurt_value = \
            cosfuncs.kurtosis_N_sto_limit(self._V, self._V_p, self._V_pp,\
                                               phi_int, phi_end, Fisher=fisher)
        return kurt_value
    
    def power_spectrum_sto_limit(self, phi_int):
        ps = cosfuncs.power_spectrum_sto_limit(self._V, self._V_p,\
                                           self._V_pp, phi_int)
        return ps
    
    #Quantum well equations
    
    #This is only true for a constant potential
    def quantum_diffusion_x(phi, phi_end, delta_phi_well):
        return np.divide(phi - phi_end, delta_phi_well)
    
    def quantum_diffusion_mean_N(self, x):
        return cosfuncs.quantum_diffusion_mean_N(self._mu, x)
    
    def quantum_diffusion_var_N(self, x):
        return cosfuncs.quantum_diffusion_var_N(self._mu, x)
    
    #This is only true for a constant potential
    #This version starts to break down for large N
    def quantum_diffusion_N_probability_dist_alt(self, N, x, n):
        the_sum = 0.0
        for i in range(n):
            first_term = 2*(i+1)-x
            second_term =2*i+x
            expo = -np.divide(self._mu**2, 4*N)
            the_sum += ((-1)**i)*( first_term*np.exp(expo*first_term**2)\
                                  + second_term*np.exp(expo*second_term**2) )
                
        answer = np.divide(self._mu, 2*(PI**0.5)*(N**1.5))*the_sum
        return answer
    
    #This is only true for a constant potential
    def quantum_diffusion_N_probability_dist(self, N, x, n):
        return cosfuncs.quantum_diffusion_N_probability_dist_alt2(N, x,\
                                                                  self._mu, n)
    
    
    #This is only true for a constant potential
    def quantum_diffusion_N_probability_dist_old(self, N, x, n):
        the_sum = 0.0
        for i in range(n):
            first_term = 2*(i+1)-x
            second_term =2*i+x
            expo = -np.divide(self._mu**2, 4*N)
            the_sum += ((-1)**i)*( first_term*np.exp(expo*first_term**2)\
                                  + second_term*np.exp(expo*second_term**2) )
                
        answer = np.divide(self._mu, 2*(PI**0.5)*(N**1.5))*the_sum
        return answer


    