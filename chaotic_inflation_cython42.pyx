"""
Created on 19/03/21

    Firstly this is Cython file for optimization.
    
    This is a more basic version of my main Cython code built
    for efficency.
    
@author: Joe Jackson
"""
import random
#This model's set up
cdef double M_PL = 1.0
cdef double m = 0.01*M_PL
cdef double PI = 3.1415926535897931159979634685
cdef double phi_end = (2**0.5)*M_PL
cdef double phi_r = 300.0*M_PL
cdef double e = 2.718281828459045


#Defining the potential



cdef double V(double phi, double mass = m):
    return 0.5*(phi*mass)**2

cdef double V_p(double phi, double mass = m):
    return phi*(mass**2)

cdef double V_pp(double phi, double mass = m):
    return (mass**2)

cdef int end_condition(double phi, double N):
    if phi>phi_end:
        return 0
    elif phi<=phi_end:
        return 1
    
cdef int reflect_condition(double phi, double N):
    return 0

cdef double reflection(phi, phi_reflect = phi_r):
    return 2*phi_reflect - phi
        


#Defining the more general functions

cdef double hubble_sr(double phi):
    cdef double H = (V(phi)/(3*M_PL**2))**0.5
    return H

cdef double phi_dN_sr(double phi):
    return -V_p(phi)/(3*hubble_sr(phi)**2)

cdef double hubble_dN_sr(double phi):       
    return -0.5*hubble_sr(phi)*(phi_dN_sr(phi)/M_PL)**2

cdef double diffusion_term(double phi, double H):
    return H/(2*PI)

cdef double diffusion_term_dphi_sr(double phi, double H):
    return V_p(phi)/(12*PI*(M_PL**2)*H)

cdef double diffusion_term_ddphi_sr(double phi, double H):
    return V_pp(phi)/(12*PI*(M_PL**2)*H) - (V_p(phi)**2)/(72*PI*(M_PL**4)*H**3)

#This assumes SR
cdef double drift_term_sr(double phi, double H):
    return (-V_p(phi)/(3*H**2))
    
#This assumes slow-roll approx
cdef double drift_term_dphi_sr(double phi, double H):
    return (-V_pp(phi)/(3*H**2)) + (V_p(phi)/(3*M_PL*H**2))**2

cpdef double get_mass():
    return m

#See (5) of the intro to Kloeden and Platen 1999
#dW is the Weiner step, normally random Gaussian x dN^1/2
cdef double euler_maruyama(double phi, double N, double dN,\
                           double dW):
    cdef double H = hubble_sr(phi)
    return phi + drift_term_sr(phi, H)*dN + diffusion_term(phi, H)*dW
        
#See (22) of the intro to Kloeden and Platen 1999   
#dW is the Weiner step, normally random Gaussian x dN^1/2
cdef double stochastic_rk(double phi, double N, double dN, double sqrt_dN,\
                          double dW):
    cdef double H = hubble_sr(phi)
    cdef double diff = diffusion_term(phi, H)
    return phi + drift_term_sr(phi, H)*dN + diff*dW +\
        0.5*(diffusion_term(phi+diff*sqrt_dN,hubble_sr(phi+diff*sqrt_dN))-diff)*\
        (dW**2-dN)/sqrt_dN
        
#See A. J. Roberts 2012	arXiv:1210.0933
cdef double improved_euler(double phi, double N, double dN, double sqrt_dN,\
                           double dW):
    cdef double H = hubble_sr(phi)
    #Randomly +/- as required by algoritham
    cdef double s_n = random.randrange(-1,2,2)
    cdef double k_1 = drift_term_sr(phi, H)*dN +\
        diffusion_term(phi, H)*(dW-s_n*sqrt_dN)
    cdef double phi_2 = phi + k_1
    #Now calculating second part of the step
    cdef double H_2 = hubble_sr(phi_2)
    cdef double k_2 = drift_term_sr(phi_2, H_2)*dN +\
        diffusion_term(phi_2, H_2)*(dW+s_n*sqrt_dN)
    return phi + (k_1+k_2)/2

#Kloeden and Platen, strong order 1
#See (16) of the intro to Kloeden and Platen 1999
#dW is the Weiner step, normally random Gaussian x dN^1/2
cdef double milstein(double phi, double N, double dN,\
                          double dW):
    cdef double H = hubble_sr(phi)
    cdef double diff = diffusion_term(phi, H)
    return phi + drift_term_sr(phi, H)*dN + diff*dW\
        + 0.5*diff*diffusion_term_dphi_sr(phi, H)*(dW**2-dN)
        
        
#Weak converrgance order 2
#See (29) of the intro to Kloeden and Platen 1999
#dW is the Weiner step, normally random Gaussian x dN^1/2
cdef double milstein_taylay(double phi, double N, double dN, double sqrt_dN,\
                            double dW):
    cdef double H, dZ, a, a_p, b, b_p, b_pp, phi_new
    H = hubble_sr(phi)
    dZ = 0.5*dW*dN
    a = drift_term_sr(phi, H)
    a_p = drift_term_dphi_sr(phi, H)
    b = diffusion_term(phi, H)
    b_p = diffusion_term_dphi_sr(phi, H)
    b_pp = diffusion_term_ddphi_sr(phi, H)
    
    phi_new = phi + a*dN + b*dW + 0.5*b*b_p*(dW**2-dN) + b*a_p*dZ +\
        0.5*(a*a_p + 0.5*a_p*b**2)*dN**2 + (a*b_p+0.5*b_pp*b**2)*(dW*dN-dZ)
    return phi_new

#This is the discreate version of calculating the bias variable, which uses
#the discreat, see Esq. (18) and (19) of arXiv:nucl-th/9809075v1
cdef double importance_sampling_variable_A_step(double A, double D,\
                                                double phi_step, double v,\
                                                    double delta_v, double dN):
    return A+delta_v*(phi_step-v*dN-0.5*delta_v*dN)/D

#This is the N derivative for the importance sampling variable w, as defined
# in Eq. (33) of arXiv:nucl-th/9809075v1. This is used to find the bias for
#an importance sampled path. I used their notation here for generality.
cdef double importance_sampling_variable_w_step(double w, double delta_v,\
                                            double D_sqrt, double dN, double dW):
    return w-w*delta_v*(delta_v*dN+2*D_sqrt*dN)/(2*D_sqrt**2)

cdef list euler_maruyama_importance_sampling(double phi, double A, double N,\
                                                double dN, double sqrt_dN,\
                                                    double dW, double bias):
    cdef double H, a_orig, a, b, phi_new, A_new
    H = hubble_sr(phi)
    a_orig = drift_term_sr(phi, H)
    a = a_orig*(1-bias)
    b = diffusion_term(phi, H)
    
    #Calculation
    phi_new = phi + a*dN + b*dW
    #Just an Euler step in the importance sampling variable
    A_new = importance_sampling_variable_A_step(A, b**2, phi_new-phi, a_orig,\
                                               -bias*a_orig, dN)
    return [phi_new, A_new]

cdef list euler_maruyama_importance_sampling_diffusion_bias(double phi, double A, double N,\
                                                double dN, double sqrt_dN,\
                                                    double dW, double bias):
    cdef double H, a_orig, a, b, phi_new, A_new
    H = hubble_sr(phi)
    a_orig = drift_term_sr(phi, H)
    b = diffusion_term(phi, H)
    a = a_orig+bias*b
    
    #Calculation
    phi_new = phi + a*dN + b*dW
    #Just an Euler step in the importance sampling variable
    A_new = importance_sampling_variable_A_step(A, b**2, phi_new-phi, a_orig,\
                                                bias*b, dN)
    return [phi_new, A_new]
    
#This method is only applicable when doing importance sampling
#Weak converrgance order 2
#See (29) of the intro to Kloeden and Platen 1999
#dW is the Weiner step, normally random Gaussian x dN^1/2
cdef list milstein_taylay_importance_sampling(double phi, double A, double N,\
                                                double dN, double sqrt_dN,\
                                                    double dW, double bias):
    cdef double H, dZ, a_orig, a, a_p, b, b_p, b_pp, phi_new, A_new
    H = hubble_sr(phi)
    dZ = 0.5*dW*dN
    a_orig = drift_term_sr(phi, H)
    a = a_orig*(1-bias)
    a_p = drift_term_dphi_sr(phi, H)*(1-bias)
    b = diffusion_term(phi, H)
    b_p = diffusion_term_dphi_sr(phi, H)
    b_pp = diffusion_term_ddphi_sr(phi, H)
    
    #Calculation
    phi_new = phi + a*dN + b*dW + 0.5*b*b_p*(dW**2-dN) + b*a_p*dZ +\
        0.5*(a*a_p + 0.5*a_p*b**2)*dN**2 + (a*b_p+0.5*b_pp*b**2)*(dW*dN-dZ)
    #Just an Euler step in the importance sampling variable
    A_new = importance_sampling_variable_A_step(A, b**2, phi_new-phi, a_orig,\
                                               -bias*a_orig, dN)
    return  [phi_new, A_new]





cdef double simulation(double phi_i, double phi_end,\
                       double N_i, double N_f, double dN):
    cdef double N, dN_sqrt, phi, noise_amp, dist_end_inflation, dW
    cdef list step_results
    cdef int reduced_step = 0
    dN_sqrt = dN**0.5
    N = N_i
    phi = phi_i
    dist_end_inflation = 0.0
    while N<N_f:
        #Define the Wiener step
        dW = random.gauss(0.0,dN_sqrt)
        phi = milstein_taylay(phi, N, dN, dN_sqrt, dW)
        noise_amp = diffusion_term(phi, hubble_sr(phi))*dN_sqrt
        N += dN
        if end_condition(phi, N) == 1:#Using 1/0 for True/False
            #print('stopped at: ' + str(N))
            break
        elif end_condition(phi+3*noise_amp, N) == 1\
            and reduced_step == 0:
            dN = 0.001*dN
            dN_sqrt = dN**0.5
            dist_end_inflation = 3*noise_amp
            reduced_step = 1
        elif end_condition(phi-3*noise_amp, N) == 1\
            and reduced_step == 0:
            dN = 0.001*dN
            dN_sqrt = dN**0.5
            dist_end_inflation = -3*noise_amp
            reduced_step = 1
            #So it has exited end region
        elif end_condition(phi+dist_end_inflation, N) == 0\
            and reduced_step == 1:
            dN = 1000*dN#Go back to original step size
            dN_sqrt = dN**0.5#Remember to update the root
            reduced_step = 0
        
    return N

#A let's us calculate the bias w=e^-A is the bias, which propagated along with 
#the importance sample path.
#See Eq. (33) of arXiv:nucl-th/9809075v1 for more info
cdef list simulation_importance_sampling(double phi_i, double phi_end,\
                       double N_i, double N_f, double dN, double bias):
    cdef double N, dN_sqrt, phi, noise_amp, dist_end_inflation, dW, A
    cdef list step_results
    cdef int reduced_step = 0
    dN_sqrt = dN**0.5
    N = N_i
    phi = phi_i
    dist_end_inflation = 0.0
    A = 0.0
    while N<N_f:
        #Define the Wiener step
        dW = random.gauss(0.0,dN_sqrt)
        step_results = euler_maruyama_importance_sampling_diffusion_bias(phi, A, N, dN,\
                                                           dN_sqrt, dW, bias)
        phi = step_results[0]
        A = step_results[1]
        noise_amp = diffusion_term(phi, hubble_sr(phi))*dN_sqrt
        N += dN
        if end_condition(phi, N) == 1:#Using 1/0 for True/False
            #print('stopped at: ' + str(N))
            break
        elif end_condition(phi+3*noise_amp, N) == 1\
            and reduced_step == 0:
            dN = 0.001*dN
            dN_sqrt = dN**0.5
            dist_end_inflation = 3*noise_amp
            reduced_step = 1
        elif end_condition(phi-3*noise_amp, N) == 1\
            and reduced_step == 0:
            dN = 0.001*dN
            dN_sqrt = dN**0.5
            dist_end_inflation = -3*noise_amp
            reduced_step = 1
            #So it has exited end region
        elif end_condition(phi+dist_end_inflation, N) == 0\
            and reduced_step == 1:
            dN = 1000*dN#Go back to original step size
            dN_sqrt = dN**0.5#Remember to update the root
            reduced_step = 0
        
    return [N, e**(-A)]

    
cpdef many_simulations(double phi_i, double phi_end,\
                     double N_i, double N_f, double dN,\
                     int num_sims):
    #preallocate memory
    N_dist =\
        [simulation(phi_i, phi_end, N_i, N_f, dN) for i in range(num_sims)]
    return N_dist

cpdef many_simulations_importance_sampling(double phi_i, double phi_end,\
                     double N_i, double N_f, double dN, double bias,\
                     int num_sims, str wind_type = ''):
    #preallocate memory
    sim_results =\
        [simulation_importance_sampling(phi_i, phi_end, N_i, N_f, dN, bias)\
         for i in range(num_sims)]
    #Correctly slicing the list with list comprehension
    Ns, ws = [[sim_results[i][j] for i in range(num_sims)] for j in range(2)]
    return Ns, ws


    

    
