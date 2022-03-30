'''
Created on 26/04/21

    This is my refined Cython code, currently in development. The idea this 
    will be more general, i.e. can take any slow roll potential and compute
    the importance sampling case. This requires both a relfective and absorbing
    surface.
    
    This assumes the evolution of the potential is right ot left in phi.
    
    I've added code which allows the bias to be passed as a function, such that
    it can take any form.
    
@author: Joe Jackson
'''
import random
#This model's set up
cdef double M_PL = 1.0
cdef double PI = 3.1415926535897931159979634685
cdef double e = 2.718281828459045



cdef int end_condition(double phi, double N, double phi_end):
    if phi>phi_end:
        return 0
    elif phi<=phi_end:
        return 1

#phi_end2 is the right hand absorbing surface, phi_end1 the left
cdef int double_end_condition(double phi, double N, double phi_end1,\
                              double phi_end2):
    if phi<=phi_end1:
        return 1
    if phi>=phi_end2:
        return 1
    else:
        return 0
    

cdef int reflect_condition(double phi, double N, double phi_r):
    if phi<phi_r:
        return 0
    elif phi>=phi_r:
        return 1

cdef double reflection(double phi, double phi_r):
    return 2*phi_r - phi
        

#Defining the more general functions

cdef double hubble_sr(double phi, V):
    cdef double H = (V(phi)/(3*M_PL**2))**0.5
    return H

cdef double phi_dN_sr(double phi, double H, V_p):
    return -V_p(phi)/(3*H**2)

cdef double hubble_dN_sr(double phi, V, V_p):       
    return -0.5*hubble_sr(phi, V)*(phi_dN_sr(phi, hubble_sr(phi, V), V_p)/M_PL)**2

cdef double diffusion_term(double phi, double H):
    return H/(2*PI)

cdef double diffusion_term_dphi_sr(double phi, double H, V_p):
    return V_p(phi)/(12*PI*(M_PL**2)*H)

cdef double diffusion_term_ddphi_sr(double phi, double H, V_p, V_pp):
    return V_pp(phi)/(12*PI*(M_PL**2)*H) - (V_p(phi)**2)/(72*PI*(M_PL**4)*H**3)

#This assumes SR
cdef double drift_term_sr(double phi, double H, V_p):
    return (-V_p(phi)/(3*H**2))
    
#This assumes slow-roll approx
cdef double drift_term_dphi_sr(double phi, double H, V_p, V_pp):
    return (-V_pp(phi)/(3*H**2)) + (V_p(phi)/(3*M_PL*H**2))**2

#See (5) of the intro to Kloeden and Platen 1999
#dW is the Weiner step, normally random Gaussian x dN^1/2
cdef double euler_maruyama(double phi, double N, double dN,\
                           double dW, V, V_p, V_pp):
    cdef double H = hubble_sr(phi, V)
    return phi + drift_term_sr(phi, H, V_p)*dN + diffusion_term(phi, H)*dW
        
#See (22) of the intro to Kloeden and Platen 1999   
#dW is the Weiner step, normally random Gaussian x dN^1/2
cdef double stochastic_rk(double phi, double N, double dN, double sqrt_dN,\
                          double dW, V, V_p, V_pp):
    cdef double H = hubble_sr(phi, V)
    cdef double diff = diffusion_term(phi, H)
    return phi + drift_term_sr(phi, H, V_p)*dN + diff*dW +\
        0.5*(diffusion_term(phi+diff*sqrt_dN,hubble_sr(phi+diff*sqrt_dN, V))-diff)*\
        (dW**2-dN)/sqrt_dN
        
#See A. J. Roberts 2012	arXiv:1210.0933
cdef double improved_euler(double phi, double N, double dN, double sqrt_dN,\
                           double dW, V, V_p, V_pp):
    cdef double H = hubble_sr(phi, V)
    #Randomly +/- as required by algoritham
    cdef double s_n = random.randrange(-1,2,2)
    cdef double k_1 = drift_term_sr(phi, H, V_p)*dN +\
        diffusion_term(phi, H)*(dW-s_n*sqrt_dN)
    cdef double phi_2 = phi + k_1
    #Now calculating second part of the step
    cdef double H_2 = hubble_sr(phi_2, V)
    cdef double k_2 = drift_term_sr(phi_2, H_2, V_p)*dN +\
        diffusion_term(phi_2, H_2)*(dW+s_n*sqrt_dN)
    return phi + (k_1+k_2)/2

#Kloeden and Platen, strong order 1
#See (16) of the intro to Kloeden and Platen 1999
#dW is the Weiner step, normally random Gaussian x dN^1/2
cdef double milstein(double phi, double N, double dN,\
                          double dW, V, V_p, V_pp):
    cdef double H = hubble_sr(phi, V)
    cdef double diff = diffusion_term(phi, H)
    return phi + drift_term_sr(phi, H, V_p)*dN + diff*dW\
        + 0.5*diff*diffusion_term_dphi_sr(phi, H, V_p)*(dW**2-dN)
        
        
#Weak converrgance order 2
#See (29) of the intro to Kloeden and Platen 1999
#dW is the Weiner step, normally random Gaussian x dN^1/2
cdef double milstein_taylay(double phi, double N, double dN, double sqrt_dN,\
                            double dW, V, V_p, V_pp):
    cdef double H, dZ, a, a_p, b, b_p, b_pp, phi_new
    H = hubble_sr(phi, V)
    dZ = 0.5*dW*dN
    a = drift_term_sr(phi, H, V_p)
    a_p = drift_term_dphi_sr(phi, H, V_p, V_pp)
    b = diffusion_term(phi, H)
    b_p = diffusion_term_dphi_sr(phi, H, V_p)
    b_pp = diffusion_term_ddphi_sr(phi, H, V_p, V_pp)
    
    phi_new = phi + a*dN + b*dW + 0.5*b*b_p*(dW**2-dN) + b*a_p*dZ +\
        0.5*(a*a_p + 0.5*a_p*b**2)*dN**2 + (a*b_p+0.5*b_pp*b**2)*(dW*dN-dZ)
    return phi_new

#Uses the weak order 2 Milstein-Taylay (mt) method to define adaptive steps, by
#comparing to the weak order 1 Euler-Maruyama (em) step
cdef list adaptive_step(double phi, double N, double dN, double sqrt_dN,\
                            double rand_num, double tol, V, V_p, V_pp):
    cdef double H, dW, dZ, a, a_p, b, b_p, b_pp, phi_new, tm_higher_order, error
    H = hubble_sr(phi, V)
    a = drift_term_sr(phi, H, V_p)
    a_p = drift_term_dphi_sr(phi, H, V_p, V_pp)
    b = diffusion_term(phi, H)
    b_p = diffusion_term_dphi_sr(phi, H, V_p)
    b_pp = diffusion_term_ddphi_sr(phi, H, V_p, V_pp)
    if a>0:#Rolling to the right case
        dW = 3*sqrt_dN
        dZ = 0.5*dW*dN
        tm_higher_order = 0.5*b*b_p*(dW**2-dN) + b*a_p*dZ +\
            0.5*(a*a_p + 0.5*a_p*b**2)*dN**2 + (a*b_p+0.5*b_pp*b**2)*(dW*dN-dZ)
        error = abs(tm_higher_order/phi)
        while error>tol:#If step too large
            dN = dN/2
            sqrt_dN = dN**0.5
            dW = 3*sqrt_dN
            dZ = 0.5*dW*dN
            tm_higher_order = 0.5*b*b_p*(dW**2-dN) + b*a_p*dZ +\
                0.5*(a*a_p + 0.5*a_p*b**2)*dN**2 +\
                    (a*b_p+0.5*b_pp*b**2)*(dW*dN-dZ)
            error = abs(tm_higher_order/phi)
        while 4*error<tol:#If step too large
            dN = 2*dN
            sqrt_dN = dN**0.5
            dW = 3*sqrt_dN#As want maximum positive step
            dZ = 0.5*dW*dN
            tm_higher_order = 0.5*b*b_p*(dW**2-dN) + b*a_p*dZ +\
                0.5*(a*a_p + 0.5*a_p*b**2)*dN**2 + (a*b_p+0.5*b_pp*b**2)*(dW*dN-dZ)
            error = abs(tm_higher_order/phi)
        #Now the correct dN is defined, can do the MT step with the correct
        #noise term.
        dW = rand_num*sqrt_dN
        dZ = 0.5*dW*dN
        phi_new = phi + a*dN + b*dW + 0.5*b*b_p*(dW**2-dN) + b*a_p*dZ +\
        0.5*(a*a_p + 0.5*a_p*b**2)*dN**2 + (a*b_p+0.5*b_pp*b**2)*(dW*dN-dZ)
    else:#Rolling to the left case
        dW = -3*sqrt_dN#As want maximum negetive step
        dZ = 0.5*dW*dN
        tm_higher_order = 0.5*b*b_p*(dW**2-dN) + b*a_p*dZ +\
            0.5*(a*a_p + 0.5*a_p*b**2)*dN**2 + (a*b_p+0.5*b_pp*b**2)*(dW*dN-dZ)
        error = abs(tm_higher_order/phi)
        while error>tol:#If step too large
            dN = dN/2
            sqrt_dN = dN**0.5
            dW = -3*sqrt_dN#As want maximum negetive step
            dZ = 0.5*dW*dN
            tm_higher_order = 0.5*b*b_p*(dW**2-dN) + b*a_p*dZ +\
                0.5*(a*a_p + 0.5*a_p*b**2)*dN**2 + (a*b_p+0.5*b_pp*b**2)*(dW*dN-dZ)
            error = abs(tm_higher_order/phi)
        while 2*error<tol:#If step too large
            dN = 2*dN
            sqrt_dN = dN**0.5
            dW = -3*sqrt_dN#As want maximum negetive step
            dZ = 0.5*dW*dN
            tm_higher_order = 0.5*b*b_p*(dW**2-dN) + b*a_p*dZ +\
                0.5*(a*a_p + 0.5*a_p*b**2)*dN**2 + (a*b_p+0.5*b_pp*b**2)*(dW*dN-dZ)
            error = abs(tm_higher_order/phi)
        #Now the correct dN is defined, can do the MT step with the correct
        #noise term.
        dW = rand_num*sqrt_dN
        dZ = 0.5*dW*dN
        phi_new = phi + a*dN + b*dW + 0.5*b*b_p*(dW**2-dN) + b*a_p*dZ +\
        0.5*(a*a_p + 0.5*a_p*b**2)*dN**2 + (a*b_p+0.5*b_pp*b**2)*(dW*dN-dZ)
    return [phi_new, dN, sqrt_dN]

#This is the discreate version of calculating the bias variable, which uses
#the discreat, see Esq. (18) and (19) of arXiv:nucl-th/9809075v1
cdef double importance_sampling_variable_A_step(double A, double D,\
                                                double phi_step, double v,\
                                                    double delta_v, double dN):
    return A+delta_v*(phi_step-v*dN-0.5*delta_v*dN)/D

#Most general form, as the bias is general. Mainly used for when the bias
#is passed as a function
cdef list euler_maruyama_importance_sampling(double phi, double A, double N,\
                                             double dN, double sqrt_dN,\
                                             double dW, double bias,\
                                             V, V_p, V_pp):
    cdef double H, a_orig, a, b, phi_new, A_new
    H = hubble_sr(phi, V)
    a_orig = drift_term_sr(phi, H, V_p)
    a = a_orig + bias
    b = diffusion_term(phi, H)
    
    #Calculation
    phi_new = phi + a*dN + b*dW
    #Just an Euler step in the importance sampling variable
    A_new = importance_sampling_variable_A_step(A, b**2, phi_new-phi, a_orig,\
                                               bias, dN)
    return  [phi_new, A_new]

cdef list euler_maruyama_importance_sampling_drift_bias(double phi, double A, double N,\
                                             double dN, double sqrt_dN,\
                                             double dW, double bias,\
                                             V, V_p, V_pp):
    cdef double H, a_orig, a, b, phi_new, A_new
    H = hubble_sr(phi, V)
    a_orig = drift_term_sr(phi, H, V_p)
    a = a_orig*(1-bias)
    b = diffusion_term(phi, H)
    
    #Calculation
    phi_new = phi + a*dN + b*dW
    #Just an Euler step in the importance sampling variable
    A_new = importance_sampling_variable_A_step(A, b**2, phi_new-phi, a_orig,\
                                               -bias*a_orig, dN)
    return  [phi_new, A_new]

cdef list euler_maruyama_importance_sampling_diffusion_bias(double phi, double A, double N,\
                                             double dN, double sqrt_dN,\
                                             double dW, double bias,\
                                             V, V_p, V_pp):
    cdef double H, a_orig, a, b, phi_new, A_new
    H = hubble_sr(phi, V)
    a_orig = drift_term_sr(phi, H, V_p)
    b = diffusion_term(phi, H)
    a = a_orig+bias*b
    
    #Calculation
    phi_new = phi + a*dN + b*dW
    #Just an Euler step in the importance sampling variable
    A_new = importance_sampling_variable_A_step(A, b**2, phi_new-phi, a_orig,\
                                               bias*b, dN)
    return  [phi_new, A_new]

cdef list euler_maruyama_importance_sampling_constant_bias(double phi, double A, double N,\
                                             double dN, double sqrt_dN,\
                                             double dW, double bias,\
                                             V, V_p, V_pp):
    cdef double H, a_orig, a, phi_new, A_new
    H = hubble_sr(phi, V)
    a_orig = drift_term_sr(phi, H, V_p)
    b = diffusion_term(phi, H)
    a = a_orig+bias
    
    #Calculation
    phi_new = phi + a*dN + b*dW
    #Just an Euler step in the importance sampling variable
    A_new = importance_sampling_variable_A_step(A, b**2, phi_new-phi, a_orig,\
                                               bias, dN)
    return  [phi_new, A_new]
    
#This method is only applicable when doing importance sampling
#Weak converrgance order 2
#See (29) of the intro to Kloeden and Platen 1999
#dW is the Weiner step, normally random Gaussian x dN^1/2
cdef list milstein_taylay_importance_sampling(double phi, double A, double N,\
                                              double dN, double sqrt_dN,\
                                              double dW, double bias,\
                                              V, V_p, V_pp):
    cdef double H, dZ, a_orig, a, a_p, b, b_p, b_pp, phi_new, A_new
    H = hubble_sr(phi, V)
    dZ = 0.5*dW*dN
    a_orig = drift_term_sr(phi, H, V_p)
    a = a_orig*(1-bias)
    a_p = drift_term_dphi_sr(phi, H, V_p, V_pp)*(1-bias)
    b = diffusion_term(phi, H)
    b_p = diffusion_term_dphi_sr(phi, H, V_p)
    b_pp = diffusion_term_ddphi_sr(phi, H, V_p, V_pp)
    
    #Calculation
    phi_new = phi + a*dN + b*dW + 0.5*b*b_p*(dW**2-dN) + b*a_p*dZ +\
        0.5*(a*a_p + 0.5*a_p*b**2)*dN**2 + (a*b_p+0.5*b_pp*b**2)*(dW*dN-dZ)
    #Just an Euler step in the importance sampling variable
    A_new = importance_sampling_variable_A_step(A, b**2, phi_new-phi, a_orig,\
                                               -bias*a_orig, dN)
    return  [phi_new, A_new]

#As this is an importance sampling the actual step is done using EM for
#consitancy with the derivation of A. This uses the diffusion to define Delta v
#Uses the weak order 2 Milstein-Taylay (mt) method to define adaptive steps, by
#comparing to the weak order 1 Euler-Maruyama (em) step
cdef list adaptive_step_importance_sampling_diffusion_bias(double phi,\
            double A, double N, double dN, double sqrt_dN, double rand_num,\
            double tol, double bias, V, V_p, V_pp):
    cdef double H, dW, dZ, a, a_orig, a_p, b, b_p, b_pp, phi_new,\
        tm_higher_order, error
    H = hubble_sr(phi, V)
    b = diffusion_term(phi, H)
    b_p = diffusion_term_dphi_sr(phi, H, V_p)
    b_pp = diffusion_term_ddphi_sr(phi, H, V_p, V_pp)
    a_orig = drift_term_sr(phi, H, V_p)
    a = a_orig + bias*b
    a_p = drift_term_dphi_sr(phi, H, V_p, V_pp)+bias*b_p
    if a>0:#Rolling to the right case
        dW = 3*sqrt_dN
        dZ = 0.5*dW*dN
        tm_higher_order = 0.5*b*b_p*(dW**2-dN) + b*a_p*dZ +\
            0.5*(a*a_p + 0.5*a_p*b**2)*dN**2 + (a*b_p+0.5*b_pp*b**2)*(dW*dN-dZ)
        error = abs(tm_higher_order/phi)
        while error>tol:#If step too large
            dN = dN/2
            sqrt_dN = dN**0.5
            dW = 3*sqrt_dN
            dZ = 0.5*dW*dN
            tm_higher_order = 0.5*b*b_p*(dW**2-dN) + b*a_p*dZ +\
                0.5*(a*a_p + 0.5*a_p*b**2)*dN**2 +\
                    (a*b_p+0.5*b_pp*b**2)*(dW*dN-dZ)
            error = abs(tm_higher_order/phi)
        while 4*error<tol:#If step too large
            dN = 2*dN
            sqrt_dN = dN**0.5
            dW = 3*sqrt_dN#As want maximum positive step
            dZ = 0.5*dW*dN
            tm_higher_order = 0.5*b*b_p*(dW**2-dN) + b*a_p*dZ +\
                0.5*(a*a_p + 0.5*a_p*b**2)*dN**2 + (a*b_p+0.5*b_pp*b**2)*(dW*dN-dZ)
            error = abs(tm_higher_order/phi)
        #Now the correct dN is defined, can do the MT step with the correct
        #noise term.
        dW = rand_num*sqrt_dN
        dZ = 0.5*dW*dN
        phi_new = phi + a*dN + b*dW + 0.5*b*b_p*(dW**2-dN) + b*a_p*dZ +\
        0.5*(a*a_p + 0.5*a_p*b**2)*dN**2 + (a*b_p+0.5*b_pp*b**2)*(dW*dN-dZ)
        A_new = importance_sampling_variable_A_step(A, b**2, phi_new-phi, a_orig,\
                                               bias*b, dN)
    else:#Rolling to the left case
        dW = -3*sqrt_dN#As want maximum negetive step
        dZ = 0.5*dW*dN
        tm_higher_order = 0.5*b*b_p*(dW**2-dN) + b*a_p*dZ +\
            0.5*(a*a_p + 0.5*a_p*b**2)*dN**2 + (a*b_p+0.5*b_pp*b**2)*(dW*dN-dZ)
        error = abs(tm_higher_order/phi)
        while error>tol:#If step too large
            dN = dN/2
            sqrt_dN = dN**0.5
            dW = -3*sqrt_dN#As want maximum negetive step
            dZ = 0.5*dW*dN
            tm_higher_order = 0.5*b*b_p*(dW**2-dN) + b*a_p*dZ +\
                0.5*(a*a_p + 0.5*a_p*b**2)*dN**2 + (a*b_p+0.5*b_pp*b**2)*(dW*dN-dZ)
            error = abs(tm_higher_order/phi)
        while 2*error<tol:#If step too large
            dN = 2*dN
            sqrt_dN = dN**0.5
            dW = -3*sqrt_dN#As want maximum negetive step
            dZ = 0.5*dW*dN
            tm_higher_order = 0.5*b*b_p*(dW**2-dN) + b*a_p*dZ +\
                0.5*(a*a_p + 0.5*a_p*b**2)*dN**2 + (a*b_p+0.5*b_pp*b**2)*(dW*dN-dZ)
            error = abs(tm_higher_order/phi)
        #Now the correct dN is defined, can do the MT step with the correct
        #noise term.
        dW = rand_num*sqrt_dN
        dZ = 0.5*dW*dN
        phi_new = phi + a*dN + b*dW + 0.5*b*b_p*(dW**2-dN) + b*a_p*dZ +\
        0.5*(a*a_p + 0.5*a_p*b**2)*dN**2 + (a*b_p+0.5*b_pp*b**2)*(dW*dN-dZ)
        A_new = importance_sampling_variable_A_step(A, b**2, phi_new-phi, a_orig,\
                                               bias*b, dN)
    return [phi_new, A_new, dN, sqrt_dN]


cdef double simulation(double phi_i, double phi_r, double phi_end, double N_i,\
                       double N_f, double dN, V, V_p, V_pp):
    cdef double N, sqrt_dN, phi, noise_amp, dist_end_inflation, dW
    cdef list step_results
    cdef int reduced_step = 0
    sqrt_dN = dN**0.5
    N = N_i
    phi = phi_i
    dist_end_inflation = 0.0
    while N<N_f:
        #Define the Wiener step
        dW = random.gauss(0.0,sqrt_dN)
        phi = milstein_taylay(phi, N, dN, sqrt_dN, dW, V, V_p, V_pp)
        noise_amp = diffusion_term(phi, hubble_sr(phi, V))*sqrt_dN
        N += dN
        if end_condition(phi, N, phi_end) == 1:#Using 1/0 for True/False
            #print('stopped at: ' + str(N))
            break
        elif end_condition(phi+3*noise_amp, N, phi_end) == 1\
            and reduced_step == 0:
            dN = 0.001*dN
            sqrt_dN = dN**0.5
            dist_end_inflation = 3*noise_amp
            reduced_step = 1
        elif end_condition(phi-3*noise_amp, N, phi_end) == 1\
            and reduced_step == 0:
            dN = 0.001*dN
            sqrt_dN = dN**0.5
            dist_end_inflation = -3*noise_amp
            reduced_step = 1
        elif end_condition(phi+dist_end_inflation, N, phi_end) == 0\
            and reduced_step == 1:
            dN = 1000*dN#Go back to original step size
            sqrt_dN = dN**0.5#Remember to update the root
            reduced_step = 0
        elif reflect_condition(phi, N, phi_r) == 1:
            phi = reflection(phi, phi_r)
        
    return N

cdef double simulation_adaptive(double phi_i, double phi_r, double phi_end,\
                                double N_i, double N_f, double dN, double tol,\
                                V, V_p, V_pp):
    cdef double N, sqrt_dN, phi, noise_amp, rand_number, dist_end_inflation
    cdef list step_results
    cdef int reduced_step = 0
    sqrt_dN = dN**0.5
    N = N_i
    phi = phi_i
    dist_end_inflation = 0.0
    while N<N_f:
        #Define the Wiener step
        rand_number = random.gauss(0.0,1)
        [phi,dN,sqrt_dN]  = adaptive_step(phi, N, dN, sqrt_dN, rand_number,\
                                          tol, V, V_p, V_pp)
        noise_amp = diffusion_term(phi, hubble_sr(phi, V))*sqrt_dN
        N += dN
        if end_condition(phi, N, phi_end) == 1:#Using 1/0 for True/False
            #print('stopped at: ' + str(N))
            break
        elif end_condition(phi+3*noise_amp, N, phi_end) == 1\
            and reduced_step == 0:
            dN = 0.001*dN
            sqrt_dN = dN**0.5
            dist_end_inflation = 3*noise_amp
            reduced_step = 1
        elif end_condition(phi-3*noise_amp, N, phi_end) == 1\
            and reduced_step == 0:
            dN = 0.001*dN
            sqrt_dN = dN**0.5
            dist_end_inflation = -3*noise_amp
            reduced_step = 1
        elif end_condition(phi+dist_end_inflation, N, phi_end) == 0\
            and reduced_step == 1:
            dN = 1000*dN#Go back to original step size
            sqrt_dN = dN**0.5#Remember to update the root
            reduced_step = 0
        elif reflect_condition(phi, N, phi_r) == 1:
            phi = reflection(phi, phi_r)
        
    return N


#A let's us calculate the bias w=e^-A is the bias, which propagated along with 
#the importance sample path.
#See Eq. (33) of arXiv:nucl-th/9809075v1 for more info
cdef list simulation_importance_sampling(double phi_i, double phi_r, double phi_end,\
                       double N_i, double N_f, double dN, double bias, V, V_p,\
                           V_pp):
    cdef double N, sqrt_dN, phi, noise_amp, dist_end_inflation, dW, A
    cdef int reduced_step = 0
    sqrt_dN = dN**0.5
    N = N_i
    phi = phi_i
    dist_end_inflation = 0.0
    A = 0.0
    while N<N_f:
        #Define the Wiener step
        dW = random.gauss(0.0,sqrt_dN)
        [phi, A] = euler_maruyama_importance_sampling_drift_bias(phi, A, N, dN,\
                                                           sqrt_dN, dW, bias,\
                                                               V, V_p, V_pp)
        noise_amp = diffusion_term(phi, hubble_sr(phi, V))*sqrt_dN
        N += dN
        if end_condition(phi, N, phi_end) == 1:#Using 1/0 for True/False
            #print('stopped at: ' + str(N))
            break
        elif end_condition(phi+3*noise_amp, N, phi_end) == 1\
            and reduced_step == 0:
            dN = 0.001*dN
            sqrt_dN = dN**0.5
            dist_end_inflation = 3*noise_amp
            reduced_step = 1
        elif end_condition(phi-3*noise_amp, N, phi_end) == 1\
            and reduced_step == 0:
            dN = 0.001*dN
            sqrt_dN = dN**0.5
            dist_end_inflation = -3*noise_amp
            reduced_step = 1
        elif end_condition(phi+dist_end_inflation, N, phi_end) == 0\
            and reduced_step == 1:
            dN = 1000*dN#Go back to original step size
            sqrt_dN = dN**0.5#Remember to update the root
            reduced_step = 0
        elif reflect_condition(phi, N, phi_r) == 1:
            phi = reflection(phi, phi_r)
        
    return [N, e**(-A)]


#A let's us calculate the bias w=e^-A is the bias, which propagated along with 
#the importance sample path.
#See Eq. (33) of arXiv:nucl-th/9809075v1 for more info
cdef list simulation_importance_sampling_diffusion_bias(double phi_i, double phi_r, double phi_end,\
                       double N_i, double N_f, double dN, double bias, V, V_p,\
                           V_pp, count_refs = False):
    cdef double N, sqrt_dN, phi, noise_amp, dist_end_inflation, dW, A
    cdef int reduced_step = 0
    cdef int num_reflects = 0
    sqrt_dN = dN**0.5
    N = N_i
    phi = phi_i
    dist_end_inflation = 0.0
    A = 0.0
    while N<N_f:
        #Define the Wiener step
        dW = random.gauss(0.0,sqrt_dN)
        [phi, A] =\
            euler_maruyama_importance_sampling_diffusion_bias(phi, A, N, dN,\
                                                           sqrt_dN, dW, bias,\
                                                               V, V_p, V_pp)
        noise_amp = diffusion_term(phi, hubble_sr(phi, V))*sqrt_dN
        N += dN
        if end_condition(phi, N, phi_end) == 1:#Using 1/0 for True/False
            #print('stopped at: ' + str(N))
            break
        elif end_condition(phi+3*noise_amp, N, phi_end) == 1\
            and reduced_step == 0:
            dN = 0.001*dN
            sqrt_dN = dN**0.5
            dist_end_inflation = 3*noise_amp
            reduced_step = 1
        elif end_condition(phi-3*noise_amp, N, phi_end) == 1\
            and reduced_step == 0:
            dN = 0.001*dN
            sqrt_dN = dN**0.5
            dist_end_inflation = -3*noise_amp
            reduced_step = 1
        elif end_condition(phi+dist_end_inflation, N, phi_end) == 0\
            and reduced_step == 1:
            dN = 1000*dN#Go back to original step size
            sqrt_dN = dN**0.5#Remember to update the root
            reduced_step = 0
        elif reflect_condition(phi, N, phi_r) == 1:
            phi = reflection(phi, phi_r)
            num_reflects += 1
            
    if count_refs == True:
        return [N, e**(-A), num_reflects]
    else:
        return [N, e**(-A)]

#This version allows the user to define the bias as a function. This function
#must take the field value as the only argument.
#A let's us calculate the bias w=e^-A is the bias, which propagated along with 
#the importance sample path.
#See Eq. (33) of arXiv:nucl-th/9809075v1 for more info
cdef list simulation_importance_sampling_bias_function(double phi_i, double phi_r, double phi_end,\
                       double N_i, double N_f, double dN, bias, V, V_p,\
                           V_pp, count_refs = False):
    cdef double N, sqrt_dN, phi, noise_amp, dist_end_inflation, dW, A
    cdef int reduced_step = 0
    cdef int num_reflects = 0
    sqrt_dN = dN**0.5
    N = N_i
    phi = phi_i
    dist_end_inflation = 0.0
    A = 0.0
    while N<N_f:
        #Define the Wiener step
        dW = random.gauss(0.0,sqrt_dN)
        bias_value = bias(phi)
        [phi, A] =\
            euler_maruyama_importance_sampling(phi, A, N, dN,\
                                               sqrt_dN, dW, bias_value,\
                                                V, V_p, V_pp)
                
        noise_amp = diffusion_term(phi, hubble_sr(phi, V))*sqrt_dN
        N += dN
        if end_condition(phi, N, phi_end) == 1:#Using 1/0 for True/False
            #print('stopped at: ' + str(N))
            break
        elif end_condition(phi+3*noise_amp, N, phi_end) == 1\
            and reduced_step == 0:
            dN = 0.001*dN
            sqrt_dN = dN**0.5
            dist_end_inflation = 3*noise_amp
            reduced_step = 1
        elif end_condition(phi-3*noise_amp, N, phi_end) == 1\
            and reduced_step == 0:
            dN = 0.001*dN
            sqrt_dN = dN**0.5
            dist_end_inflation = -3*noise_amp
            reduced_step = 1
        elif end_condition(phi+dist_end_inflation, N, phi_end) == 0\
            and reduced_step == 1:
            dN = 1000*dN#Go back to original step size
            sqrt_dN = dN**0.5#Remember to update the root
            reduced_step = 0
        elif reflect_condition(phi, N, phi_r) == 1:
            phi = reflection(phi, phi_r)
            num_reflects += 1
            
    if count_refs == True:
        return [N, e**(-A), num_reflects]
    else:
        return [N, e**(-A)]
        

cdef list simulation_importance_sampling_constant_bias(double phi_i, double phi_r, double phi_end,\
                       double N_i, double N_f, double dN, double bias, V, V_p,\
                           V_pp):
    cdef double N, sqrt_dN, phi, noise_amp, dist_end_inflation, dW, A
    cdef int reduced_step = 0
    sqrt_dN = dN**0.5
    N = N_i
    phi = phi_i
    dist_end_inflation = 0.0
    A = 0.0
    while N<N_f:
        #Define the Wiener step
        dW = random.gauss(0.0,sqrt_dN)
        [phi, A] =\
            euler_maruyama_importance_sampling_constant_bias(phi, A, N, dN,\
                                                           sqrt_dN, dW, bias,\
                                                               V, V_p, V_pp)
        noise_amp = diffusion_term(phi, hubble_sr(phi, V))*sqrt_dN
        N += dN
        if end_condition(phi, N, phi_end) == 1:#Using 1/0 for True/False
            #print('stopped at: ' + str(N))
            break
        elif end_condition(phi+3*noise_amp, N, phi_end) == 1\
            and reduced_step == 0:
            dN = 0.001*dN
            sqrt_dN = dN**0.5
            dist_end_inflation = 3*noise_amp
            reduced_step = 1
        elif end_condition(phi-3*noise_amp, N, phi_end) == 1\
            and reduced_step == 0:
            dN = 0.001*dN
            sqrt_dN = dN**0.5
            dist_end_inflation = -3*noise_amp
            reduced_step = 1
        elif end_condition(phi+dist_end_inflation, N, phi_end) == 0\
            and reduced_step == 1:
            dN = 1000*dN#Go back to original step size
            sqrt_dN = dN**0.5#Remember to update the root
            reduced_step = 0
        elif reflect_condition(phi, N, phi_r) == 1:
            phi = reflection(phi, phi_r)
        
    return [N, e**(-A)]


#A let's us calculate the bias w=e^-A is the bias, which propagated along with 
#the importance sample path.
#See Eq. (33) of arXiv:nucl-th/9809075v1 for more info
cdef list simulation_importance_sampling_diffusion_bias_double_end(\
           double phi_i, double phi_end_r, double phi_end_l, double N_i, double N_f,\
        double dN, double bias, V, V_p,\
                           V_pp):
    cdef double N, sqrt_dN, phi, noise_amp, dist_end_inflation, dW, A
    cdef int reduced_step = 0
    sqrt_dN = dN**0.5
    N = N_i
    phi = phi_i
    dist_end_inflation = 0.0
    A = 0.0
    while N<N_f:
        #Define the Wiener step
        dW = random.gauss(0.0,sqrt_dN)
        [phi, A] =\
            euler_maruyama_importance_sampling_diffusion_bias(phi, A, N, dN,\
                                                           sqrt_dN, dW, bias,\
                                                               V, V_p, V_pp)
        noise_amp = diffusion_term(phi, hubble_sr(phi, V))*sqrt_dN
        N += dN
        if double_end_condition(phi, N, phi_end_l, phi_end_r) == 1:#Using 1/0 for True/False
            #print('stopped at: ' + str(N))
            break
        elif double_end_condition(phi+3*noise_amp, N, phi_end_l, phi_end_r) == 1\
            and reduced_step == 0:
            dN = 0.001*dN
            sqrt_dN = dN**0.5
            dist_end_inflation = 3*noise_amp
            reduced_step = 1
        elif double_end_condition(phi-3*noise_amp, N, phi_end_l, phi_end_r) == 1\
            and reduced_step == 0:
            dN = 0.001*dN
            sqrt_dN = dN**0.5
            dist_end_inflation = -3*noise_amp
            reduced_step = 1
        elif double_end_condition(phi+dist_end_inflation, N, phi_end_l, phi_end_r) == 0\
            and reduced_step == 1:
            dN = 1000*dN#Go back to original step size
            sqrt_dN = dN**0.5#Remember to update the root
            reduced_step = 0
        
    return [N, e**(-A)]


#A let's us calculate the bias w=e^-A is the bias, which propagated along with 
#the importance sample path.
#See Eq. (33) of arXiv:nucl-th/9809075v1 for more info
cdef list simulation_importance_sampling_diffusion_bias_adaptive(double phi_i,\
            double phi_r, double phi_end, double N_i, double N_f, double dN,\
            double tol, double bias, V, V_p,\
                           V_pp):
    cdef double N, sqrt_dN, phi, noise_amp, rand_number, dist_end_inflation, A
    cdef int reduced_step = 0
    sqrt_dN = dN**0.5
    N = N_i
    phi = phi_i
    dist_end_inflation = 0.0
    A = 0.0
    while N<N_f:
        #Define the Wiener step
        rand_number = random.gauss(0.0,1.0)
        [phi, A, dN, sqrt_dN] =\
            adaptive_step_importance_sampling_diffusion_bias(phi, A, N, dN,\
                                                             sqrt_dN,\
                                                             rand_number,\
                                                             tol, bias, V, V_p,\
                                                             V_pp)
        noise_amp = diffusion_term(phi, hubble_sr(phi, V))*sqrt_dN
        N += dN
        if end_condition(phi, N, phi_end) == 1:#Using 1/0 for True/False
            #print('stopped at: ' + str(N))
            break
        elif end_condition(phi+3*noise_amp, N, phi_end) == 1\
            and reduced_step == 0:
            dN = 0.001*dN
            sqrt_dN = dN**0.5
            dist_end_inflation = 3*noise_amp
            reduced_step = 1
        elif end_condition(phi-3*noise_amp, N, phi_end) == 1\
            and reduced_step == 0:
            dN = 0.001*dN
            sqrt_dN = dN**0.5
            dist_end_inflation = -3*noise_amp
            reduced_step = 1
        elif end_condition(phi+dist_end_inflation, N, phi_end) == 0\
            and reduced_step == 1:
            dN = 1000*dN#Go back to original step size
            sqrt_dN = dN**0.5#Remember to update the root
            reduced_step = 0
        elif reflect_condition(phi, N, phi_r) == 1:
            phi = reflection(phi, phi_r)
        
    return [N, e**(-A)]


    
cpdef many_simulations(double phi_i, double phi_r, double phi_end,\
                     double N_i, double N_f, double dN,\
                     int num_sims, V, V_p, V_pp, tol = 0.0):
    if tol != 0:
        #preallocate memory
        N_dist =\
            [simulation_adaptive(phi_i, phi_r, phi_end, N_i, N_f, dN, tol, V,\
            V_p, V_pp) for i in range(num_sims)]
    else:
        #preallocate memory
        N_dist =\
            [simulation(phi_i, phi_r, phi_end, N_i, N_f, dN, V, V_p, V_pp)\
             for i in range(num_sims)]
    return N_dist

cpdef many_simulations_importance_sampling(double phi_i, double phi_r, double phi_end,\
                     double N_i, double N_f, double dN, bias,\
                     int num_sims, V, V_p, V_pp, bias_type = 'diffusion',\
                     boundary = 'reflective', tol = 0.0, count_refs = False):
    if bias_type == 'drift':
        #preallocate memory
        sim_results =\
            [simulation_importance_sampling(phi_i, phi_r, phi_end, N_i, N_f, dN, bias, V,\
                                            V_p, V_pp) for i in range(num_sims)]
        #Correctly slicing the list with list comprehension
        Ns, ws = [[sim_results[i][j] for i in range(num_sims)] for j in range(2)]
    elif bias_type == 'diffusion':#
        if tol == 0:
            #preallocate memory
            if boundary == 'reflective':
                sim_results =\
                    [simulation_importance_sampling_diffusion_bias(phi_i, phi_r,\
                    phi_end, N_i, N_f, dN, bias, V, V_p, V_pp,\
                    count_refs=count_refs) for i in range(num_sims)]
            elif boundary == 'double absorbing':
                sim_results =\
                    [simulation_importance_sampling_diffusion_bias_double_end(\
                    phi_i, phi_r, phi_end, N_i, N_f, dN, bias, V, V_p, V_pp)\
                    for i in range(num_sims)]
                #Correctly slicing the list with list comprehension
        elif tol != 0:
            #preallocate memory
            sim_results =\
                [simulation_importance_sampling_diffusion_bias_adaptive(phi_i,\
                phi_r, phi_end, N_i, N_f, dN, tol, bias, V, V_p, V_pp)\
                for i in range(num_sims)]
            #Correctly slicing the list with list comprehension

    elif bias_type == 'constant':
        sim_results =\
            [simulation_importance_sampling_constant_bias(phi_i, phi_r, phi_end, N_i, N_f, dN, bias, V,\
                                            V_p, V_pp) for i in range(num_sims)]
                
    elif bias_type == 'custom':
        sim_results =\
            [simulation_importance_sampling_bias_function(phi_i, phi_r,\
            phi_end, N_i, N_f, dN, bias, V, V_p, V_pp,\
            count_refs=count_refs) for i in range(num_sims)]
    
    #Correctly slicing the list with list comprehension
    if count_refs == False:
        Ns, ws = [[sim_results[i][j] for i in range(num_sims)] for j in range(2)]
        return Ns, ws
    elif count_refs == True:
        Ns, ws, num_reflects = [[sim_results[i][j] for i in range(num_sims)]\
                                for j in range(3)]
        return Ns, ws, num_reflects
    else:
        raise ValueError('Unknown parameter for coundting reflections')


    

    
