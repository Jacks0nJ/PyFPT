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



cdef double diffusion_term(double phi, double H):
    return H/(2*PI)


#This assumes SR
cdef double drift_term_sr(double phi, double H, V_p):
    return (-V_p(phi)/(3*H**2))
    


#This is the discreate version of calculating the bias variable, which uses
#the discreat, see Esq. (18) and (19) of arXiv:nucl-th/9809075v1
cdef double i_s_A_step(double A, double D, double phi_step, double v,\
                       double delta_v, double dN):
    return A+delta_v*(phi_step-v*dN-0.5*delta_v*dN)/D

#Most general form, as the bias is general. Mainly used for when the bias
#is passed as a function
cdef list step(double phi, double A, double N, double dN, double sqrt_dN,\
               double dW, double bias, V, V_p, V_pp):
    cdef double H, a_orig, a, b, phi_new, A_new
    H = hubble_sr(phi, V)
    a_orig = drift_term_sr(phi, H, V_p)
    a = a_orig + bias
    b = diffusion_term(phi, H)
    
    #Calculation
    phi_new = phi + a*dN + b*dW
    #Just an Euler step in the importance sampling variable
    A_new = i_s_A_step(A, b**2, phi_new-phi, a_orig,\
                                               bias, dN)
    return  [phi_new, A_new]


cdef list diffusion_step(double phi, double A, double N, double dN,\
                         double sqrt_dN, double dW, double bias, V, V_p, V_pp):
    cdef double H, a_orig, a, b, phi_new, A_new
    H = hubble_sr(phi, V)
    a_orig = drift_term_sr(phi, H, V_p)
    b = diffusion_term(phi, H)
    a = a_orig+bias*b
    
    #Calculation
    phi_new = phi + a*dN + b*dW
    #Just an Euler step in the importance sampling variable
    A_new = i_s_A_step(A, b**2, phi_new-phi, a_orig,\
                                               bias*b, dN)
    return  [phi_new, A_new]


#A let's us calculate the bias w=e^-A is the bias, which propagated along with 
#the importance sample path.
#See Eq. (33) of arXiv:nucl-th/9809075v1 for more info
cdef list simulation_i_s_diff_bias(double phi_i, double phi_r, double phi_end,\
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
            diffusion_step(phi, A, N, dN, sqrt_dN, dW, bias, V, V_p, V_pp)
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
cdef list simulation_i_s_bias_func(double phi_i, double phi_r, double phi_end,\
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
            step(phi, A, N, dN, sqrt_dN, dW, bias_value, V, V_p, V_pp)
                
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



cpdef importance_sampling_simulations(double phi_i, double phi_r, double phi_end,\
                     double N_i, double N_f, double dN, bias,\
                     int num_runs, V, V_p, V_pp, bias_type = 'diffusion',\
                     count_refs = False):
    if bias_type == 'diffusion':
        results =\
            [simulation_i_s_diff_bias(phi_i, phi_r,\
            phi_end, N_i, N_f, dN, bias, V, V_p, V_pp,\
            count_refs=count_refs) for i in range(num_runs)]
                
    elif bias_type == 'custom':
        results =\
            [simulation_i_s_bias_func(phi_i, phi_r,\
            phi_end, N_i, N_f, dN, bias, V, V_p, V_pp,\
            count_refs=count_refs) for i in range(num_runs)]
    
    #Correctly slicing the list with list comprehension
    if count_refs == False:
        Ns, ws = [[results[i][j] for i in range(num_runs)] for j in range(2)]
        return Ns, ws
    elif count_refs == True:
        Ns, ws, num_reflects = [[results[i][j] for i in range(num_runs)]\
                                for j in range(3)]
        return Ns, ws, num_reflects
    else:
        raise ValueError('Unknown parameter for coundting reflections')


    

    
