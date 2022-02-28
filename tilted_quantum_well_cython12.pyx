'''
Created on 07/04/21

    Using Cython to look at importance sampling of the simple quantum well
    case. This also includes the ability to look at a constant, non-vanishing
    slope case
    
@author: Joe Jackson
'''
import random
#This mode
cdef double M_PL = 1.0
cdef double PI = 3.1415926535897931159979634685
cdef double e = 2.718281828459045

#Prealoocating the defining conditions of the tilted quantum well
cdef double phi_end = 1*M_PL
cdef double V_0 = 0.0001#This is just a place holding value
cdef double v_0 = V_0/(24*(PI**2)*(M_PL**4))# This is the reduced potential



#Defining the potential for the tilted quantum well

cdef double V(double phi, double tilt, double V=V_0):
    return V*(1+tilt*(phi-phi_end))

cdef double V_p(double phi, double tilt, double V=V_0):
    return V*tilt
#Remember the second derivative vanishes
cdef double V_pp(double phi, double tilt, double V=V_0):
    return 0.0

cdef int end_condition(double phi, double phi_end, double N):
    if phi>phi_end:
        return 0
    elif phi<=phi_end:
        return 1
 #This is for the double absorbing well   
cdef int end_condition_double_absorbing(double phi, double phi_l_end,\
                                        double phi_r_end, double N):
    if phi<=phi_l_end:
        return 1
    elif phi>=phi_r_end:
        return 1
    else:
        return 0
    
cdef int reflect_condition(double phi, double N, double phi_reflect):
    if phi<phi_reflect:
        return 0
    elif phi>=phi_reflect:
        return 1

cdef double reflection(phi, N, phi_reflect):
    return 2*phi_reflect - phi
        
#Defining the more general functions

cdef double hubble_sr(double phi, double tilt):
    cdef double H = (V(phi, tilt)/(3*M_PL**2))**0.5
    return H

cdef double phi_dN_sr(double phi, double tilt):
    return -V_p(phi, tilt)/(3*hubble_sr(phi, tilt)**2)

cdef double hubble_dN_sr(double phi, double tilt):       
    return -0.5*hubble_sr(phi, tilt)*(phi_dN_sr(phi, tilt)/M_PL)**2

cdef double diffusion_term(double phi, double H, double tilt):
    return H/(2*PI)

cdef double diffusion_term_dphi_sr(double phi, double H, double tilt):
    return V_p(phi, tilt)/(12*PI*(M_PL**2)*H)

cdef double diffusion_term_ddphi_sr(double phi, double H, double tilt):
    return V_pp(phi, tilt)/(12*PI*(M_PL**2)*H) -\
        (V_p(phi, tilt)**2)/(72*PI*(M_PL**4)*H**3)

#This assumes SR
cdef double drift_term_sr(double phi, double H, double tilt):
    return (-V_p(phi, tilt)/(3*H**2))
    
#This assumes slow-roll approx
cdef double drift_term_dphi_sr(double phi, double H, double tilt):
    return (-V_pp(phi, tilt)/(3*H**2)) + (V_p(phi, tilt)/(3*M_PL*H**2))**2

#See (5) of the intro to Kloeden and Platen 1999
#dW is the Weiner step, normally random Gaussian x dN^1/2
cdef double euler_maruyama(double phi, double tilt, double N, double dN,\
                           double dW):
    cdef double H = hubble_sr(phi, tilt)
    return phi + drift_term_sr(phi, H, tilt)*dN+diffusion_term(phi, H, tilt)*dW
        
#See (22) of the intro to Kloeden and Platen 1999   
#dW is the Weiner step, normally random Gaussian x dN^1/2
cdef double stochastic_rk(double phi, double tilt, double N, double dN,\
                          double sqrt_dN, double dW):
    cdef double H = hubble_sr(phi, tilt)
    cdef double diff = diffusion_term(phi, H, tilt)
    return phi + drift_term_sr(phi, H, tilt)*dN + diff*dW +\
        0.5*(diffusion_term(phi+diff*sqrt_dN,\
        hubble_sr(phi+diff*sqrt_dN, tilt), tilt)-diff)*\
        (dW**2-dN)/sqrt_dN
        
#See A. J. Roberts 2012	arXiv:1210.0933
cdef double improved_euler(double phi, double tilt, double N, double dN,\
                           double sqrt_dN, double dW):
    cdef double H = hubble_sr(phi, tilt)
    #Randomly +/- as required by algoritham
    cdef double s_n = random.randrange(-1,2,2)
    cdef double k_1 = drift_term_sr(phi, H, tilt)*dN +\
        diffusion_term(phi, H, tilt)*(dW-s_n*sqrt_dN)
    cdef double phi_2 = phi + k_1
    #Now calculating second part of the step
    cdef double H_2 = hubble_sr(phi_2, tilt)
    cdef double k_2 = drift_term_sr(phi_2, H_2, tilt)*dN +\
        diffusion_term(phi_2, H_2, tilt)*(dW+s_n*sqrt_dN)
    return phi + (k_1+k_2)/2

#Kloeden and Platen, strong order 1
#See (16) of the intro to Kloeden and Platen 1999
#dW is the Weiner step, normally random Gaussian x dN^1/2
cdef double milstein(double phi, double tilt, double N, double dN,\
                          double dW):
    cdef double H = hubble_sr(phi, tilt)
    cdef double diff = diffusion_term(phi, H, tilt)
    return phi + drift_term_sr(phi, H, tilt)*dN + diff*dW\
        + 0.5*diff*diffusion_term_dphi_sr(phi, H, tilt)*(dW**2-dN)
        
        
#Weak converrgance order 2
#See (29) of the intro to Kloeden and Platen 1999
#dW is the Weiner step, normally random Gaussian x dN^1/2
cdef double milstein_taylay(double phi, double tilt, double N, double dN, double sqrt_dN,\
                            double dW):
    cdef double H, dZ, a, a_p, b, b_p, b_pp, phi_new
    H = hubble_sr(phi, tilt)
    dZ = 0.5*dW*dN
    a = drift_term_sr(phi, H, tilt)
    a_p = drift_term_dphi_sr(phi, H, tilt)
    b = diffusion_term(phi, H, tilt)
    b_p = diffusion_term_dphi_sr(phi, H, tilt)
    b_pp = diffusion_term_ddphi_sr(phi, H, tilt)
    
    phi_new = phi + a*dN + b*dW + 0.5*b*b_p*(dW**2-dN) + b*a_p*dZ +\
        0.5*(a*a_p + 0.5*a_p*b**2)*dN**2 + (a*b_p+0.5*b_pp*b**2)*(dW*dN-dZ)
    return phi_new

#This is the discreate version of calculating the bias variable, which uses
#the discreat, see Esq. (18) and (19) of arXiv:nucl-th/9809075v1
#There general notation is used here
cdef double importance_sampling_variable_A_step(double A, double D,\
                                                double x_step, double v,\
                                                    double delta_v, double dt):
    return A+delta_v*(x_step-v*dt-0.5*delta_v*dt)/D


#This is the N derivative for the importance sampling variable w, as defined
# in Eq. (33) of arXiv:nucl-th/9809075v1. This is used to find the bias for
#an importance sampled path. I used their notation here for generality.
cdef double importance_sampling_w_variable_step(double w, double delta_v,\
                                            double D_sqrt, double dt, double dW):
    return w-w*delta_v*(delta_v*dt+2*D_sqrt*dt)/(2*D_sqrt**2)

#The quantum well uses the diffusion to define the modification to the drift
cdef list euler_maruyama_importance_sampling(double phi, double A,\
                                             double tilt, double N, double dN,\
                                             double sqrt_dN, double dW,\
                                             double eta):
    cdef double H, a_orig, a, b, phi_new, A_new
    H = hubble_sr(phi, tilt)
    b = diffusion_term(phi, H, tilt)
    #Using the diffusion term to modify the drift
    a_orig = drift_term_sr(phi, H, tilt) 
    a = a_orig+eta*b
    
    #Calculation
    phi_new = phi + a*dN + b*dW
    #The new value for the importance sampling variable
    A_new = importance_sampling_variable_A_step(A, b**2, phi_new-phi, a_orig,\
                                               eta*b, dN)
    return  [phi_new, A_new]

#As this is the quantum well case, I'm using the diffusion term to define the
#the `wind' term of the importance sampling
cdef list milstein_taylay_importance_sampling(double phi, double A,\
                                              double tilt, double N,\
                                              double dN, double sqrt_dN,\
                                                    double dW, double eta):
    cdef double H, dZ, a_orig, a, a_p, b, b_p, b_pp, phi_new, A_new
    H = hubble_sr(phi, tilt)
    dZ = 0.5*dW*dN
    b = diffusion_term(phi, H, tilt)
    b_p = diffusion_term_dphi_sr(phi, H, tilt)
    b_pp = diffusion_term_ddphi_sr(phi, H, tilt)
    #Using the diffusion term to modify the drift
    a_orig = drift_term_sr(phi, H, tilt) 
    a = a_orig  + eta*b
    a_p = drift_term_dphi_sr(phi, H, tilt) + eta*b_p
    
    #Calculation
    phi_new = phi + a*dN + b*dW + 0.5*b*b_p*(dW**2-dN) + b*a_p*dZ +\
        0.5*(a*a_p + 0.5*a_p*b**2)*dN**2 + (a*b_p+0.5*b_pp*b**2)*(dW*dN-dZ)
    #Using the drift term to define the `wind' of the importance sampling and 
    #as we want it to tend to move up the potential, it is positive
    A_new = importance_sampling_variable_A_step(A, b**2, phi_new-phi, a,\
                                               eta*b, dN)
    return  [phi_new, A_new]


cpdef test_args(c, t=6.0):
    return t

cdef double simulation(double phi_i, double phi_end, double phi_r, double tilt,\
                       double N_i, double N_f, double dN):
    cdef double N = N_i
    cdef double phi = phi_i
    cdef int reduced_step = 0
    cdef double noise_amp, dist_end_inflation, dW
    dist_end_inflation = 0.0
    cdef double dN_sqrt = dN**0.5
    while N<N_f:
        #Define the Wiener step
        dW = random.gauss(0.0,dN_sqrt)
        phi = milstein_taylay(phi, tilt, N, dN, dN_sqrt, dW)
        noise_amp = diffusion_term(phi, hubble_sr(phi, tilt), tilt)*dN_sqrt
        N += dN
        if end_condition(phi, phi_end, N) == 1:#Using 1/0 for True/False
            #print('stopped at: ' + str(N))
            break
        elif end_condition(phi+3*noise_amp, phi_end, N) == 1\
            and reduced_step == 0:
            dN = 0.001*dN
            dN_sqrt = dN**0.5
            dist_end_inflation = 3*noise_amp
            reduced_step = 1
        elif end_condition(phi-3*noise_amp, phi_end, N) == 1\
            and reduced_step == 0:
            dN = 0.001*dN
            dN_sqrt = dN**0.5
            dist_end_inflation = -3*noise_amp
            reduced_step = 1
        elif end_condition(phi+dist_end_inflation, phi_end, N) == 1\
            and reduced_step == 1:
            dN = 1000*dN#Go back to original step size
            dN_sqrt = dN**0.5#Remember to update the root
            reduced_step = 0
        elif reflect_condition(phi, N, phi_r) == 1:
            phi = reflection(phi, N, phi_r)
        
    return N

#w is the bias, which propagated anlong with the importance sample path.
#See Eq. (33) of arXiv:nucl-th/9809075v1 for more info
cdef list simulation_importance_sampling(double phi_i, double phi_end,\
                                         double phi_r, double tilt,\
                                         double N_i, double N_f,\
                                         double dN, double eta,\
                                         str count_reflects = 'no'):
    cdef double N, phi, noise_amp, dist_end_inflation, dW, A
    cdef int reduced_step = 0
    cdef int num_reflects = 0
    cdef double dN_sqrt = dN**0.5
    cdef list step_results
    N = N_i
    phi = phi_i
    dist_end_inflation = 0.0
    A = 0.0
    while N<N_f:
        #Define the Wiener step
        dW = random.gauss(0.0,dN_sqrt)
        step_results = euler_maruyama_importance_sampling(phi, A, tilt, N, dN,\
                                                           dN_sqrt, dW, eta)
        phi = step_results[0]
        A = step_results[1]
        noise_amp = diffusion_term(phi, hubble_sr(phi, tilt), tilt)*dN_sqrt
        N += dN
        if end_condition(phi, phi_end, N) == 1:#Using 1/0 for True/False
            break
        elif end_condition(phi+3*noise_amp, phi_end, N) == 1\
            and reduced_step == 0:
            dN = 0.001*dN
            dN_sqrt = dN**0.5
            dist_end_inflation = 3*noise_amp
            reduced_step = 1
        elif end_condition(phi-3*noise_amp, phi_end, N) == 1\
            and reduced_step == 0:
            dN = 0.001*dN
            dN_sqrt = dN**0.5
            dist_end_inflation = -3*noise_amp
            reduced_step = 1
        elif end_condition(phi+dist_end_inflation, phi_end, N) == 1\
            and reduced_step == 1:
            dN = 1000*dN#Go back to original step size
            dN_sqrt = dN**0.5#Remember to update the root
            reduced_step = 0
        elif reflect_condition(phi, N, phi_r) == 1:
            num_reflects = num_reflects + 1
            phi = reflection(phi, N, phi_r)
    if count_reflects == 'yes':
        [N, e**(-A), num_reflects]
    else:
        return [N, e**(-A)]

#This is a temporarily code to investigate the effect of having a double
#absorbing well.
cdef list simulation_importance_sampling_double_absorbing(double phi_i, double phi_end,\
                                         double phi_r, double tilt,\
                                         double N_i, double N_f,\
                                         double dN, double eta,\
                                         str wind_type = 'one way'):
    cdef double N, phi, noise_amp, dist_end_inflation, dW, A, phi_l_end, phi_r_end
    cdef int reduced_step = 0
    cdef int wind_switched = 0
    cdef double dN_sqrt = dN**0.5
    cdef list step_results
    N = N_i
    phi = phi_i
    phi_l_end = phi_end
    phi_r_end = 2*phi_r-phi_end
    dist_end_inflation = 0.0
    A = 0.0
    while N<N_f:
        #Define the Wiener step
        dW = random.gauss(0.0,dN_sqrt)
        step_results = euler_maruyama_importance_sampling(phi, A, tilt, N, dN,\
                                                           dN_sqrt, dW, eta)
        phi = step_results[0]
        A = step_results[1]
        noise_amp = diffusion_term(phi, hubble_sr(phi, tilt), tilt)*dN_sqrt
        N += dN
        if end_condition_double_absorbing(phi, phi_l_end, phi_r_end, N) == 1:#Using 1/0 for True/False
            break
        elif end_condition_double_absorbing(phi+3*noise_amp, phi_l_end,\
                                            phi_r_end, N) == 1\
            and reduced_step == 0:
            dN = 0.001*dN
            dN_sqrt = dN**0.5
            dist_end_inflation = 3*noise_amp
            reduced_step = 1
        elif end_condition_double_absorbing(phi-3*noise_amp, phi_l_end,\
                                            phi_r_end, N) == 1\
            and reduced_step == 0:
            dN = 0.001*dN
            dN_sqrt = dN**0.5
            dist_end_inflation = -3*noise_amp
            reduced_step = 1
        elif end_condition_double_absorbing(phi+dist_end_inflation, phi_l_end,\
                                            phi_r_end, N) == 0\
            and reduced_step == 1:
            dN = 1000*dN#Go back to original step size
            dN_sqrt = dN**0.5#Remember to update the root
            reduced_step = 0
        elif wind_type == 'symmetric':
            if phi>phi_r and wind_switched == 0:
                eta = -eta
                wind_switched = 1
            elif phi<phi_r and wind_switched == 1:
                eta = -eta
                wind_switched = 0
    return [N, e**(-A)]

#Remember this for the quantum well case, which is defined by different
#parameters!
cpdef many_simulations(double mu, double x, double tilt, double N_i,\
                       double N_f, double dN, int num_sims,\
                       str boundary_type = 'reflective'):
    cdef double delta_phi = M_PL*mu*(v_0**0.5)
    cdef double phi_i = x*delta_phi+phi_end
    cdef double phi_r = delta_phi+phi_end
    if boundary_type == 'reflective':
        N_dist =\
            [simulation(phi_i, phi_end, phi_r, tilt, N_i, N_f, dN) for i in range(num_sims)]
    else:
        ValueError('Unknown bondary type')
    return N_dist

#Remember this for the quantum well case, which is defined by different
#parameters!
cpdef many_simulations_importance_sampling(double mu, double x, double tilt,\
                                           double N_i, double N_f, double dN,\
                                           double eta, int num_sims,\
                                           str boundary_type = 'reflective',\
                                           str wind_type = 'one way',\
                                           str count_reflects = 'no'):
    cdef double delta_phi = M_PL*mu*(v_0**0.5)
    cdef double phi_i = x*delta_phi+phi_end
    cdef double phi_r = delta_phi+phi_end
    if boundary_type == 'reflective':
        #Rescaling the potential to fit this mu value
        #preallocate memory
        sim_results =\
            [simulation_importance_sampling(phi_i, phi_end, phi_r, tilt, N_i,\
             N_f, dN, eta, count_reflects=count_reflects) for i in\
             range(num_sims)]
    elif boundary_type == 'double absorbing':
        #Rescaling the potential to fit this mu value
        #preallocate memory
        print('Using double absorbing surface, wind type is:'+wind_type)
        sim_results =\
            [simulation_importance_sampling_double_absorbing(phi_i, phi_end,\
             phi_r, tilt, N_i, N_f, dN, eta, wind_type=wind_type) for i in range(num_sims)]
    else:
        raise ValueError('Unknown boundary type')
    #Correctly slicing the list with list comprehension
    if count_reflects == 'no':
        Ns, ws = [[sim_results[i][j] for i in range(num_sims)] for j in range(2)]
        return Ns, ws
    elif count_reflects == 'yes':
        Ns, ws, num_reflects = [[sim_results[i][j] for i in range(num_sims)]\
                                for j in range(2)]
        return Ns, ws, num_reflects
    else:
        raise ValueError('Unknown parameter for coundting reflections')
        


    

    
