import random

cdef double e = 2.718281828459045



cdef int end_condition(double x, double t, double x_end, double direction):
    # May seem strange, but this logic means you have x_in either side of x_end
    if direction*x <= direction*x_end:
        return 1
    else:
        return 0
    

cdef int reflect_condition(double x, double t, double x_r, double direction):
    # May seem strange, but this logic means you have x_in either side of x_r
    if direction*x > direction*x_r:
        return 1
    else:
        return 0

cdef double reflection_value(double x, double x_r):
    # This relation is true regardless if left or right propagation in x
    return 2*x_r - x


# This is the discreate version of calculating the bias variable, which uses
# the discreat, see Esq. (18) and (19) of arXiv:nucl-th/9809075v1
# Note the notation is different
cdef double i_s_A_step(double A, double S, double x_step, double D,\
                       double B, double dt):
    return A+B*(x_step-D*dt-0.5*B*dt)/(S**2)


# Most general form, as the bias is general. Mainly used for when the bias
# is passed as a function
cdef list general_step(double x, double A, double t, double dt,\
                       double dW, double bias, drift, diffusion):
    cdef double D_orig, D, S, x_new, A_new
    D_orig = drift(x, t)
    D = D_orig + bias
    S = diffusion(x, t)
    
    #Calculation
    x_new = x + D*dt + S*dW
    #Just an Euler step in the importance sampling variable
    A_new = i_s_A_step(A, S, x_new-x, D_orig, bias, dt)
    return  [x_new, A_new]


cdef list diffusion_step(double x, double A, double t, double dt,\
                         double dW, double bias_amp, drift, diffusion):
    cdef double D_orig, D, S, x_new, A_new
    D_orig = drift(x, t)
    S = diffusion(x, t)
    D = D_orig+bias_amp*S
    
    #Calculation
    x_new = x + D*dt + S*dW
    #Just an Euler step in the importance sampling variable
    A_new = i_s_A_step(A, S, x_new-x, D_orig, bias_amp*S, dt)
    return  [x_new, A_new]


#A let's us calculate the bias w=e^-A is the bias, which propagated along with 
#the importance sample path.
#See Eq. (33) of arXiv:nucl-th/9809075v1 for more info
cdef list simulation_diff(double x_in, double x_r, double x_end, double t_i,\
                          double t_f, double dt, double bias_amp, drift,\
                          diffusion, count_refs = False):
    cdef double t, sqrt_dt, x, noise_amp, dist_end_inflation, dW, A, direction
    cdef int reduced_step = 0
    cdef int num_reflects = 0
    sqrt_dt = dt**0.5
    t = t_i
    x = x_in
    dist_end_inflation = 0.0
    A = 0.0
    direction = x_in-x_end

    while t<t_f:
        # Define the Wiener step
        dW = random.gauss(0.0, sqrt_dt)
        # Step in x and A simultanioues
        [x, A] =\
            diffusion_step(x, A, t, dt, dW, bias_amp, drift, diffusion)
        # This is used to see the end surface has been approached
        noise_amp = diffusion(x, t)*sqrt_dt
        t += dt
        if end_condition(x, t, x_end, direction) == 1:  # Using 1/0 for True/False
            break
        elif end_condition(x+4*noise_amp, t, x_end, direction) == 1\
            and reduced_step == 0:
            dt = 0.001*dt
            sqrt_dt = dt**0.5
            dist_end_inflation = 4*noise_amp
            reduced_step = 1
        elif end_condition(x-4*noise_amp, t, x_end, direction) == 1\
            and reduced_step == 0:
            dt = 0.001*dt
            sqrt_dt = dt**0.5
            dist_end_inflation = -4*noise_amp
            reduced_step = 1
        elif end_condition(x+dist_end_inflation, t, x_end, direction) == 0\
            and reduced_step == 1:
            dt = 1000*dt  # Go back to original step size
            sqrt_dt = dt**0.5#Remember to update the root
            reduced_step = 0
        elif reflect_condition(x, t, x_r, direction) == 1:
            x = reflection_value(x, x_r)
            num_reflects += 1

    if count_refs == True:
        return [t, e**(-A), num_reflects]
    else:
        return [t, e**(-A)]
   
# This version uses a general form for the bias. This function
# must both x and t as arguments.
cdef list simulation_general(double x_in, double x_r, double x_end,\
                             double t_i, double t_f, double dt, bias, drift,\
                             diffusion, count_refs = False):
    cdef double t, sqrt_dt, x, noise_amp, dist_end_inflation, dW, A, direction
    cdef int reduced_step = 0
    cdef int num_reflects = 0
    sqrt_dt = dt**0.5
    t = t_i
    x = x_in
    dist_end_inflation = 0.0
    A = 0.0
    direction = x_in-x_end

    while t<t_f:
        # Define the Wiener step
        dW = random.gauss(0.0, sqrt_dt)
        # Need to evaluate the general bias at this step
        bias_value = bias(x, t)
        # Step in x and A simultanioues
        [x, A] =\
            general_step(x, A, t, dt, dW, bias_value, drift, diffusion)
        # This is used to see the end surface has been approached
        noise_amp = diffusion(x, t)*sqrt_dt
        t += dt
        if end_condition(x, t, x_end, direction) == 1:  # Using 1/0 for True/False
            break
        elif end_condition(x+4*noise_amp, t, x_end, direction) == 1\
            and reduced_step == 0:
            dt = 0.001*dt
            sqrt_dt = dt**0.5
            dist_end_inflation = 4*noise_amp
            reduced_step = 1
        elif end_condition(x-4*noise_amp, t, x_end, direction) == 1\
            and reduced_step == 0:
            dt = 0.001*dt
            sqrt_dt = dt**0.5
            dist_end_inflation = -4*noise_amp
            reduced_step = 1
        elif end_condition(x+dist_end_inflation, t, x_end, direction) == 0\
            and reduced_step == 1:
            dt = 1000*dt  # Go back to original step size
            sqrt_dt = dt**0.5#Remember to update the root
            reduced_step = 0
        elif reflect_condition(x, t, x_r, direction) == 1:
            x = reflection_value(x, x_r)
            num_reflects += 1
            
    if count_refs == True:
        return [t, e**(-A), num_reflects]
    else:
        return [t, e**(-A)]


cpdef importance_sampling_simulations(double x_in, double x_r, double x_end,\
                     double t_i, double t_f, double dt, bias,\
                     int num_runs, drift, diffusion, bias_type = 'diffusion',\
                     count_refs = False):
    if bias_type == 'diffusion':
        results =\
            [simulation_diff(x_in, x_r,\
            x_end, t_i, t_f, dt, bias, drift, diffusion,\
            count_refs=count_refs) for i in range(num_runs)]
                
    elif bias_type == 'custom':
        results =\
            [simulation_general(x_in, x_r,\
            x_end, t_i, t_f, dt, bias, drift, diffusion,\
            count_refs=count_refs) for i in range(num_runs)]
    
    #Correctly slicing the list with list comprehension
    if count_refs == False:
        ts, ws = [[results[i][j] for i in range(num_runs)] for j in range(2)]
        return ts, ws
    elif count_refs == True:
        ts, ws, num_reflects = [[results[i][j] for i in range(num_runs)]\
                                for j in range(3)]
        return ts, ws, num_reflects
    else:
        raise ValueError('Unknown parameter for coundting reflection_values')
