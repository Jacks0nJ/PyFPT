# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 15:33:01 2021

    This is an attempt at creating a stochastic Runge-Kutta 4 method for
    inflationary models. Adeptive steps can still be used but after each step
    is completed, a noise term will be added. 
    
    There are a number of optional arguments which can be given, including 
    weather to print progess, if to use adaptive steps or not, as well as the
    type of error to use if using adaptive steps. The main one is the optional
    end condition and reflection for the stochastic code. The end condition
    only needs to return a Boolean, but the reflection condition also needs to
    return the value it is reflected to.
    
    The adaptive steps default to fractional errors (between the full step and 
    two half steps), but now have the option of using absolute errors. There
    is also an option for having a floor to the steps, but of course this can 
    be set to 0 if unwanted.
    
    Note, the 't' is a dummy varibe and can in fact represent any form of 
    'time' step. This currently is restricted to eing real and positive.
    
    The simulation defaults to ending at some defined finish time, but it can
    take arguments of a general ending condition. This means that the 
    simulation can break out of the while loop before the defined end time, be
    modiefied as needed, and then start again with the modifications. The print
    statements for the progression should be uneffected, as they only depend on
    the start and end times, rather than how many times the simulation has 
    exited to be modified.
    
    
@author: Joe Jackson
"""
import numpy as np

def time_step_size_estimator(phase_vecs, accel_vec, time, accuracy):
    #Note, 'accuracy' is the decimal size of the Euler step
    #Copying the input vector to prevent any chance of updating it
    _phase_vecs = np.copy(phase_vecs)
    _accel_vec_0 = accel_vec(_phase_vecs, time)
    #Finding 'velocity' matrix
    k_0 = __k__(_phase_vecs, _accel_vec_0)
    #Using the Euler method to estimate dt for each varible, a positive value
    dt_estimates_matrix = np.absolute(np.divide(_phase_vecs, k_0)*(accuracy))
    
    #Taking the minimum of these values as the initial estimate
    dt_best = np.amin(dt_estimates_matrix)
    
    #If rare case of this being zero, set to maximum step and let adeptive
    
    #steps do their work
    if dt_best == 0:
        dt_best = np.amax(phase_vecs)*accuracy
        
    return dt_best

# Each k calculation is only based on numbers, does not need function args
def __k__(_phase_vecs, accel_vec):
    #Creating k matrix by first setting the 'position' row to 'acceleration'
    _k = np.copy(_phase_vecs)
    _k[0] = accel_vec
    if _k.shape[0] ==1:
        return _k
    #Then reordering so 'velocity' is first, acceleration is 'last'
    indx = np.arange(1, _k.shape[0]+1)
    indx[indx.shape[0]-1] = 0
    _k =  _k[indx]
    return _k


#Need to pass it a function which can calcuate all of the different 
#'accelerations', i.e. a function which returns a vector of accelerations
def single_step(phase_vecs, dt, accel_vec, time):
    #Copying the input vector to prevent any chance of updating it
    _phase_vecs = np.copy(phase_vecs)
    dt_half = dt/2#Used as a short hand
    
    #Calculates the 'acceleration' and makes the time derivative matrix
    accel_vec_0 = accel_vec(_phase_vecs, time)
    k_0 = __k__(_phase_vecs, accel_vec_0)
    
    #First estimate of 'position/velocity' at half step, then 'acceleration'
    _phase_vecs1 = _phase_vecs + dt_half*k_0
    accel_vec_1 = accel_vec(_phase_vecs1, time+dt_half)
    k_1 = __k__(_phase_vecs1, accel_vec_1)
    
    #Second stimate of 'position/velocity' at half step, then 'acceleration'
    _phase_vecs2 = _phase_vecs + dt_half*k_1
    accel_vec_2 = accel_vec(_phase_vecs2, time+dt_half)
    k_2 = __k__(_phase_vecs2, accel_vec_2)
    
    #Estimate of the 'position/velocity' and 'acceleration' at full step
    _phase_vecs3 = _phase_vecs + dt*k_2
    accel_vec_3 = accel_vec(_phase_vecs3, time+dt)
    k_3 = __k__(_phase_vecs3, accel_vec_3)

    _phase_vecs += (dt/6)*(k_0+2*k_1+2*k_2+k_3)
    return _phase_vecs

#Same requirements as the RK4 step, but instead uses Euler method
def single_step_euler(phase_vecs, dt, accel_vec, time):
    #Copying the input vector to prevent any chance of updating it
    _phase_vecs = np.copy(phase_vecs)
    
    #Calculates the 'acceleration' and makes the time derivative matrix
    accel_vec_0 = accel_vec(_phase_vecs, time)
    k_0 = __k__(_phase_vecs, accel_vec_0)


    _phase_vecs += k_0*dt
    return _phase_vecs

# This just returns the result of a single step.
#Need to pass it a function which can calcuate all of the different 
#'accelerations', i.e. a function which returns a vector of accelerations
def just_the_single_step(phase_vecs, dt, accel_vec, time):
    #Copying the input vector to prevent any chance of updating it
    _phase_vecs = np.copy(phase_vecs)
    dt_half = dt/2#Used as a short hand
    
    #Calculates the 'acceleration' and makes the time derivative matrix
    accel_vec_0 = accel_vec(_phase_vecs, time)
    k_0 = __k__(_phase_vecs, accel_vec_0)

    #First estimate of 'position/velocity' at half step, then 'acceleration'
    _phase_vecs1 = _phase_vecs + dt_half*k_0
    accel_vec_1 = accel_vec(_phase_vecs1, time+dt_half)
    k_1 = __k__(_phase_vecs1, accel_vec_1)
    
    #Second stimate of 'position/velocity' at half step, then 'acceleration'
    _phase_vecs2 = _phase_vecs + dt_half*k_1
    accel_vec_2 = accel_vec(_phase_vecs2, time+dt_half)
    k_2 = __k__(_phase_vecs2, accel_vec_2)
    
    #Estimate of the 'position/velocity' and 'acceleration' at full step
    _phase_vecs3 = _phase_vecs + dt*k_2
    accel_vec_3 = accel_vec(_phase_vecs3, time+dt)
    k_3 = __k__(_phase_vecs3, accel_vec_3)
    
    _step = (dt/6)*(k_0+2*k_1+2*k_2+k_3)
    return _step


#Makes two half steps, should have increased accuracy
def just_the_two_half_steps(phase_vecs, dt, accel_vec, time):
    #Copying the input vector to prevent any chance of updating it
    _phase_vecs = np.copy(phase_vecs)
    dt_half = dt/2#Used as a short hand

    #Using the standard single step calculator to find the half step
    first_half_step = just_the_single_step(_phase_vecs, dt_half, accel_vec,\
                                           time)
    #Now adding this half step onto the position to find full step
    second_half_steps = just_the_single_step(phase_vecs+first_half_step,\
                                             dt_half, accel_vec, time+dt_half)
    two_half_steps = first_half_step + second_half_steps
    return two_half_steps

#This calculates the best POSITION
def best_single_step_for_dt(phase_vecs, dt, accel_vec, time, error_type):
    _dt = np.copy(dt)#Prevent that stupid error
    
    #The two steps, 1 being two half steps
    _one_step = just_the_single_step(phase_vecs, _dt, accel_vec, time)
    _two_half_steps = just_the_two_half_steps(phase_vecs, _dt, accel_vec, time)

    #Using both of these values to get a weighted average estimate of step
    _best_step = __weighted_average__(_one_step, _two_half_steps)
    
    #Estimating the errors, for every varible. Can be fractional or absolute
    if error_type == 'absolute':
        errors = __error_calculator_absolute__(_one_step,_two_half_steps)
    elif error_type == 'fractional':
        errors = __error_calculator_fractional__(_one_step,_two_half_steps)
    else:
        print('Unknown error type input, must be \"fractional\" ' +\
              'or \"absolute\". Defaulting to fractional error typea.')
        errors = __error_calculator_fractional__(_one_step,_two_half_steps)
    
    #Now using this best estimate step to find new 'position'
    best_phase_vector = phase_vecs+_best_step
    return best_phase_vector, errors

#This uses the Euler-Maruyama method to make a stochastic step, see Wikipedia
def sto_euler_maruyama_step(phase_vecs, dt, accel_vec, noise_vec, time):
    noise_amp = noise_vec(phase_vecs, time)
    #Adding the noise terms for this estimated step
    #Rememer white noise scales with the square root
    new_phase_vec = single_step_euler(phase_vecs, dt, accel_vec, time) +\
        noise_amp*np.random.normal()*np.sqrt(dt)
    return new_phase_vec, noise_amp
    

#Like the Euler-Maruyama method, but ueses RK4 to make the drift step
def sto_rk_maruyama_step(phase_vecs, dt, accel_vec, noise_vec, time):
    noise_amp = noise_vec(phase_vecs, time)
    #Adding the noise terms for this estimated step
    #Rememer white noise scales with the square root
    new_phase_vec = single_step(phase_vecs, dt, accel_vec, time) +\
        noise_amp*np.random.normal()*np.sqrt(dt)
    return new_phase_vec, noise_amp

#Uses the Stochastic Runge-Kutta Approximation, see Wikipedia
def sto_rk_approx_step(phase_vecs, dt, accel_vec, noise_vec, time):
    dt_sqrt = np.sqrt(dt)
    noise_amp = noise_vec(phase_vecs, time)
    euler_step = single_step_euler(phase_vecs, dt, accel_vec, time)
    euler_sto_term = noise_amp*dt_sqrt
    gauss_noise = np.random.normal()
    higher_order_term = 0.5*(noise_vec(euler_step+euler_sto_term, time+dt) -\
                             noise_amp)*((gauss_noise**2)-1)*dt_sqrt
    new_phase_vec = euler_step+euler_sto_term*gauss_noise + higher_order_term
    
    return new_phase_vec, noise_amp

#Uses the improved Euler method, found at arXiv:1210.0933v1
def sto_euler_improved_step(phase_vecs, dt, accel_vec, noise_vec, time):
    #Terms unqiue to this method
    noise_amp1 = noise_vec(phase_vecs, time)
    gauss_noise = np.random.normal()
    s = np.random.choice([-1,1])#Randomly +/- 1
    accel_vec_1 = accel_vec(phase_vecs, time)
    #
    sqrt_dt = np.sqrt(dt)
    k_1_drift = __k__(phase_vecs, accel_vec_1)
    k_1_sto = k_1_drift*dt + noise_amp1*(gauss_noise-s)*sqrt_dt
    
    phase_vecs1 = phase_vecs+k_1_sto
    
    noise_amp2 = noise_vec(phase_vecs1, time+dt)
    accel_vec_2 = accel_vec(phase_vecs1, time+dt)
    k_2_drift = __k__(phase_vecs1, accel_vec_2)
    k_2_sto = k_2_drift*dt + noise_amp2*(gauss_noise+s)*sqrt_dt
    
    new_phase_vec = phase_vecs + np.mean([k_1_sto, k_2_sto])
    noise_amp = np.mean([noise_amp1, noise_amp2])
    return new_phase_vec, noise_amp

#This is general and can return a numpy matrix
def __error_calculator_fractional__(answer_1, answer_2):
    _answer_1 = np.copy(answer_1)#To prevent data referance error
    _answer_2 = np.copy(answer_2)
    
    #Assuming answer_2 is correct
    error = np.absolute(np.divide(_answer_1-_answer_2, _answer_2))
    return error

#This is for when the absolute error is used, returns numpy array
def __error_calculator_absolute__(answer_1, answer_2):
    _answer_1 = np.copy(answer_1)#To prevent data referance error
    _answer_2 = np.copy(answer_2)
    
    #Assuming answer_2 is correct
    error_absolute = _answer_2 - _answer_1
    return error_absolute
    

#Returns the weighted average for the RK4 method for a dt and 2 dt/2 steps
def __weighted_average__(answer_1, answer_2):
    #Assuming answer_2 is the better one
    best_answer = (16*answer_2 - answer_1)/15
    return best_answer

#This is a classical adaptive step (does not account for stochastic noise)
def adeptive_step(phase_vecs, dt, accel_vec, time, tol, error_type, dt_min):
    _dt = np.copy(dt)#Prevent that stupid error
    _phase_vecs = np.copy(phase_vecs)
    
    best, errors = best_single_step_for_dt(_phase_vecs,\
                                           _dt, accel_vec, time, error_type)
    max_error = np.amax(errors)
    #If step too large for tolerance
    #This is because if the anye rror >tol, the largest one must be
    if max_error > tol:
        num_iters = 0
        min_step_used = False
        while max_error > tol and min_step_used==False:
            #Need smaller step sizes to be in tolerance, so halfing
            _dt = _dt/2
            
            #This gives a floor to how small the steps can be, forces loop exit
            if _dt < dt_min:
                _dt = dt_min
                min_step_used = True
            
            best, errors = best_single_step_for_dt(_phase_vecs, _dt,\
                                                   accel_vec, time, error_type)
            max_error = np.amax(errors)
            num_iters += 1
            if num_iters > 49:
                ##
                ##
                #Need better error throwing
                ##
                ##
                raise ValueError('Infinite loop error: decreasing steps')
    #If step is in effciently large and all errors small, increasing the step.
    # This is because if the largest error is less than tol, all others are
    elif max_error < tol:
        num_iters = 0
        while max_error < tol:
            #Step size too small
            _dt_temp = 2*_dt
            best_temp, errors = best_single_step_for_dt(_phase_vecs, _dt_temp,\
                                                   accel_vec, time, error_type)
            max_error = np.amax(errors)
            num_iters += 1
            if num_iters > 49:
                raise ValueError('Infinite loop error: increasing steps')
            #If the new step is still within tol, save this tentative step
            elif max_error < tol:
                best = best_temp
                _dt = _dt_temp
    else:
        print('Errors are:')
        print(errors)
        print('This caused an error in the RK4 simulation.')
        print(best)
        raise ValueError('Unknown adeptive step error')

    return best, _dt


#Allows the error type to be checked
def _check_error_type(_error_type):
    if _error_type != 'absolute' and _error_type != 'fractional':
        print('Unknown error type input, must be \"fractional\" ' +\
              'or \"absolute\". Defaulting to fractional error typea.')
        _error_type = 'fractional'
    return _error_type

#Function to check the progression of the simulation
def check_sim_progression(time_vec, t_i, length_of_sim, tick):
    if (time_vec[-1,0,0]-t_i)>(length_of_sim/4) and tick<1:
            print('Simulation 25% done')
            tick = 1
    elif (time_vec[-1,0,0]-t_i)>(length_of_sim/2) and tick<2:
        print('Simulation 50% done')
        tick = 2
    elif (time_vec[-1,0,0]-t_i)>(3*length_of_sim/4) and tick<3:
        print('Simulation 75% done')
        tick = 3
            
    return tick

#This is a modification to the general function, which efficently solves the
#1D case of a stochastic differential equation, thus 1D intial input.
def simulation_stochastic_efficent(phase_matrix, accel_vec,\
                                   noise_vec, t_i, dt_input, end_cond,\
                                       reflect_cond=None):      
    #Copying to avoid errors
    current_position = np.copy(phase_matrix)
    t = t_i
    dt = dt_input
    min_step = False
    dist_inflation_end = 0.0
    for i in range(10**6):#Compare to last time found
         
        noise = noise_vec(current_position, t)
        #Adding the noise terms for this estimated step
        #Rememer white noise scales with the square root
        current_position = single_step(current_position, dt, accel_vec, t) +\
            noise*np.random.normal()*np.sqrt(dt)
                   
        #First store data for this step
        #Update time
        t += dt
        
        #Check if reflected and update accordingly
        if reflect_cond != None:
            will_reflect, reflect_value = reflect_cond(current_position, t+dt)
            if will_reflect == True:
                #Using the reflected value for the step
                current_position = reflect_value


        #Stop if end surface is passed
        if end_cond(current_position, t) == True:
            print('Stopped at ', t)
            break
        elif end_cond(current_position+3*noise*np.sqrt(dt), t+dt) == True and\
             min_step == False:
                 
            dt = dt*(10**(-3))
            dist_inflation_end = 3*noise*np.sqrt(dt)
            min_step = True
        elif end_cond(current_position-3*noise*np.sqrt(dt), t+dt) == True and\
             min_step == False:
                 
            dt = dt*(10**(-3))
            dist_inflation_end = -3*noise*np.sqrt(dt)
            min_step = True
        #If stochastically exited end region
        elif end_cond(current_position+dist_inflation_end,t+dt)==False and\
            min_step == True:
            #Change back to larger step
            dt = dt_input
            #Reset distance to end
            dist_inflation_end = 0.0
            min_step = False
                 
    return current_position, t

def simulation_looping(phase_matrix, accel_vec, t_i, t_f, dt,\
                           tol=None, dt_min=None, end_cond = None,\
                           reflect_cond = None,\
                           noise_vec = None, error_type='fractional',\
                           step_method = 'RK4', store_steps = False):
    #Sorting through the optional arguments
    if dt_min is None:
        dt_min = dt*10**(-3)
    
    if end_cond is None:
        end_cond = _null_end_condition
    
    #The choice of steps
    if noise_vec == None:
        #Defining the step method used
        if step_method == 'euler':
            step_method = single_step_euler
        else:
            step_method = single_step
    #If stochastic simulation, use stochastic steps
    elif noise_vec != None:
                #Defining the step method used
        if step_method == 'euler_maruyama':
            step_method = sto_euler_maruyama_step
        elif step_method == 'stochastic_rk':
            step_method = sto_rk_approx_step
        elif step_method == 'improved_euler':
            step_method = sto_euler_improved_step
        else:
            step_method = sto_rk_maruyama_step
    
    #Storing the steps
    if store_steps == True:
        phase_storage = np.copy(phase_matrix)
        time_storage = np.copy(t_i)
        
    #Varibles that need defining outside loop
    min_step = False
    dist_inflation_end = 0.0
    dt_input = dt#Storing for referance
    current_phase = phase_matrix
    t = t_i
    while t<t_f:#Compare to last time found
        #If classical simulation
        if noise_vec == None:
            #Checking if adaptive steps used, thus a tolerance specifed
            if tol != None:
                #Make an adeptive step and set time step equal to that found
                new_phase, dt = adeptive_step(current_phase, dt,\
                                                  accel_vec, t, tol,\
                                                      error_type, dt_min)
                #Checking if min step floor was reached for adaptive steps
                if dt == dt_min and min_step == False:
                    print('Minimum floor of \'time\' steps reached')
                    min_step = True
            else:#Just use a fixed time step
                new_phase = step_method(current_phase, dt, accel_vec, t)
                
            #If still less than t_f and end condition unmet, continue
            if  end_cond(new_phase, t+dt) == False and (t+dt)<t_f:
                #Update time
                t += dt
                #Update to current position
                current_phase = new_phase
                #If storing step data
                if store_steps == True:
                    #Change data to correct format, using underscore to denote
                    #change
                    _phase = np.dstack(np.copy(new_phase)).transpose(0,2,1)
                    _t = np.dstack(np.array([t])).transpose(0,2,1)
                    #Store this data
                    phase_storage = np.concatenate((phase_storage, _phase),\
                                                   axis=0)
                    time_storage = np.concatenate((time_storage, _t), axis=0)
            #Over stepped the time, can then finish exactly at t_f
            elif (t+dt)>=t_f:
                dt = t_f - t
                #Step stopping at t_f, updating to value at t_f
                current_phase = single_step(current_phase, dt, accel_vec, t)
                #Stoiring values as normal
                #Update time
                t += dt
                #If storing step data
                if store_steps == True:
                    #Change data to correct format, using underscore to denote
                    #change
                    _phase = np.dstack(np.copy(new_phase)).transpose(0,2,1)
                    _t = np.dstack(np.array([t])).transpose(0,2,1)
                    #Store this data
                    phase_storage = np.concatenate((phase_storage, _phase),\
                                                   axis=0)
                    time_storage = np.concatenate((time_storage, _t), axis=0)
                #Declaring this run is stopped
                print('Stopped at ', t)
                break
            #Met end condition
            elif end_cond(current_phase, t+dt) == True:
                #If already at minimum step, store values. Equally, If using
                #adaptive steps, simply store and exit 
                if dt <= dt_min or tol != None:
                    #Updating values
                    t += dt
                    new_phase = current_phase
                    #If storing step data
                    if store_steps == True:
                        #Change data to correct format, using underscore to denote
                        #change
                        _phase = np.dstack(np.copy(new_phase)).transpose(0,2,1)
                        _t = np.dstack(np.array([t])).transpose(0,2,1)
                        #Store this data
                        phase_storage = np.concatenate((phase_storage, _phase),\
                                                       axis=0)
                        time_storage = np.concatenate((time_storage, _t), axis=0)
                    break 
                #Half time step instead, so end condiition can be approached
                else:
                    dt = dt/2
                    #Redoing step, so not storing values
                    break
            else:
                ValueError('Unknown error approaching the end of simulation.')
        #Stochastic case, i.e. this is Langevin equation
        elif noise_vec != None:
             
            #Adding the noise terms for this estimated step
            #Rememer white noise scales with the square root
            current_phase, noise = step_method(current_phase, dt,\
                                                    accel_vec, noise_vec, t)
                       
            #First store data for this step
            #Update time
            t += dt
            
            #Check if reflected and update accordingly
            if reflect_cond != None:
                will_reflect, reflect_value = reflect_cond(current_phase, t+dt)
                if will_reflect == True:
                    #Using the reflected value for the step
                    current_phase = reflect_value
                    
            #If storing step data
            if store_steps == True:
                #Change data to correct format, using underscore to denote
                #change
                _phase = np.dstack(np.copy(current_phase)).transpose(0,2,1)
                _t = np.dstack(np.array([t])).transpose(0,2,1)
                #Store this data
                phase_storage = np.concatenate((phase_storage, _phase),\
                                               axis=0)
                time_storage = np.concatenate((time_storage, _t), axis=0)
    
            #Stop if end surface is passed
            if end_cond(current_phase, t) == True:
                print('Stopped at ', t)
                break
            elif end_cond(current_phase+3*noise*np.sqrt(dt), t+dt) == True and\
                 min_step == False:
                     
                dt = dt*(10**(-3))
                dist_inflation_end = 3*noise*np.sqrt(dt)
                min_step = True
            elif end_cond(current_phase-3*noise*np.sqrt(dt), t+dt) == True and\
                 min_step == False:
                     
                dt = dt*(10**(-3))
                dist_inflation_end = -3*noise*np.sqrt(dt)
                min_step = True
            #If stochastically exited end region
            elif end_cond(current_phase+dist_inflation_end,t+dt)==False and\
                min_step == True:
                #Change back to larger step
                dt = dt_input
                #Reset distance to end
                dist_inflation_end = 0.0
                min_step = False
        else:
            ValueError('Not classic or stochastic code')
    
    #Return end state if steps not stored  
    if store_steps == False:
        return current_phase, t
    #Return end state and all the previus step values
    elif store_steps == True:
        return phase_storage, time_storage




#This is a full simulation from start to finish, with options to make the
#simulation stochastic and add custom end conditions
def full_simulation(phase_vecs_i, accel_vec, t_f, t_i=0.0, dt=None,\
                    tol=None, dt_min = None, end_cond=None,\
                    reflect_cond = None, noise_vec=None,\
                    error_type='fractional', step_method = 'RK4',\
                    store_steps = False):
    
    
    #If using adaptibe steps     
    if tol != None and dt == None:
        #Estimate the optimal time step, using the tol as accuracy
        dt = time_step_size_estimator(phase_vecs_i, accel_vec, t_i, 0.001)
    #When nothing is specified
    elif tol == None and dt == None:
        raise ValueError('Did not specify constant or adaptive steps')
        
    if isinstance(phase_vecs_i, float):
        phase_vecs_i = np.array([phase_vecs_i])
        
    if store_steps == False:
        phase_matrix = phase_vecs_i
        time_vec = t_i
    elif store_steps == True:
        #Defining 3D arrays to store the simulation data
        phase_matrix = np.dstack(np.copy(phase_vecs_i)).transpose(0,2,1)
        time_vec = np.dstack((np.array([t_i]))).transpose(0,2,1)
        #Checking error type
    error_type = _check_error_type(error_type)
    
    ###Running the simulation
    
    phase_matrix, time_vec =\
        simulation_looping(phase_matrix, accel_vec, time_vec, t_f, dt,\
                               tol=tol, dt_min=dt_min, end_cond=end_cond,\
                               reflect_cond=reflect_cond,\
                               noise_vec = noise_vec, error_type=error_type,\
                               step_method=step_method,\
                               store_steps = store_steps)
      
    return phase_matrix, time_vec


    
def _null_end_condition(phi_mat, time_v):
    return False



'''Old method
def simulation_looping_all_old(phase_matrix, accel_vec, time_vec, t_f, dt,\
                           tol=None, dt_min=None, end_cond = None,\
                           reflect_cond = None, print_progress = False,\
                           noise_vec = None, error_type='fractional',\
                           step_method = 'RK4'):
    #Sorting through the optional arguments
    if dt_min is None:
        dt_min = dt*10**(-3)
    
    if end_cond is None:
        end_cond = _null_end_condition
        
    if step_method == 'euler':
        step_method = single_step_euler
    else:
        step_method = single_step
        
    #Copying to avoid errors
    _phase_matrix = np.copy(phase_matrix)
    _time_vec = np.copy(time_vec)
    #Varibles that need defining outside loop
    min_step_floor = False
    t_i = _time_vec[0,0,0]#The first value was the start of the simulation.
    length_of_sim = t_f-t_i
    dist_inflation_end_sto = 0.0
    dt_input = dt#Storing for referance
    tick = 0
    while _time_vec[-1,0,0]<t_f:#Compare to last time found
        #Defining most recent position and time
        t = _time_vec[-1,0,0]
        _phase_vec = _phase_matrix[-1,:,:]
        
        #Checking if adaptive steps used, thus a tolerance specifed
        if tol != None:
            #Make an adeptive step and set time step equal to that found
            step, dt = adeptive_step(_phase_vec, dt, accel_vec, t, tol,\
                                 error_type, dt_min)
            #Checking if min step floor was reached for adaptive steps
            if dt == dt_min and min_step_floor == False:
                print('Minimum floor of \'time\' steps reached')
                min_step_floor = True
        else:#Just use a fixed time step
            step = step_method(_phase_vec, dt, accel_vec, t)
            
            
        #Adding noise term if stochastic simulation
        if noise_vec is not None:
            noise = noise_vec(_phase_vec, t)
            #Adding the noise terms for this estimated step
            #Rememer white noise scales with the square root
            step += noise*np.random.normal()*np.sqrt(dt)
        
        ###
        Smartly storing the data, depending if stochastic or not
        ###
        
        #If non-stochastic simulation
        if noise_vec is None:
            #If still less than t_f and end condition unmet, continue
            if  end_cond(step, t+dt) == False and (t+dt)<t_f:
                #Update time
                t += dt
                #Change data to correct format, using underscore to denote change
                _step = np.dstack(np.copy(step)).transpose(0,2,1)
                _t = np.dstack(np.array([t])).transpose(0,2,1)
                #Store this data
                _phase_matrix = np.concatenate((_phase_matrix, _step), axis=0)
                _time_vec = np.concatenate((_time_vec, _t), axis=0)
            #Over stepped the time, can then finish exactly at t_f
            elif (t+dt)>=t_f:
                dt = t_f - t
                #Step stopping at t_f
                step = single_step(_phase_vec, dt, accel_vec, t)
                #Stoiring values as normal
                #Update time
                t += dt
                #Change data to correct format, using underscore to denote change
                _step = np.dstack(np.copy(step)).transpose(0,2,1)
                _t = np.dstack(np.array([t])).transpose(0,2,1)
                #Store this data
                _phase_matrix = np.concatenate((_phase_matrix, _step), axis=0)
                _time_vec = np.concatenate((_time_vec, _t), axis=0)
                
                #Declaring this run is stopped
                print('Stopped at ', t)
                break
            #Met end condition
            elif end_cond(step, t+dt) == True:
                    
                #If already at minimum step, store values. Equally, If using
                #adaptive steps, simply store and exit 
                if dt <= dt_min or tol != None:
                    t += dt
                    #Change data to correct format, using underscore to denote change
                    _step = np.dstack(np.copy(step)).transpose(0,2,1)
                    _t = np.dstack(np.array([t])).transpose(0,2,1)
                    #Store this data
                    _phase_matrix = np.concatenate((_phase_matrix, _step), axis=0)
                    _time_vec = np.concatenate((_time_vec, _t), axis=0)
                    break 
                #Half time step instead, so end condiition can be approached
                else:
                    dt = dt/2
                    #Redoing step, so not storing values
                    break
            else:
                ValueError('Unknown error approaching the end of simulation.')
                
        #If stochastic simulation
        elif noise_vec is not None:
            if end_cond(step,t+dt)==False:
                #First store data for this step
                
                #Update time
                t += dt
                #Change data to correct format
                _step = np.dstack(np.copy(step)).transpose(0,2,1)
                _t = np.dstack(np.array([t])).transpose(0,2,1)
                #Store this data
                _phase_matrix = np.concatenate((_phase_matrix, _step), axis=0)
                _time_vec = np.concatenate((_time_vec, _t), axis=0)
                
                

                #Check if reflected and update accordingly
                if reflect_cond != None:
                    will_reflect, reflect_value = reflect_cond(step, t+dt)
                    if will_reflect == True:
                        #Using the reflected value for the step
                        step_r = reflect_value
                        #Change data to correct format
                        _step_r = np.dstack(np.copy(step_r)).transpose(0,2,1)
                        #Store reflected value instead
                        _phase_matrix[-1,:,:] = _step_r
                        
                        
                #Check if approaching the end surface and still large step,
                #change to minimum step and store distance to this surface
                
                #If 0.03% channce of noise causing crossing
                large_noise = 3*noise*np.sqrt(dt)
                if end_cond(step-large_noise,t+dt)==True and\
                    min_step_floor == False:  
                    #Change to minimum step
                    dt = dt_min
                    #Store the distance to the end of inflation, negative case
                    dist_inflation_end_sto = -large_noise
                    min_step_floor = True
                #checking the other side
                elif end_cond(step+large_noise,t+dt)==True and\
                    min_step_floor == False:
                    #Change to minimum step
                    dt = dt_min
                    #Store the distance to the end of inflation, postive case
                    dist_inflation_end_sto = large_noise
                    min_step_floor = True
                #If using small time step and exited end region
                elif end_cond(step+dist_inflation_end_sto,t+dt)==False and\
                    min_step_floor == True:
                    #Change back to larger step
                    dt = dt_input
                    #Reset distance to end
                    dist_inflation_end_sto = 0.0
                    min_step_floor = False
                    
            #Check if simulation is over
            elif end_cond(step,t+dt)==True:
                #Update time
                t += dt
                #Change data to correct format
                _step = np.dstack(np.copy(step)).transpose(0,2,1)
                _t = np.dstack(np.array([t])).transpose(0,2,1)
                #Store this data
                _phase_matrix = np.concatenate((_phase_matrix, _step), axis=0)
                _time_vec = np.concatenate((_time_vec, _t), axis=0)
                #End simulatuion
                print('Stopped at ', t)
                break
                    
                    
                
        #Check if print progress
        if print_progress == True:
            tick = check_sim_progression(_time_vec, t_i, length_of_sim, tick)
        
        #Stop the simulation getting carried away
        if np.shape(_time_vec)[0] > np.array([10**7]):
            raise ValueError('Too many adeptive steps')        
        
    return _phase_matrix, _time_vec

'''

