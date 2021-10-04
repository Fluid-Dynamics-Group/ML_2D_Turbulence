import numpy as np
import matplotlib.pyplot as plt
import time
from numpy import f2py

#Standard functions
from Fortran_Objects import Fortran_Functions, Spectral_Poisson
#, Multigrid_Solver

#Standard turbulence closures
from Fortran_Objects import Standard_Models
from Fortran_Objects import Relaxation_Filtering

#F2PY objects
#from Fortran_Objects import Ml_Convolution, ML_Regression, ML_AD_Classification
#from Fortran_Objects import ML_Nearest_Neighbors, ML_Feature_Functions, ML_Logistic_Functions
#from Fortran_Objects import ML_TBDNN

# Temporal trackers
from Fortran_Objects import Temporal_Tracker

#Seeds
np.random.seed(10)

#-------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------#
#Standard functions for solver and inputs
#-------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------#
def init_domain():
    global nx, ny, kappa, Re_n, lx, ly, lt, nt, dt, dx, dy, gs_tol
    global problem
    global sigma#For AD-LES
    global closure_choice
    global n_samples #For ML-AD-LES framework
    global class_tracker, class_use
    
    nx = 256
    ny = 256
    kappa = 2.0
    Re_n = 32000.0
    kappa = 2.0
    lx = 2.0 * np.pi
    ly = 2.0 * np.pi
    dx = lx / float(nx)
    dy = ly / float(ny)#Not nx-1 since last point is to be assumed within the domain
    sigma = 0.49 #Filter width for AD (Gaussian - 1.0/2.0/3.0), Filter width for Pade (0 - 0.499)

    gs_tol = 1.0e-2#Keep low for GS, keep high for MG

    n_samples = 8000#For ML - AD - LES classification

    problem = 'HIT'
    '''
    Closure choices:
    0 - UDNS
    5 - Smagorinsky
    10 - Dynamic Smagorinsky (Comput. Fluids 2017 formulation)
    '''
    closure_choice = 10
    class_use = False

    lt = 1.0 # Final time

    dt = 1.0e-3
    nt = int(lt/dt)

    omega = np.zeros(shape=(nx,ny),dtype='double', order='F')
    psi = np.zeros(shape=(nx, ny), dtype='double', order='F')

    return omega, psi

def initialize_ic_bc(omega, psi):
    Fortran_Functions.hit_init_cond(omega, dx, dy)
    Spectral_Poisson.solve_poisson(psi, -omega, dx, dy)

def post_process(omega,psi):

    fig, ax = plt.subplots(nrows=1,ncols=1)

    #levels = np.linspace(-50,50,10)

    ax.set_title("Numerical Solution - HIT")
    plot1 = ax.contourf(omega[:, :])#,levels=levels)
    plt.colorbar(plot1, format="%.2f")

    plt.show()

    arr_len = int(0.5*np.sqrt(float(nx*nx + ny*ny)))-1
    eplot = np.zeros(arr_len+1,dtype='double')
    kplot = np.arange(0,arr_len+1,1,dtype='double')

    Fortran_Functions.spec(omega,eplot)

    scale_plot = np.array([[10,0.1],[100,1.0e-4]])

    plt.loglog(kplot,eplot)
    plt.loglog(scale_plot[:,0],scale_plot[:,1])
    plt.xlim([1,1.0e3])
    plt.ylim([1e-8,1])
    plt.xlabel('k')
    plt.ylabel('E(k)')
    plt.title('Angle averaged energy spectra')
    plt.show()

    np.save('Field.npy',omega)
    np.save('Spectra.npy',[kplot, eplot])


def deploy_model_classification(omega,ml_model):

    #Sampling the field randomly for omega stencils
    sampling_matrix = ML_AD_Classification.field_sampler(omega, n_samples)
    #Classify
    return_matrix = ml_model([sampling_matrix])[0]#Using precompiled ML network
    #Summing up different predictions for filter strength
    return_matrix = np.sum(return_matrix,axis=0)
    #Finding the filter strength with max predictions
    bucket = np.argmax(return_matrix,axis=0)

    #Modifying the global sigma
    global sigma

    #Note inverse relationship with training
    if bucket == 0:
        sigma = 2.0
    elif bucket == 1:
        sigma = 1.5
    elif bucket == 2:
        sigma = 1.0

def tvdrk3_fortran_smag_sgs(omega,psi):

    oneth = 1.0 / 3.0
    twoth = 2.0 / 3.0

    # Step 1
    # Calculate RHS
    f = Fortran_Functions.rhs_periodic(psi, omega, dx, dy, Re_n)
    #Standard Smag
    sgs = smag_sgs_calc(omega,psi)

    omega_1 = omega + dt * (f + sgs)

    # Fortran update for Poisson Equation
    #Multigrid_Solver.solve_poisson_periodic(psi, -omega_1, dx, dy, gs_tol)
    Spectral_Poisson.solve_poisson(psi,-omega_1, dx, dy)

    # Step 2
    # Calculate RHS
    f = Fortran_Functions.rhs_periodic(psi, omega_1, dx, dy, Re_n)
    # Standard Smag
    sgs = smag_sgs_calc(omega_1,psi)
    omega_2 = 0.75 * omega + 0.25 * omega_1 + 0.25 * dt * (f + sgs)

    # Fortran update for Poisson Equation
    Spectral_Poisson.solve_poisson(psi,-omega_2, dx, dy)

    # Step 2
    # Calculate RHS
    f = Fortran_Functions.rhs_periodic(psi, omega_2, dx, dy, Re_n)
    # Standard Smag
    sgs = smag_sgs_calc(omega_2,psi)

    omega[:, :] = oneth * omega[:, :] + twoth * omega_2[:, :] + twoth * dt * (f[:, :] + sgs[:, :])

    # Fortran update for Poisson Equation
    Spectral_Poisson.solve_poisson(psi,-omega, dx, dy)

def deploy_model_logistic_blended(omega,psi,ml_model):

    #Sampling the field randomly for omega stencils
    sampling_matrix = ML_Logistic_Functions.field_sampler(omega, psi)
    #Classify - softmax
    return_matrix = ml_model([sampling_matrix])[0]#Using precompiled ML network

    # Smagorinsky model estimate
    sgs_smag = Standard_Models.smag_source_term(omega, psi, dx, dy)

    # AD estimate
    sfs_ad = Standard_Models.approximate_deconvolution(omega, psi, dx, dy, sigma)

    # Defining the sgs array
    sgs = np.zeros(shape=(nx,ny),dtype='double', order='F')
    # Calculating the optimal SGS model according to the prediction
    ML_Logistic_Functions.sgs_calculate_blended(return_matrix, sgs, sgs_smag, sfs_ad)

    # Track classifications
    classification_tracker(return_matrix)

    return sgs

def deploy_model_logistic_blended_five_class(omega,psi,ml_model):

    #Sampling the field randomly for omega stencils
    sampling_matrix = ML_Logistic_Functions.field_sampler(omega, psi)
    #Classify - softmax
    return_matrix = ml_model([sampling_matrix])[0]#Using precompiled ML network

    # Smagorinsky model estimate
    sgs_leith = Standard_Models.leith_source_term(omega, psi, dx, dy)

    # Smagorinsky model estimate
    sgs_smag = Standard_Models.smag_source_term(omega, psi, dx, dy)

    # AD estimate
    sfs_ad = Standard_Models.approximate_deconvolution(omega, psi, dx, dy, sigma)

    # AD estimate
    sfs_bd = Standard_Models.bardina(omega, psi, dx, dy, sigma)

    # Defining the sgs array
    sgs = np.zeros(shape=(nx,ny),dtype='double', order='F')
    # Calculating the optimal SGS model according to the prediction
    ML_Logistic_Functions.sgs_calculate_blended_five_class(return_matrix, sgs, sgs_leith, sgs_smag, sfs_ad, sfs_bd)

    # Track classifications
    classification_tracker(return_matrix)

    return sgs

def deploy_model_tbdnn(omega,psi,ml_model):

    # Validated - samples the entire field for omega, psi stencil - 18 inputs
    sampling_matrix = ML_TBDNN.field_sampler(omega, psi)

    # Smagorinsky model estimate
    sgs_smag = Standard_Models.smag_source_term(omega, psi, dx, dy)

    # Leith model estimate
    sgs_leith = Standard_Models.leith_source_term(omega, psi, dx, dy)

    # AD estimate
    sfs_ad = Standard_Models.approximate_deconvolution(omega, psi, dx, dy, sigma)

    # Reshape sgs_smag, sfs_ad to new placeholder
    basis_matrix = ML_TBDNN.basis_matrix_shaper(sgs_smag,sgs_leith,sfs_ad)

    # Prediction from precompiled keras function - validated
    return_matrix = ml_model([sampling_matrix,basis_matrix])[0]

    print(np.shape(return_matrix))

    # Defining the sgs array
    sgs = np.zeros(shape=(nx,ny),dtype='double', order='F')

    # Reshaping keras prediction in sgs array - no truncation here
    ML_TBDNN.sgs_reshape(return_matrix,sgs)

    return sgs

def deploy_model_par_log(omega,psi,ml_model):

    #Sampling the field randomly for omega stencils
    w_sampling_matrix = ML_Logistic_Functions.field_sampler_f(omega)
    s_sampling_matrix = ML_Logistic_Functions.field_sampler_f(psi)
    #Classify - softmax
    return_matrix = ml_model([w_sampling_matrix,s_sampling_matrix])[0]#Using precompiled ML network

    # Smagorinsky model estimate
    sgs_smag = Standard_Models.smag_source_term(omega, psi, dx, dy)

    # AD estimate
    sfs_ad = Standard_Models.approximate_deconvolution(omega, psi, dx, dy, sigma)

    # Defining the sgs array
    sgs = np.zeros(shape=(nx,ny),dtype='double', order='F')
    # Calculating the optimal SGS model according to the prediction
    ML_Logistic_Functions.sgs_calculate(return_matrix, sgs, sgs_smag, sfs_ad)

    return sgs

#-------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------#
#Standard eddy-viscosity closures/ AD
#-------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------#
def smag_sgs_calc(omega,psi):

    sgs = Standard_Models.smag_source_term_div(omega,psi,dx,dy)
    return sgs

def leith_sgs_calc(omega,psi):

    sgs = Standard_Models.leith_source_term_div(omega,psi,dx,dy)
    return sgs

def ad_sfs_calc(omega,psi):
    sfs = Standard_Models.approximate_deconvolution(omega,psi,dx,dy,sigma)
    return sfs

def bd_sfs_calc(omega,psi):
    sfs = Standard_Models.bardina(omega,psi,dx,dy,sigma)
    return sfs

#-------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------#
#Dynamic eddy-viscosity closures
#-------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------#
def dynamic_smag_calc(omega,psi):

    laplacian = Fortran_Functions.laplacian_calculator(omega, dx, dy)
    sgs = Standard_Models.dynamic_smagorinsky(omega,psi,laplacian,dx,dy)

    return sgs

def dynamic_leith_calc(omega,psi):

    laplacian = Fortran_Functions.laplacian_calculator(omega, dx, dy)
    sgs = Standard_Models.dynamic_leith(omega,psi,laplacian,dx,dy)

    return sgs

#-------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------#
#Runge Kutta third-order for different closures
#-------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------#
def tvdrk3_fortran_rf_les(omega,psi):

    Relaxation_Filtering.filter_pade(omega,sigma)

    oneth = 1.0 / 3.0
    twoth = 2.0 / 3.0

    # Step 1
    # Calculate RHS
    f = Fortran_Functions.rhs_periodic(psi, omega, dx, dy, Re_n)
    omega_1 = omega + dt * (f)

    # Fortran update for Poisson Equation
    # Multigrid_Solver.solve_poisson_periodic(psi, -omega_1, dx, dy, gs_tol)
    Spectral_Poisson.solve_poisson(psi,-omega_1, dx, dy)

    # Step 2
    # Calculate RHS
    f = Fortran_Functions.rhs_periodic(psi, omega_1, dx, dy, Re_n)
    omega_2 = 0.75 * omega + 0.25 * omega_1 + 0.25 * dt * (f)

    # Fortran update for Poisson Equation
    Spectral_Poisson.solve_poisson(psi,-omega_2, dx, dy)

    # Step 2
    # Calculate RHS
    f = Fortran_Functions.rhs_periodic(psi, omega_2, dx, dy, Re_n)

    omega[:,:] = oneth * omega[:,:] + twoth * omega_2[:,:] + twoth * dt * (f[:,:])

    # Fortran update for Poisson Equation
    Spectral_Poisson.solve_poisson(psi,-omega, dx, dy)

def tvdrk3_fortran_ml_ad_les(omega,psi,ml_model):

    #Need to add ML based sigma estimation - modifies global
    deploy_model_classification(omega,ml_model)

    oneth = 1.0 / 3.0
    twoth = 2.0 / 3.0

    # Step 1
    # Calculate RHS
    f = Fortran_Functions.rhs_periodic(psi, omega, dx, dy, Re_n)
    sfs = ad_sfs_calc(omega,psi)
    omega_1 = omega + dt * (f + sfs)

    # Fortran update for Poisson Equation
    # Multigrid_Solver.solve_poisson_periodic(psi, -omega_1, dx, dy, gs_tol)
    Spectral_Poisson.solve_poisson(psi,-omega_1, dx, dy)

    # Step 2
    # Calculate RHS
    f = Fortran_Functions.rhs_periodic(psi, omega_1, dx, dy, Re_n)
    sfs = ad_sfs_calc(omega_1,psi)
    omega_2 = 0.75 * omega + 0.25 * omega_1 + 0.25 * dt * (f + sfs)

    # Fortran update for Poisson Equation
    Spectral_Poisson.solve_poisson(psi,-omega_2, dx, dy)

    # Step 2
    # Calculate RHS
    f = Fortran_Functions.rhs_periodic(psi, omega_2, dx, dy, Re_n)
    sfs = ad_sfs_calc(omega_2,psi)

    omega[:, :] = oneth * omega[:, :] + twoth * omega_2[:, :] + twoth * dt * (f[:, :] + sfs[:, :])

    # Fortran update for Poisson Equation
    Spectral_Poisson.solve_poisson(psi,-omega, dx, dy)

def tvdrk3_fortran_dyn_smag_sgs(omega,psi):

    oneth = 1.0 / 3.0
    twoth = 2.0 / 3.0

    # Step 1
    # Calculate RHS
    f = Fortran_Functions.rhs_periodic(psi, omega, dx, dy, Re_n)
    #Dynamic Smagorinsky SGS
    sgs = dynamic_smag_calc(omega,psi)

    omega_1 = omega + dt * (f + sgs)

    # Fortran update for Poisson Equation
    #Multigrid_Solver.solve_poisson_periodic(psi, -omega_1, dx, dy, gs_tol)
    Spectral_Poisson.solve_poisson(psi,-omega_1, dx, dy)

    # Step 2
    # Calculate RHS
    f = Fortran_Functions.rhs_periodic(psi, omega_1, dx, dy, Re_n)
    # Dynamic Smagorinsky SGS
    sgs = dynamic_smag_calc(omega_1,psi)
    omega_2 = 0.75 * omega + 0.25 * omega_1 + 0.25 * dt * (f + sgs)

    # Fortran update for Poisson Equation
    Spectral_Poisson.solve_poisson(psi,-omega_2, dx, dy)

    # Step 2
    # Calculate RHS
    f = Fortran_Functions.rhs_periodic(psi, omega_2, dx, dy, Re_n)
    # Dynamic Smagorinsky SGS
    sgs = dynamic_smag_calc(omega_2,psi)

    omega[:, :] = oneth * omega[:, :] + twoth * omega_2[:, :] + twoth * dt * (f[:, :] + sgs[:, :])

    # Fortran update for Poisson Equation
    Spectral_Poisson.solve_poisson(psi,-omega, dx, dy)


def tvdrk3_fortran_leith_sgs(omega,psi):

    oneth = 1.0 / 3.0
    twoth = 2.0 / 3.0

    # Step 1
    # Calculate RHS
    f = Fortran_Functions.rhs_periodic(psi, omega, dx, dy, Re_n)
    # Standard Leith
    sgs = leith_sgs_calc(omega,psi)

    omega_1 = omega + dt * (f + sgs)

    # Fortran update for Poisson Equation
    #Multigrid_Solver.solve_poisson_periodic(psi, -omega_1, dx, dy, gs_tol)
    Spectral_Poisson.solve_poisson(psi,-omega_1, dx, dy)

    # Step 2
    # Calculate RHS
    f = Fortran_Functions.rhs_periodic(psi, omega_1, dx, dy, Re_n)
    # Standard Leith
    sgs = leith_sgs_calc(omega_1,psi)
    omega_2 = 0.75 * omega + 0.25 * omega_1 + 0.25 * dt * (f + sgs)

    # Fortran update for Poisson Equation
    Spectral_Poisson.solve_poisson(psi,-omega_2, dx, dy)

    # Step 2
    # Calculate RHS
    f = Fortran_Functions.rhs_periodic(psi, omega_2, dx, dy, Re_n)
    # Standard Leith
    sgs = leith_sgs_calc(omega_2,psi)

    omega[:, :] = oneth * omega[:, :] + twoth * omega_2[:, :] + twoth * dt * (f[:, :] + sgs[:, :])

    # Fortran update for Poisson Equation
    Spectral_Poisson.solve_poisson(psi,-omega, dx, dy)

def tvdrk3_fortran_dyn_leith_sgs(omega,psi):

    oneth = 1.0 / 3.0
    twoth = 2.0 / 3.0

    # Step 1
    # Calculate RHS
    f = Fortran_Functions.rhs_periodic(psi, omega, dx, dy, Re_n)
    #Dynamic Leith SGS
    sgs = dynamic_leith_calc(omega,psi)

    omega_1 = omega + dt * (f + sgs)

    # Fortran update for Poisson Equation
    Spectral_Poisson.solve_poisson(psi,-omega_1, dx, dy)

    # Step 2
    # Calculate RHS
    f = Fortran_Functions.rhs_periodic(psi, omega_1, dx, dy, Re_n)
    # Dynamic Leith SGS
    sgs = dynamic_leith_calc(omega_1,psi)
    omega_2 = 0.75 * omega + 0.25 * omega_1 + 0.25 * dt * (f + sgs)

    # Fortran update for Poisson Equation
    Spectral_Poisson.solve_poisson(psi,-omega_2, dx, dy)

    # Step 2
    # Calculate RHS
    f = Fortran_Functions.rhs_periodic(psi, omega_2, dx, dy, Re_n)
    # Dynamic Leith SGS
    sgs = dynamic_leith_calc(omega_2,psi)

    omega[:, :] = oneth * omega[:, :] + twoth * omega_2[:, :] + twoth * dt * (f[:, :] + sgs[:, :])

    # Fortran update for Poisson Equation
    Spectral_Poisson.solve_poisson(psi,-omega, dx, dy)

def tvdrk3_fortran(omega,psi):

    oneth = 1.0/3.0
    twoth = 2.0/3.0

    #Step 1
    #Calculate RHS
    f = Fortran_Functions.rhs_periodic(psi,omega, dx, dy, Re_n)
    omega_1 = omega + dt*(f)

    #Fortran update for Poisson Equation
    #Multigrid_Solver.solve_poisson_periodic(psi,-omega_1, dx, dy, gs_tol)
    Spectral_Poisson.solve_poisson(psi,-omega_1, dx, dy)

    #Step 2
    #Calculate RHS
    f = Fortran_Functions.rhs_periodic(psi,omega_1, dx, dy, Re_n)
    omega_2 = 0.75*omega + 0.25*omega_1 + 0.25*dt*(f)

    #Fortran update for Poisson Equation
    Spectral_Poisson.solve_poisson(psi,-omega_2, dx, dy)

    #Step 2
    #Calculate RHS
    f = Fortran_Functions.rhs_periodic(psi,omega_2, dx, dy, Re_n)

    omega[:, :] = oneth * omega[:, :] + twoth * omega_2[:, :] + twoth * dt * (f[:, :])

    #Fortran update for Poisson Equation
    Spectral_Poisson.solve_poisson(psi,-omega, dx, dy)

def tvdrk3_fortran_iles(omega,psi):

    oneth = 1.0/3.0
    twoth = 2.0/3.0

    #Step 1
    #Calculate RHS
    f = Fortran_Functions.rhs_periodic_iles(psi,omega, dx, dy, Re_n)
    omega_1 = omega + dt*(f)

    #Fortran update for Poisson Equation
    #Multigrid_Solver.solve_poisson_periodic(psi,-omega_1, dx, dy, gs_tol)
    Spectral_Poisson.solve_poisson(psi,-omega_1, dx, dy)

    #Step 2
    #Calculate RHS
    f = Fortran_Functions.rhs_periodic_iles(psi,omega_1, dx, dy, Re_n)
    omega_2 = 0.75*omega + 0.25*omega_1 + 0.25*dt*(f)

    #Fortran update for Poisson Equation
    Spectral_Poisson.solve_poisson(psi,-omega_2, dx, dy)

    #Step 2
    #Calculate RHS
    f = Fortran_Functions.rhs_periodic_iles(psi,omega_2, dx, dy, Re_n)

    omega[:, :] = oneth * omega[:, :] + twoth * omega_2[:, :] + twoth * dt * (f[:, :])

    #Fortran update for Poisson Equation
    Spectral_Poisson.solve_poisson(psi,-omega, dx, dy)

def tvdrk3_fortran_fou(omega,psi):

    oneth = 1.0/3.0
    twoth = 2.0/3.0

    #Step 1
    #Calculate RHS
    f = Fortran_Functions.rhs_periodic_fou(psi,omega, dx, dy, Re_n)
    omega_1 = omega + dt*(f)

    #Fortran update for Poisson Equation
    #Multigrid_Solver.solve_poisson_periodic(psi,-omega_1, dx, dy, gs_tol)
    Spectral_Poisson.solve_poisson(psi,-omega_1, dx, dy)

    #Step 2
    #Calculate RHS
    f = Fortran_Functions.rhs_periodic_fou(psi,omega_1, dx, dy, Re_n)
    omega_2 = 0.75*omega + 0.25*omega_1 + 0.25*dt*(f)

    #Fortran update for Poisson Equation
    Spectral_Poisson.solve_poisson(psi,-omega_2, dx, dy)

    #Step 2
    #Calculate RHS
    f = Fortran_Functions.rhs_periodic_fou(psi,omega_2, dx, dy, Re_n)

    omega[:, :] = oneth * omega[:, :] + twoth * omega_2[:, :] + twoth * dt * (f[:, :])

    #Fortran update for Poisson Equation
    Spectral_Poisson.solve_poisson(psi,-omega, dx, dy)

def tvdrk3_fortran_ad_les(omega,psi):

    oneth = 1.0/3.0
    twoth = 2.0/3.0

    #Step 1
    #Calculate RHS
    f = Fortran_Functions.rhs_periodic(psi,omega, dx, dy, Re_n)
    sfs = ad_sfs_calc(omega,psi)
    omega_1 = omega + dt*(f+sfs)

    #Fortran update for Poisson Equation
    # Multigrid_Solver.solve_poisson_periodic(psi,-omega_1, dx, dy, gs_tol)
    Spectral_Poisson.solve_poisson(psi,-omega_1, dx, dy)

    #Step 2
    #Calculate RHS
    f = Fortran_Functions.rhs_periodic(psi,omega_1, dx, dy, Re_n)
    sfs = ad_sfs_calc(omega_1, psi)
    omega_2 = 0.75*omega + 0.25*omega_1 + 0.25*dt*(f+sfs)

    #Fortran update for Poisson Equation
    Spectral_Poisson.solve_poisson(psi,-omega_2, dx, dy)

    #Step 2
    #Calculate RHS
    f = Fortran_Functions.rhs_periodic(psi,omega_2, dx, dy, Re_n)
    sfs = ad_sfs_calc(omega_2, psi)

    omega[:, :] = oneth * omega[:, :] + twoth * omega_2[:, :] + twoth * dt * (f[:, :] + sfs[:, :])

    #Fortran update for Poisson Equation
    Spectral_Poisson.solve_poisson(psi,-omega, dx, dy)

def tvdrk3_fortran_bd_les(omega,psi):

    oneth = 1.0/3.0
    twoth = 2.0/3.0

    #Step 1
    #Calculate RHS
    f = Fortran_Functions.rhs_periodic(psi,omega, dx, dy, Re_n)
    sfs = bd_sfs_calc(omega,psi)
    omega_1 = omega + dt*(f+sfs)

    #Fortran update for Poisson Equation
    # Multigrid_Solver.solve_poisson_periodic(psi,-omega_1, dx, dy, gs_tol)
    Spectral_Poisson.solve_poisson(psi,-omega_1, dx, dy)

    #Step 2
    #Calculate RHS
    f = Fortran_Functions.rhs_periodic(psi,omega_1, dx, dy, Re_n)
    sfs = bd_sfs_calc(omega_1, psi)
    omega_2 = 0.75*omega + 0.25*omega_1 + 0.25*dt*(f+sfs)

    #Fortran update for Poisson Equation
    Spectral_Poisson.solve_poisson(psi,-omega_2, dx, dy)

    #Step 2
    #Calculate RHS
    f = Fortran_Functions.rhs_periodic(psi,omega_2, dx, dy, Re_n)
    sfs = bd_sfs_calc(omega_2, psi)

    omega[:, :] = oneth * omega[:, :] + twoth * omega_2[:, :] + twoth * dt * (f[:, :] + sfs[:, :])

    #Fortran update for Poisson Equation
    Spectral_Poisson.solve_poisson(psi,-omega, dx, dy)


def tvdrk3_fortran_ml_logistic_blended(omega,psi,ml_model):

    oneth = 1.0 / 3.0
    twoth = 2.0 / 3.0

    # Step 1
    # Calculate RHS
    f = Fortran_Functions.rhs_periodic(psi, omega, dx, dy, Re_n)
    #Need to add ML based sgs computation
    sgs = deploy_model_logistic_blended(omega,psi,ml_model)

    omega_1 = omega + dt * (f + sgs)

    # Fortran update for Poisson Equation
    #Multigrid_Solver.solve_poisson_periodic(psi, -omega_1, dx, dy, gs_tol)
    Spectral_Poisson.solve_poisson(psi,-omega_1, dx, dy)

    # Step 2
    # Calculate RHS
    f = Fortran_Functions.rhs_periodic(psi, omega_1, dx, dy, Re_n)
    #Need to add ML based sgs computation
    sgs = deploy_model_logistic_blended(omega_1,psi,ml_model)
    omega_2 = 0.75 * omega + 0.25 * omega_1 + 0.25 * dt * (f + sgs)

    # Fortran update for Poisson Equation
    Spectral_Poisson.solve_poisson(psi,-omega_2, dx, dy)

    # Step 2
    # Calculate RHS
    f = Fortran_Functions.rhs_periodic(psi, omega_2, dx, dy, Re_n)
    #Need to add ML based sgs computation
    sgs = deploy_model_logistic_blended(omega_2,psi,ml_model)

    omega[:, :] = oneth * omega[:, :] + twoth * omega_2[:, :] + twoth * dt * (f[:, :] + sgs[:, :])

    # Fortran update for Poisson Equation
    Spectral_Poisson.solve_poisson(psi,-omega, dx, dy)

def tvdrk3_fortran_ml_logistic_blended_five_class(omega,psi,ml_model):

    oneth = 1.0 / 3.0
    twoth = 2.0 / 3.0

    # Step 1
    # Calculate RHS
    f = Fortran_Functions.rhs_periodic(psi, omega, dx, dy, Re_n)
    #Need to add ML based sgs computation
    sgs = deploy_model_logistic_blended_five_class(omega,psi,ml_model)

    omega_1 = omega + dt * (f + sgs)

    # Fortran update for Poisson Equation
    #Multigrid_Solver.solve_poisson_periodic(psi, -omega_1, dx, dy, gs_tol)
    Spectral_Poisson.solve_poisson(psi,-omega_1, dx, dy)

    # Step 2
    # Calculate RHS
    f = Fortran_Functions.rhs_periodic(psi, omega_1, dx, dy, Re_n)
    #Need to add ML based sgs computation
    sgs = deploy_model_logistic_blended_five_class(omega_1,psi,ml_model)
    omega_2 = 0.75 * omega + 0.25 * omega_1 + 0.25 * dt * (f + sgs)

    # Fortran update for Poisson Equation
    Spectral_Poisson.solve_poisson(psi,-omega_2, dx, dy)

    # Step 2
    # Calculate RHS
    f = Fortran_Functions.rhs_periodic(psi, omega_2, dx, dy, Re_n)
    #Need to add ML based sgs computation
    sgs = deploy_model_logistic_blended_five_class(omega_2,psi,ml_model)

    omega[:, :] = oneth * omega[:, :] + twoth * omega_2[:, :] + twoth * dt * (f[:, :] + sgs[:, :])

    # Fortran update for Poisson Equation
    Spectral_Poisson.solve_poisson(psi,-omega, dx, dy)

def tvdrk3_fortran_ML_TBDNN(omega,psi,ml_model):

    oneth = 1.0 / 3.0
    twoth = 2.0 / 3.0

    # Step 1
    # Calculate RHS
    f = Fortran_Functions.rhs_periodic(psi, omega, dx, dy, Re_n)
    #Need to add ML based sgs computation
    sgs = deploy_model_tbdnn(omega,psi,ml_model)

    omega_1 = omega + dt * (f + sgs)

    # Fortran update for Poisson Equation
    #Multigrid_Solver.solve_poisson_periodic(psi, -omega_1, dx, dy, gs_tol)
    Spectral_Poisson.solve_poisson(psi,-omega_1, dx, dy)

    # Step 2
    # Calculate RHS
    f = Fortran_Functions.rhs_periodic(psi, omega_1, dx, dy, Re_n)
    #Need to add ML based sgs computation
    sgs = deploy_model_tbdnn(omega_1,psi,ml_model)
    omega_2 = 0.75 * omega + 0.25 * omega_1 + 0.25 * dt * (f + sgs)

    # Fortran update for Poisson Equation
    Spectral_Poisson.solve_poisson(psi,-omega_2, dx, dy)

    # Step 2
    # Calculate RHS
    f = Fortran_Functions.rhs_periodic(psi, omega_2, dx, dy, Re_n)
    #Need to add ML based sgs computation
    sgs = deploy_model_tbdnn(omega_2,psi,ml_model)

    omega[:, :] = oneth * omega[:, :] + twoth * omega_2[:, :] + twoth * dt * (f[:, :] + sgs[:, :])

    # Fortran update for Poisson Equation
    Spectral_Poisson.solve_poisson(psi,-omega, dx, dy)

def tvdrk3_fortran_ml_par_log(omega,psi,ml_model):

    oneth = 1.0 / 3.0
    twoth = 2.0 / 3.0

    # Step 1
    # Calculate RHS
    f = Fortran_Functions.rhs_periodic(psi, omega, dx, dy, Re_n)
    #Need to add ML based sgs computation
    sgs = deploy_model_par_log(omega,psi,ml_model)

    omega_1 = omega + dt * (f + sgs)

    # Fortran update for Poisson Equation
    #Multigrid_Solver.solve_poisson_periodic(psi, -omega_1, dx, dy, gs_tol)
    Spectral_Poisson.solve_poisson(psi,-omega_1, dx, dy)

    # Step 2
    # Calculate RHS
    f = Fortran_Functions.rhs_periodic(psi, omega_1, dx, dy, Re_n)
    #Need to add ML based sgs computation
    sgs = deploy_model_par_log(omega_1,psi,ml_model)
    omega_2 = 0.75 * omega + 0.25 * omega_1 + 0.25 * dt * (f + sgs)

    # Fortran update for Poisson Equation
    Spectral_Poisson.solve_poisson(psi,-omega_2, dx, dy)

    # Step 2
    # Calculate RHS
    f = Fortran_Functions.rhs_periodic(psi, omega_2, dx, dy, Re_n)
    #Need to add ML based sgs computation
    sgs = deploy_model_par_log(omega_2,psi,ml_model)

    omega[:, :] = oneth * omega[:, :] + twoth * omega_2[:, :] + twoth * dt * (f[:, :] + sgs[:, :])

    # Fortran update for Poisson Equation
    Spectral_Poisson.solve_poisson(psi,-omega, dx, dy)

#-------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------#
# Temporal quantity trackers
#-------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------#
def tke_tracker(psi):
    return Temporal_Tracker.tke_tracker(psi,dx,dy)

def ens_tracker(omega):
    return Temporal_Tracker.enstrophy_tracker(omega)

def var_tracker(omega):
    return Temporal_Tracker.vort_var_tracker(omega)

def classification_tracker(return_matrix):
    global class_tracker # Since we are modifying this
    Temporal_Tracker.classification_tracker(return_matrix, class_tracker)

#-------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------#
#Main time integrator
#-------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------#

def main_func():

    #Initialize my domain and constants
    omega, psi = init_domain()

    initialize_ic_bc(omega,psi)

    t = 0.0

    #clock_time_init = time.clock()
    clock_time_start = time.time()
    # Defining numpy array for storing temporal info
    sigma_history = np.asarray([sigma], dtype='double')
    tke_counter = np.zeros(shape=(nt,3),dtype='double')
    class_counter = np.zeros(shape=(nt,3), dtype='int')

    for tstep in range(nt):

        t = t + dt
        print(t)

        #TVD - RK3 Fortran
        if closure_choice == 5:#Smag SGS
            tvdrk3_fortran_smag_sgs(omega,psi)
        elif closure_choice == 10:#Dynamic Smagorinsky
            tvdrk3_fortran_dyn_smag_sgs(omega, psi)

        if np.isnan(np.sum(omega))==1:
            print('overflow')
            exit()

        # Measurement of temporal quantities
        tke_counter[tstep, 0] = tke_tracker(psi)
        tke_counter[tstep, 1] = ens_tracker(omega)
        tke_counter[tstep, 2] = var_tracker(omega)


    total_clock_time = time.time() - clock_time_start

    print('Total Clock Time = ',total_clock_time)
    post_process(omega,psi)

    np.savetxt('Time_Evolution.txt',tke_counter)
    if class_use:
        np.savetxt('Classification_history.txt',class_counter)

#-------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------#
##### RUN HERE #####
#-------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------#
main_func()
