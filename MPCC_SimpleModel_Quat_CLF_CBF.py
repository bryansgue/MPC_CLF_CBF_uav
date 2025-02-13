from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from acados_template import AcadosModel
import scipy.linalg
import numpy as np
import time
import matplotlib.pyplot as plt
from casadi import Function
from casadi import MX
from casadi import reshape
from casadi import vertcat
from casadi import horzcat
from casadi import cos
from casadi import sin
from casadi import solve
from casadi import inv
from casadi import mtimes
from casadi import sqrt
from casadi import jacobian
from scipy.integrate import quad
from scipy.optimize import bisect
from casadi import sqrt, dot, fabs, sign
from scipy.interpolate import CubicSpline
import rospy
from scipy.spatial.transform import Rotation as R
from nav_msgs.msg import Odometry
#from c_generated_code.acados_ocp_solver_pyx import AcadosOcpSolverCython
from geometry_msgs.msg import TwistStamped
import math
import os
from scipy.io import savemat

# CARGA FUNCIONES DEL PROGRAMA
from fancy_plots import plot_pose, plot_error, plot_time

from Functions_SimpleModel import calc_M, calc_C, calc_G, QuatToRot, QuatToRot_nunpy, log_cuaternion_casadi
from Functions_SimpleModel import f_d, odometry_call_back_1, odometry_call_back_2, get_odometry_simple_quat_1, get_odometry_simple_quat_2, send_velocity_control, pub_odometry_sim_quat, euler_to_quaternion, quaternion_error, publish_matrix 
#import P_UAV_simple

# Global variables Odometry Drone
x_real_1 = 1
y_real_1 = 1
z_real_1 = 5
vx_real_1 = 0.0
vy_real_1 = 0.0
vz_real_1 = 0.0
qw_real_1 = 1
qx_real_1 = 0
qy_real_1 = 0.0
qz_real_1 = 0
wx_real_1 = 0.0
wy_real_1 = 0.0
wz_real_1 = 0.0

# Definir el valor global
value = 10
valueB = 7 # Buencomportameinto con 5

uav_r = 0.15
margen = 0.1
obsmovil_r = 0.4


def f_system_simple_model_quat():
    # Name of the system
    model_name = 'Drone_ode'
    # Dynamic Values of the system

    chi = [0.6756,    1.0000,    0.6344,    1.0000,    0.4080,    1.0000,    1.0000,    1.0000,    0.2953,    0.5941,   -0.8109,    1.0000,    0.3984,    0.7040,    1.0000,    0.9365,    1.0000, 1.0000,    0.9752]# Position
    
    # set up states & controls
    # Position
    nx = MX.sym('nx') 
    ny = MX.sym('ny')
    nz = MX.sym('nz')
    qw = MX.sym('qw')
    qx = MX.sym('qx')
    qy = MX.sym('qy')
    qz = MX.sym('qz')
    ul = MX.sym('ul')
    um = MX.sym('um')
    un = MX.sym('un')
    w = MX.sym('w')

    # General vector of the states
    x = vertcat(nx, ny, nz, qw, qx, qy, qz, ul, um, un, w)

    # Action variables
    ul_ref = MX.sym('ul_ref')
    um_ref = MX.sym('um_ref')
    un_ref = MX.sym('un_ref')
    w_ref = MX.sym('w_ref')

    # General Vector Action variables
    u = vertcat(ul_ref,um_ref,un_ref,w_ref)

    # Variables to explicit function
    nx_p = MX.sym('nx_p')
    ny_p = MX.sym('ny_p')
    nz_p = MX.sym('nz_p')
    qw_p = MX.sym('qw_p')
    qx_p = MX.sym('qx_p')
    qy_p = MX.sym('qy_p')
    qz_p = MX.sym('qz_p')
    ul_p = MX.sym('ul_p')
    um_p = MX.sym('um_p')
    un_p = MX.sym('un_p')
    w_p = MX.sym('w_p')

    # general vector X dot for implicit function
    xdot = vertcat(nx_p,ny_p,nz_p,qw_p,qx_p,qy_p,qz_p,ul_p,um_p,un_p,w_p)

    # Ref system as a external value
    nx_d = MX.sym('nx_d')
    ny_d = MX.sym('ny_d')
    nz_d = MX.sym('nz_d')
    qw_d = MX.sym('qw_d')
    qx_d = MX.sym('qx_d')
    qy_d = MX.sym('qy_d')
    qz_d = MX.sym('qz_d')
    ul_d = MX.sym('ul_d')
    um_d= MX.sym('um_d')
    un_d = MX.sym('un_d')
    w_d = MX.sym('w_d')

    ul_ref_d= MX.sym('ul_ref_d')
    um_ref_d= MX.sym('um_ref_d')
    un_ref_d = MX.sym('un_ref_d')
    w_ref_d = MX.sym('w_ref_d')

    obs1_x = MX.sym('obs1_x')
    obs1_y = MX.sym('obs1_y')
    obs1_z = MX.sym('obs1_z')
    obs1_r = MX.sym('obs1_r')
    obs2_x = MX.sym('obs2_x')
    obs2_y = MX.sym('obs2_y')
    obs2_z = MX.sym('obs2_z')

    dp_dsx = MX.sym('dp_dsx')
    dp_dsy = MX.sym('dp_dsy')
    dp_dsz = MX.sym('dp_dsz')


    values = vertcat(dp_dsx, dp_dsy, dp_dsz, obs1_x,obs1_y,obs1_z,obs1_r, obs2_x,obs2_y,obs2_z)
    
    p = vertcat(nx_d, ny_d, nz_d, qw_d, qx_d, qy_d, qz_d, ul_d, um_d, un_d, w_d, ul_ref_d, um_ref_d, un_ref_d, w_ref_d, values)

    # Rotational Matrix
    a = 0
    b = 0
    
    M = calc_M(chi,a,b)
    C = calc_C(chi,a,b, w)
    G = calc_G()

    # Crea una lista de MX con los componentes del cuaternión
    quat = [qw, qx, qy, qz]

    # Obtener la matriz de rotación
    J = QuatToRot(quat)

    # Evolucion quat
    p_x = 0
    q = 0
    r = w

    S = vertcat(
        horzcat(0, -p_x, -q, -r),
        horzcat(p_x, 0, r, -q),
        horzcat(q, -r, 0, p_x),
        horzcat(r, q, -p_x, 0)
    )


    # Crear matriz A
    A_1 = horzcat(MX.zeros(3, 7), J, MX.zeros(3, 1))
    A_2 = horzcat(MX.zeros(4, 3), 1/2*S, MX.zeros(4, 4))
    A_3 = horzcat(MX.zeros(4, 7), -mtimes(inv(M), C))
   
    A = vertcat(A_1, A_2, A_3)

    # Crear matriz B
    B_top = MX.zeros(7, 4)
    B_bottom = inv(M)
    B = vertcat(B_top, B_bottom)

    f_expl = MX.zeros(11, 1)
    f_expl = A @ x + B @ u 

    #f_x = A @ x 
    #g_x = B

    # Define f_x and g_x
    f_x = Function('f_x', [x], [f_expl])
    g_x = Function('g_x', [x, u], [jacobian(f_expl, u)])

    f_system = Function('system',[x, u], [f_expl])
     # Acados Model
    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name
    model.p = p

    return model, f_system, f_x, g_x

def create_ocp_solver_description(x0, N_horizon, t_horizon, zp_max, zp_min, phi_max, phi_min, theta_max, theta_min, psi_max, psi_min) -> AcadosOcp:
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    model, f_system, f_x, g_x  = f_system_simple_model_quat()
    ocp.model = model
    ocp.p = model.p
    n_x = model.x.size()[0]
    n_u = model.u.size()[0]
    values_extra = 10
    ny = n_x + n_u + values_extra

    # set dimensions
    ocp.dims.N = N_horizon

    Q_mat = MX.zeros(3, 3)
    Q_mat[0, 0] = 1
    Q_mat[1, 1] = 1
    Q_mat[2, 2] = 1

    R_mat = MX.zeros(4, 4)
    R_mat[0, 0] = 1
    R_mat[1, 1] = 1
    R_mat[2, 2] = 1
    R_mat[3, 3] = 1
    
    ocp.parameter_values = np.zeros(ny)

    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    

    p = vertcat(model.x[0], model.x[1], model.x[2])  # Posiciones p1, p2, p3
    q = vertcat(model.x[3], model.x[4], model.x[5], model.x[6])  # Posiciones p1, p2, p3
    nu = vertcat(model.x[7], model.x[8], model.x[9])  # velocidades tridimencionales en el cuerpo

    error_pose = ocp.p[0:3] - model.x[0:3]
    quat_error = quaternion_error(ocp.p[3:7], q)
    log_q = log_cuaternion_casadi(quat_error)
    
    sd = ocp.p[0:3]
    #Vector tangente
    sd_p  = ocp.p[15:18]  # Si dp_dsx, dp_dsy y dp_dsz están en positions 16, 17 y 18

    e_t = (sd - p)
    tangent_normalized = sd_p # sd_p / norm_2(sd_p) ---> por propiedad la nomra de la recta tangente en longotud de arco ya es unitario
    el = dot(tangent_normalized, e_t) * tangent_normalized

    # ERROR DE CONTORNO
    I = MX.eye(3) 
    P_ec = I - tangent_normalized.T @ tangent_normalized
    ec = P_ec @ e_t 

    # Obtener la matriz de rotación
    q_list = [model.x[3], model.x[4], model.x[5], model.x[6]] 
    J = QuatToRot(q_list)
    v = J@nu
    #Velocidad de avance
    vel_progres = dot(tangent_normalized, v)


    #Ganancias
    # set cost
    Q_q = 1* np.diag([1, 1, 1])  # [x,th,dx,dth]
    Q_el = 1 * np.eye(3)  
    Q_ec = 1* np.eye(3) 
    U_mat = 0.01 * np.diag([ 1,1,1,1])
    Q_vels = 0.1


    #COSTO EXTERNO
    control_cost = 1*model.u.T @ U_mat @ model.u 
    actitud_cost = log_q.T @ Q_q @ log_q 
    error_contorno = 1*ec.T @ Q_ec @ ec
    error_lag = 1*el.T @ Q_el @ el
    vel_progres_cost = Q_vels*vel_progres 


    ocp.model.cost_expr_ext_cost = error_contorno + error_lag + actitud_cost + control_cost - vel_progres_cost
    ocp.model.cost_expr_ext_cost_e = error_contorno + error_lag + actitud_cost - vel_progres_cost


    #RESTRICCIONES NO LINEALES

    # Definir la matriz de ponderación para los errores de arrastre y contorno
    Q_l = MX.eye(3)  # Matriz de ponderación para el error de arrastre
    Q_c = MX.eye(3)  # Matriz de ponderación para el error de contorno

    # Función candidata de Lyapunov basada en el error de arrastre
    V= (1/2) * ec.T @ Q_c @ ec + (1/2) * el.T @ Q_l @ el

    # Derivada temporal de V (Lyapunov function)
    V_p = jacobian(V, p) @ v

    obs_x = ocp.p[18,0]     # Posicion donde actualiza la posicion del obstaculo
    obs_y= ocp.p[19,0]
    obs_z= ocp.p[20,0]
    obs_r =ocp.p[21,0]


    obs1 = ocp.p[18:22] 

    margen = 0.1

    obsmovil_x = ocp.p[22,0]     # Posicion donde actualiza la posicion del obstaculo
    obsmovil_y= ocp.p[23,0]
    obsmovil_z= ocp.p[24,0]

    obs2 = ocp.p[22:25]

    # PRIMERA BARRERA (en el espacio 3D)
    h = sqrt((model.x[0] - obs1[0])**2 + (model.x[1] -  obs1[1])**2 + (model.x[2] -  obs1[2])**2) - (uav_r + obs1[3] + margen)
    h_movil = sqrt((model.x[0] - obs2[0])**2 + (model.x[1] - obs2[1])**2 + (model.x[2] - obs2[2])**2) - (uav_r + obsmovil_r  + margen)

    f_x_val = f_x(model.x)
    g_x_val = g_x(model.x,model.u)

    # Derivada de Lie de primer orden
    Lf_h = jacobian(h, model.x) @ f_x_val 
    Lg_h = jacobian(h, model.x) @ g_x_val

    Lf_h_movil = jacobian(h_movil, model.x) @ f_x_val 
    Lg_h_movil = jacobian(h_movil, model.x) @ g_x_val

    # Derivada de Lie de segundo orden
    Lf2_h = jacobian(Lf_h, model.x) @ f_x_val
    Lg_L_f_h = jacobian(Lf_h, model.x) @ g_x_val 
    
    Lf2_h_movil = jacobian(Lf_h_movil, model.x) @ f_x_val
    Lg_L_f_h_movil = jacobian(Lf_h_movil, model.x) @ g_x_val 

    # Barreras temporales
    h_p = Lf_h + Lg_h @ model.u
    h_pp = Lf2_h + Lg_L_f_h @ model.u

    h_p_movil = Lf_h_movil + Lg_h_movil @ model.u
    h_pp_movil = Lf2_h_movil + Lg_L_f_h_movil @ model.u

    # set constraints
    ocp.constraints.constr_type = 'BGH'

    # Funciones de barrera de segundo orden
    nb_1 = h
    nb_2 = vertcat(h, Lf_h) 

    nb_1_movil = h_movil
    nb_2_movil = vertcat(h_movil, Lf_h_movil) 

    K_alpha = MX([40, 12]).T ## 20 8
    K_alpha_movil = MX([20, 15]).T ## 20 8

    #constraints = vertcat(h_p + 5*nb_1)
    CBF_static = h_pp +  K_alpha @ nb_2
    CBF_movil = h_pp_movil +  K_alpha_movil @ nb_2_movil

    constraints = vertcat(CBF_static, CBF_movil)
    #constraints = vertcat(model.x[0] )

    # Asigna las restricciones al modelo del OCP
    N_constraints = constraints.size1()

    ocp.model.con_h_expr = constraints
    ocp.constraints.lh = np.array([0,0])  # Límite inferior 
    ocp.constraints.uh = np.array([1e9,1e9])  # Límite superior

    # Configuración de las restricciones suaves
    cost_weights =  np.array([0.1,0.1])
    ocp.cost.zu = 1*cost_weights 
    ocp.cost.zl = 1*cost_weights 
    ocp.cost.Zl = 1 * cost_weights 
    ocp.cost.Zu = 1 * cost_weights 

    # Índices para las restricciones suaves (necesario si se usan)
    ocp.constraints.idxsh = np.arange(N_constraints)  # Índices de las restricciones suaves

    ocp.constraints.x0 = x0

    # set options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"  # 'GAUSS_NEWTON', 'EXACT'
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"  # SQP_RTI, SQP
    #ocp.solver_options.tol = 1e-4

    # set prediction horizon
    ocp.solver_options.tf = t_horizon

    return ocp

def trayectoria(t):

    def xd(t):
        return 7 * np.sin(value * 0.04 * t) + 3

    def yd(t):
        return 7 * np.sin(value * 0.08 * t)

    def zd(t):
        return 1.5 * np.sin(value * 0.08 * t) + 6

    def xd_p(t):
        return 7 * value * 0.04 * np.cos(value * 0.04 * t)

    def yd_p(t):
        return 7 * value * 0.08 * np.cos(value * 0.08 * t)

    def zd_p(t):
        return 1.5 * value * 0.08 * np.cos(value * 0.08 * t)

    return xd, yd, zd, xd_p, yd_p, zd_p

def trayectoriaB(t):
    def xd(t):
        return 7 * np.sin(-valueB * 0.04 * t) + 3

    def yd(t):
        return 7 * np.sin(-valueB * 0.08 * t)

    def zd(t):
        return 1.5 * np.sin(-valueB * 0.08 * t) + 6

    def xd_p(t):
        return -7 * valueB * 0.04 * np.cos(-valueB * 0.04 * t)

    def yd_p(t):
        return -7 * valueB * 0.08 * np.cos(-valueB * 0.08 * t)

    def zd_p(t):
        return -1.5 * valueB * 0.08 * np.cos(-valueB * 0.08 * t)

    return xd, yd, zd, xd_p, yd_p, zd_p

def calculate_errors_norm(sd, sd_p, model_x):
    """
    Calcula las normas del error de contorno, error de arrastre, error total y la velocidad de avance.
    
    Parámetros:
    sd       -- (np.array) posición deseada (vector de 3 elementos)
    sd_p     -- (np.array) velocidad tangente deseada (vector de 3 elementos, ya normalizado)
    model_x  -- (np.array) estado actual del UAV (6 elementos: 0-2 posición, 3-5 velocidad)

    Devuelve:
    norm_error_contorno, norm_error_arrastre, error_total, vel_progres
    """

    # ERROR DE POSICIÓN
    e_t = sd - model_x[0:3]

    # ERROR DE ARRASTRE (norma del vector)
    tangent_normalized = sd_p  # ya está normalizado por longitud de arco
    el = np.dot(tangent_normalized, e_t) * tangent_normalized
    norm_error_arrastre = np.linalg.norm(el)

    # ERROR DE CONTORNO (norma del vector)
    I = np.eye(3)
    P_ec = I - np.outer(tangent_normalized, tangent_normalized)  # proyección ortogonal
    error_contorno = np.dot(P_ec, e_t)
    norm_error_contorno = np.linalg.norm(error_contorno)

    # ERROR TOTAL (suma de las normas)
    error_total = error_contorno + el

    # VELOCIDAD DE AVANCE (escalar)

    nu = vertcat(model_x[7], model_x[8], model_x[9])  # velocidades tridimencionales en el cuerpo
    q_list = [model_x[3], model_x[4], model_x[5], model_x[6]] 
    J = QuatToRot_nunpy(q_list)
    v = J@nu

    vel_progres = np.dot(tangent_normalized, v)

    return error_contorno, el, error_total, vel_progres

def calculate_positions_and_arc_length(xd, yd, zd, xd_p, yd_p, zd_p, t_range, t_max):

    
    def r(t):
        """ Devuelve el punto en la trayectoria para el parámetro t usando las funciones de trayectoria. """
        return np.array([xd(t), yd(t), zd(t)])

    def r_prime(t):
        """ Devuelve la derivada de la trayectoria en el parámetro t usando las derivadas de las funciones de trayectoria. """
        return np.array([xd_p(t), yd_p(t), zd_p(t)])

    def integrand(t):
        """ Devuelve la norma de la derivada de la trayectoria en el parámetro t. """
        return np.linalg.norm(r_prime(t))

    def arc_length(tk, t0=0):
        """ Calcula la longitud de arco desde t0 hasta tk usando las derivadas de la trayectoria. """
        length1, _ = quad(integrand, t0, (t0 + tk) / 2, limit=50)
        length2, _ = quad(integrand, (t0 + tk) / 2, tk, limit=50)
        length = length1 + length2
        length, _ = quad(integrand, t0, tk, limit=100)
        return length

    def find_t_for_length(theta, t0=0):
        """ Encuentra el parámetro t que corresponde a una longitud de arco theta. """
        func = lambda t: arc_length(t, t0) - theta
        return bisect(func, t0, t_max)

    # Generar las posiciones y longitudes de arco
    positions = []
    arc_lengths = []
    
    for tk in t_range:
        theta = arc_length(tk)
        arc_lengths.append(theta)
        point = r(tk)
        positions.append(point)

    arc_lengths = np.array(arc_lengths)
    positions = np.array(positions).T  # Convertir a array 2D (3, N)

    # Crear splines cúbicos para la longitud de arco con respecto al tiempo
    spline_t = CubicSpline(arc_lengths, t_range)
    spline_x = CubicSpline(t_range, positions[0, :])
    spline_y = CubicSpline(t_range, positions[1, :])
    spline_z = CubicSpline(t_range, positions[2, :])

    # Función que retorna la posición dado un valor de longitud de arco
    def position_by_arc_length(s):
        t_estimated = spline_t(s)  # Usar spline para obtener la estimación precisa de t
        return np.array([spline_x(t_estimated), spline_y(t_estimated), spline_z(t_estimated)])

    return arc_lengths, positions, position_by_arc_length

def main(vel_pub, vel_msg, odom_sim_pub_1, odom_sim_msg_1):
    # Initial Values System
    # Simulation Time
    t_final = 30
    # Sample time
    frec= 30
    t_s = 1/frec
    # Prediction Time
    N_horizont = 30
    t_prediction = N_horizont/frec

    # Nodes inside MPC
    N = np.arange(0, t_prediction + t_s, t_s)
    N_prediction = N.shape[0]

    # Time simulation
    t = np.arange(0, t_final + t_s, t_s)

    # Sample time vector
    delta_t = np.zeros((1, t.shape[0] - N_prediction), dtype=np.double)
    t_sample = t_s*np.ones((1, t.shape[0] - N_prediction), dtype=np.double)

    h_CBF_1 = np.zeros((1, t.shape[0] - N_prediction), dtype=np.double)
    h_CBF_2 = np.zeros((1, t.shape[0] - N_prediction), dtype=np.double)
    CLF = np.zeros((1, t.shape[0] - N_prediction), dtype=np.double)
    e_contorno = np.zeros((3, t.shape[0] - N_prediction), dtype=np.double)
    e_arrastre = np.zeros((3, t.shape[0] - N_prediction), dtype=np.double)
    e_total = np.zeros((3, t.shape[0] - N_prediction), dtype=np.double)
    vel_progres = np.zeros((1, t.shape[0] - N_prediction), dtype=np.double)
    vel_progress_ref =  np.zeros((1, t.shape[0] - N_prediction), dtype=np.double)

    # Vector Initial conditions
    x = np.zeros((11, t.shape[0]+1-N_prediction), dtype = np.double)

    # Read Real data
    x[:, 0] = get_odometry_simple_quat_1()
    x[:, 0] = [1,1,5,1,0,0,0.5,0,0,0,0]
   
    # Obtener las funciones de trayectoria y sus derivadas
    xd, yd, zd, xd_p, yd_p, zd_p = trayectoria(t)
    #xd, yd, zd, xd_p, yd_p, zd_p = trayectoria_hiper(t)
    xd_obs1, yd_obs1, zd_obs1, xd_p_obs1, yd_p_obs1, zd_p_obs1 = trayectoriaB(t)

    # Calcular posiciones parametrizadas en longitud de arco
    t_finer = np.linspace(0, t_final, len(t))  # Duplicar el tiempo y generar más puntos

    arc_lengths, pos_ref, position_by_arc_length= calculate_positions_and_arc_length(xd, yd, zd, xd_p, yd_p, zd_p, t_finer , t_max=t_final)
    dp_ds = np.gradient(pos_ref, arc_lengths, axis=1)
    
    vmax = 5
    alpha= 0.2
    #pos_ref, s_progress, v_ref, dp_ds = calculate_reference_positions_and_curvature(arc_lengths, position_by_arc_length, t, t_s, vmax  , alpha)


    # Evaluar las derivadas en cada instante
    xd_p_vals = xd_p(t)
    yd_p_vals = yd_p(t)

    # Calcular psid y su derivada
    psid = np.arctan2(yd_p_vals, xd_p_vals)

    #quaternion = euler_to_quaternion(0, 0, psid) 
    quatd= np.zeros((4, t.shape[0]), dtype = np.double)


    # Calcular los cuaterniones utilizando la función euler_to_quaternion para cada psid
    for i in range(t.shape[0]):
        quaternion = euler_to_quaternion(0, 0, psid[i])  # Calcula el cuaternión para el ángulo de cabeceo en el instante i
        quatd[:, i] = quaternion  # Almacena el cuaternión en la columna i de 'quatd'


    
    # Reference Signal of the system  11 states + 4
    xref = np.zeros((15, t.shape[0]), dtype = np.double)
    xref[0, :] = pos_ref[0, :]  # px_d
    xref[1, :] = pos_ref[1, :]  # py_d
    xref[2, :] = pos_ref[2, :]  # pz_d  
    xref[3,:] = quatd[0, :]   #qw 
    xref[4,:] = quatd[1, :]   #qx
    xref[5,:] = quatd[2, :]   #qy 
    xref[6,:] = quatd[3, :]   #qz 
    xref[7,:] = 0   #ul   
    xref[8,:] = 0   #um 
    xref[9,:] = 0   #un
    xref[10,:] = 0  #w

    movil_obs1 = np.zeros((3, t.shape[0]), dtype = np.double)
    movil_obs1[0, :] = xd_obs1(t)  # px_d
    movil_obs1[1, :] = yd_obs1(t)  # py_d
    movil_obs1[2, :] = zd_obs1(t)  # pz_d

    # Initial Control values
    u_control = np.zeros((4, t.shape[0]-N_prediction), dtype = np.double)
    #u_control = np.zeros((4, t.shape[0]), dtype = np.double)

    # Limits Control values
    zp_ref_max = 3
    phi_max = 3
    theta_max = 3
    psi_max = 2

    zp_ref_min = -zp_ref_max
    phi_min = -phi_max
    theta_min = -theta_max
    psi_min = -psi_max

    # Create Optimal problem
    model, f, f_x, g_x = f_system_simple_model_quat()

    ocp = create_ocp_solver_description(x[:,0], N_prediction, t_prediction, zp_ref_max, zp_ref_min, phi_max, phi_min, theta_max, theta_min, psi_max, psi_min)
    #acados_ocp_solver = AcadosOcpSolver(ocp, json_file="acados_ocp_" + ocp.model.name + ".json", build= True, generate= True)

    solver_json = 'acados_ocp_' + model.name + '.json'
    AcadosOcpSolver.generate(ocp, json_file=solver_json)
    AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
    acados_ocp_solver = AcadosOcpSolver.create_cython_solver(solver_json)
    #acados_ocp_solver = AcadosOcpSolverCython(ocp.model.name, ocp.solver_options.nlp_solver_type, ocp.dims.N)

    nx = ocp.model.x.size()[0]
    nu = ocp.model.u.size()[0]

    simX = np.ndarray((nx, N_prediction+1))
    simU = np.ndarray((nu, N_prediction))

     # Initial States Acados
    for stage in range(N_prediction + 1):
        acados_ocp_solver.set(stage, "x", x[:, 0])
    for stage in range(N_prediction):
        acados_ocp_solver.set(stage, "u", np.zeros((nu,)))

    # Errors of the system
    Error = np.zeros((3, t.shape[0]-N_prediction), dtype = np.double)

    # Simulation System
    ros_rate = 30  # Tasa de ROS en Hz
    rate = rospy.Rate(ros_rate)  # Crear un objeto de la clase rospy.Rate


    constraint_expr = ocp.model.con_h_expr

    # Crear la función de CasADi para evaluar la restricción
    constraint_func = Function('constraint_func', [ocp.model.x, ocp.model.u, ocp.model.p], [constraint_expr])


    # Posiciones de los obstáculos y sus radios
    obs_pos = np.array([
    [4.8753, 3.6136, 6.7743],  # 1
    [9.8215, 3.0610, 6.6559],  # 2
    [9.9405, -1.8058, 5.6130],  # 3
    [5.0403, -3.9034, 5.1636],  # 4
     # 5
    [0.5521, 4.5867, 6.9829],  # 6
    [-3.8693, 2.6429, 6.5663],  # 7
    [-3.7215, -3.7539, 5.1956],  # 8
    [0.8770, -4.0461, 5.1330]   # 9
    ])

    obs_ra = 0.8*np.array([0.35, 0.40, 0.45, 0.5, 0.45, 0.35, 0.30, 0.25])


    for k in range(0, t.shape[0]-N_prediction):
        tic = time.time()

        #print(quatd[:, k])
        print(xref[3:7, k])
        
        Error[:,k] = xref[0:3, k] - x[0:3, k]

        # Control Law Section
        acados_ocp_solver.set(0, "lbx", x[:,k])
        acados_ocp_solver.set(0, "ubx", x[:,k])


        distances = np.linalg.norm(obs_pos - x[0:3,k], axis=1)
        idx_closest = np.argmin(distances)  # Encuentra el índice de la distancia mínima

        # Obtener la posición y radio del obstáculo más cercano
        obs_x_closest = obs_pos[idx_closest, 0]
        obs_y_closest = obs_pos[idx_closest, 1]
        obs_z_closest = obs_pos[idx_closest, 2]
        obs_r_closest = obs_ra[idx_closest]

        values = [obs_x_closest ,obs_y_closest, obs_z_closest , obs_r_closest , movil_obs1[0, k], movil_obs1[1, k],movil_obs1[2, k]]

        #AVALUACION DE LA FUNCION DE BARRERA
        obst_static = np.array([obs_x_closest, obs_y_closest ,obs_z_closest]) 
        obst_movil = movil_obs1[:3,k]


        h_CBF_1[:, k] = np.linalg.norm(x[:3, k] - obst_static) - (uav_r + obs_r_closest + margen)
        h_CBF_2[:, k] = np.linalg.norm(x[:3, k] - obst_movil) - (uav_r + obsmovil_r + margen)
        print(h_CBF_1[:, k])

        #constraint_value = constraint_func(x[:, k],u_control[:, k],  np.hstack([xref[:,k], values])   )
        #CLF[:,k] = (constraint_value[2])


        # SET REFERENCES       
        for j in range(N_prediction):
            yref = xref[:,k+j]
            dpds = dp_ds[:,k+j]
            parameters = np.hstack([yref, dpds, values])
            acados_ocp_solver.set(j, "p", parameters )
           
        yref_N = xref[:,k+N_prediction]
        dpds_N = dp_ds[:,k+N_prediction]
        parameters_N = np.hstack([yref_N, dpds_N, values])
        acados_ocp_solver.set(N_prediction, "p", parameters_N)
        
        # get solution
        for i in range(N_prediction):
            simX[:,i] = acados_ocp_solver.get(i, "x")
            simU[:,i] = acados_ocp_solver.get(i, "u")
        simX[:,N_prediction] = acados_ocp_solver.get(N_prediction, "x")

        publish_matrix(simX[0:3, 0:N_prediction], '/Prediction')

        u_control[:, k] = simU[:,0]

        # Get Computational Time
        status = acados_ocp_solver.solve()

        toc_solver = time.time()- tic

        # Get Control Signal
        u_control[:, k] = acados_ocp_solver.get(0, "u")

        #print(u_control[:, k])
        #u_control[:, k] = [0.0, 0.0, 0.0, 0]
        send_velocity_control(u_control[:, k], vel_pub, vel_msg)

        # System Evolution
        opcion = "Sim"  # Valor que quieres evaluar

        if opcion == "Real":
            x[:, k+1] = get_odometry_simple_quat_1()
        elif opcion == "Sim":
            x[:, k+1] = f_d(x[:, k], u_control[:, k], t_s, f)
            pub_odometry_sim_quat(x[:, k+1], odom_sim_pub_1, odom_sim_msg_1)
        else:
            print("Opción no válida")
        
        
        delta_t[:, k] = toc_solver
        
        #print("x:", " ".join("{:.2f}".format(value) for value in np.round(x[0:12, k], decimals=2)))
        
        rate.sleep() 
        toc = time.time() - tic 

        e_contorno[:,k], e_arrastre[:,k], e_total[:,k], vel_progres[:,k] = calculate_errors_norm(xref[0:3,k], xref[3:6,k], x[:,k])

        
        vel_progress_ref[:, k] =  0
        
        
    
    send_velocity_control([0, 0, 0, 0], vel_pub, vel_msg)

    fig1 = plot_pose(x, xref, t)
    fig1.savefig("1_pose.png")
    fig2 = plot_error(Error, t)
    fig2.savefig("2_error_pose.png")
    fig3 = plot_time(t_sample, delta_t , t)
    fig3.savefig("3_Time.png")

    print(f'Mean iteration time with MLP Model: {1000*np.mean(delta_t):.1f}ms -- {1/np.mean(delta_t):.0f}Hz)')


            #For MODEL TESTS
    # Ruta que deseas verificar
    pwd = "/home/bryansgue/Doctoral_Research/Matlab/Results_MPCC_CLF_CBF_uav"

    # Verificar si la ruta no existe
    if not os.path.exists(pwd) or not os.path.isdir(pwd):
        print(f"La ruta {pwd} no existe. Estableciendo la ruta local como pwd.")
        pwd = os.getcwd()  # Establece la ruta local como pwd

    #SELECCION DEL EXPERIMENTO
   
    experiment_number = 1
    name_file = "Results_MPCC_CLF_CBF_uav" + str(experiment_number) + ".mat"
    
    save = True
    if save==True:
        savemat(os.path.join(pwd, name_file), {
            'states': x,
            'T_control': u_control,
            'CBF_1': h_CBF_1,
            'CBF_2': h_CBF_2,
            'CLF': CLF,
            'time': t,
            'ref': xref,
            'obs_movil': movil_obs1,
            'e_total': e_total,
            'e_contorno': e_contorno,
            'e_arrastre': e_arrastre,
            'vel_progres': vel_progres,
            'vel_progres_ref':vel_progress_ref})

    return None



if __name__ == '__main__':
    try:
        # Node Initialization
        rospy.init_node("Acados_controller",disable_signals=True, anonymous=True)

        # SUCRIBER
        Matrice_1 = rospy.Subscriber("/dji_sdk/odometry", Odometry, odometry_call_back_1)
        Matrice_2 = rospy.Subscriber("/M2/dji_sdk/odometry", Odometry, odometry_call_back_2)
        
        # PUBLISHER
        velocity_message = TwistStamped()
        velocity_publisher = rospy.Publisher("/m100/velocityControl", TwistStamped, queue_size=10)

        odometry_sim_msg_1 = Odometry()
        odom_sim_pub_1 = rospy.Publisher('/dji_sdk/odometry', Odometry, queue_size=10)
    

        main(velocity_publisher, velocity_message, odom_sim_pub_1, odometry_sim_msg_1)
    except(rospy.ROSInterruptException, KeyboardInterrupt):
        print("\nError System")
        send_velocity_control([0, 0, 0, 0], velocity_publisher, velocity_message)
        pass
    else:
        print("Complete Execution")
        pass
