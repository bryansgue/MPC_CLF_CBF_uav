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
import rospy
from scipy.spatial.transform import Rotation as R
from nav_msgs.msg import Odometry
#from c_generated_code.acados_ocp_solver_pyx import AcadosOcpSolverCython
from geometry_msgs.msg import TwistStamped
import math

# CARGA FUNCIONES DEL PROGRAMA
from fancy_plots import plot_pose, plot_error, plot_time

from Functions_SimpleModel import f_system_simple_model_quat
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


def create_ocp_solver_description(x0, N_horizon, t_horizon, zp_max, zp_min, phi_max, phi_min, theta_max, theta_min, psi_max, psi_min) -> AcadosOcp:
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    model, f_system, f_x, g_x  = f_system_simple_model_quat()
    ocp.model = model
    ocp.p = model.p
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ny = nx + nu + 2

    # set dimensions
    ocp.dims.N = N_horizon

    Q_mat = MX.zeros(3, 3)
    Q_mat[0, 0] = 2.1
    Q_mat[1, 1] = 2.1
    Q_mat[2, 2] = 2.1

    R_mat = MX.zeros(4, 4)
    R_mat[0, 0] = 1
    R_mat[1, 1] = 1
    R_mat[2, 2] = 1
    R_mat[3, 3] = 1
    
    ocp.parameter_values = np.zeros(ny)

    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    error_pose = ocp.p[0:3] - model.x[0:3]


    quat_error = quaternion_error(ocp.p[3:7], model.x[3:7])
    ocp.model.cost_expr_ext_cost = error_pose.T @ Q_mat @error_pose  + model.u.T @ R_mat @ model.u + 1 * (1 - quat_error[0]) + 7 * (quat_error[1:]).T @ quat_error[1:]
    ocp.model.cost_expr_ext_cost_e = error_pose.T @ Q_mat @ error_pose +  1 * (1 - quat_error[0]) + 7 * (quat_error[1:4]).T @ quat_error[1:4]


    # set constraints
    u = model.u
    x = model.x[0:11]

    obs_x = ocp.p[15,0]     # Posicion donde actualiza la posicion del obstaculo
    obs_y= ocp.p[16,0]
    obs_r = 0.4
    rob_r = 0.4
    h = sqrt((model.x[0] - obs_x)**2 + (model.x[1] - obs_y)**2) - (rob_r + obs_r)  
    
    f_x_val = f_x(model.x)
    g_x_val = g_x(model.x,model.u)


    # Derivada de Lie de primer orden
    Lf_h = jacobian(h, model.x) @ f_x_val 
    Lg_h = jacobian(h, model.x) @ g_x_val

    # Derivada de Lie de segundo orden
    Lf2_h = jacobian(Lf_h, model.x) @ f_x_val
    Lg_L_f_h = jacobian(Lf_h, model.x) @ g_x_val 

    # Derivada de Lie de tercer orden
    Lf3_h = jacobian(Lf2_h, model.x) @ f_x_val 
    Lg_L_f2_h = jacobian(Lf2_h, model.x) @ g_x_val  

    # Barreras temporales
    h_p = Lf_h + Lg_h @ model.u
    h_pp = Lf2_h + Lg_L_f_h @ model.u
    h_ppp = Lf3_h + Lg_L_f2_h @ model.u

    nb_1 = h
    nb_2 = vertcat(h, Lf_h) 
    nb_3 = vertcat(h, Lf_h, Lf2_h) 

    K_alpha = MX([10, 10]).T 

    K_beta = MX([10, 10, 10]).T 

    # set constraints
    ocp.constraints.constr_type = 'BGH'

    ocp.constraints.lbu = np.array([-8, -8, -8])
    ocp.constraints.ubu = np.array([8, 8, 8])
    ocp.constraints.idxbu = np.array([0, 1, 2])

    constraints = vertcat(h_p + 5*h, h_pp +  K_alpha @ nb_2 )
    #constraints = vertcat(B_p + 1*B)
    #constraints = vertcat(model.x[1])
    
    ocp.model.con_h_expr = constraints
    Dim_constraints = 2
    #ocp.model.con_h_expr_e =  ocp.model.con_h_expr

    # We put all constraint cost weights to 0 and only set them at runtime
    cost_weights = np.ones(Dim_constraints)
    ocp.cost.zl = cost_weights
    ocp.cost.Zl = 100*cost_weights
    ocp.cost.Zu = 100* cost_weights
    ocp.cost.zu = cost_weights

    ocp.constraints.lh = np.array([0,0])  #min
    #ocp.constraints.lh_e = -1e9* np.ones(Dim_constraints)
    ocp.constraints.uh = np.array([1e9,1e9])#max
    #ocp.constraints.uh_e = np.zeros(Dim_constraints)
    ocp.constraints.idxsh = np.arange(Dim_constraints)

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

def main(vel_pub, vel_msg, odom_sim_pub_1, odom_sim_msg_1):
    # Initial Values System
    # Simulation Time
    t_final = 60
    # Sample time
    frec= 30
    t_s = 1/frec
    # Prediction Time
    N_horizont = 50
    t_prediction = N_horizont/frec

    # Nodes inside MPC
    N = np.arange(0, t_prediction + t_s, t_s)
    N_prediction = N.shape[0]

    # Time simulation
    t = np.arange(0, t_final + t_s, t_s)

    # Sample time vector
    delta_t = np.zeros((1, t.shape[0] - N_prediction), dtype=np.double)
    t_sample = t_s*np.ones((1, t.shape[0] - N_prediction), dtype=np.double)

    # Vector Initial conditions
    x = np.zeros((11, t.shape[0]+1-N_prediction), dtype = np.double)
   

    # Read Real data
    x[:, 0] = get_odometry_simple_quat_1()
    x[:, 0] = [1,1,5,1,0,0,0.5,0,0,0,0]

    #TAREA DESEADA
    value = 15
    xd = lambda t: 4 * np.sin(value*0.04*t) + 3
    yd = lambda t: 4 * np.sin(value*0.08*t)
    zd = lambda t: 2 * np.sin(value*0.08*t) + 6
    xdp = lambda t: 4 * value * 0.04 * np.cos(value*0.04*t)
    ydp = lambda t: 4 * value * 0.08 * np.cos(value*0.08*t)
    zdp = lambda t: 2 * value * 0.08 * np.cos(value*0.08*t)

    hxd = xd(t)
    hyd = yd(t)
    hzd = zd(t)             
    hxdp = xdp(t)
    hydp = ydp(t)
    hzdp = zdp(t)

    psid = np.arctan2(hydp, hxdp)
    psidp = np.gradient(psid, t_s)

    #quaternion = euler_to_quaternion(0, 0, psid) 
    quatd= np.zeros((4, t.shape[0]), dtype = np.double)


    # Calcular los cuaterniones utilizando la función euler_to_quaternion para cada psid
    for i in range(t.shape[0]):
        quaternion = euler_to_quaternion(0, 0, psid[i])  # Calcula el cuaternión para el ángulo de cabeceo en el instante i
        quatd[:, i] = quaternion  # Almacena el cuaternión en la columna i de 'quatd'


    
    # Reference Signal of the system  11 states + 4
    xref = np.zeros((15, t.shape[0]), dtype = np.double)
    xref[0,:] = hxd  #nx
    xref[1,:] = hyd  #ny
    xref[2,:] = hzd  #nz  
    xref[3,:] = quatd[0, :]   #qw 
    xref[4,:] = quatd[1, :]   #qx
    xref[5,:] = quatd[2, :]   #qy 
    xref[6,:] = quatd[3, :]   #qz 
    xref[7,:] = 0   #ul   
    xref[8,:] = 0   #um 
    xref[9,:] = 0   #un
    xref[10,:] = 0  #w

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
        acados_ocp_solver.set(stage, "x", 1 * np.ones(x[:,0].shape))
    for stage in range(N_prediction):
        acados_ocp_solver.set(stage, "u", np.zeros((nu,)))

    # Errors of the system
    Error = np.zeros((3, t.shape[0]-N_prediction), dtype = np.double)

    # Simulation System
    ros_rate = 30  # Tasa de ROS en Hz
    rate = rospy.Rate(ros_rate)  # Crear un objeto de la clase rospy.Rate

    for k in range(0, t.shape[0]-N_prediction):
        tic = time.time()

        #print(quatd[:, k])
        print(xref[3:7, k])
        
        Error[:,k] = xref[0:3, k] - x[0:3, k]

        # Control Law Section
        acados_ocp_solver.set(0, "lbx", x[:,k])
        acados_ocp_solver.set(0, "ubx", x[:,k])

        # SET OBSTACLE
        matrice_2 = get_odometry_simple_quat_2()
        
        nx_obs = 1* matrice_2[0]
        ny_obs = 1* matrice_2[1]

        print(matrice_2[0])

        # SET REFERENCES       
        for j in range(N_prediction):
            yref = xref[:,k+j]
            acados_ocp_solver.set(j, "p",  np.append(yref, [nx_obs , ny_obs]))
           

        yref_N = xref[:,k+N_prediction]
        acados_ocp_solver.set(N_prediction, "p",np.append(yref_N, [nx_obs , ny_obs]))
        
        # get solution
        for i in range(N_prediction):
            simX[:,i] = acados_ocp_solver.get(i, "x")
            simU[:,i] = acados_ocp_solver.get(i, "u")
        simX[:,N_prediction] = acados_ocp_solver.get(N_prediction, "x")

        publish_matrix(simX[0:3, 0:N_prediction], '/Prediction_1')

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
        
        
    
    send_velocity_control([0, 0, 0, 0], vel_pub, vel_msg)

    fig1 = plot_pose(x, xref, t)
    fig1.savefig("1_pose.png")
    fig2 = plot_error(Error, t)
    fig2.savefig("2_error_pose.png")
    fig3 = plot_time(t_sample, delta_t , t)
    fig3.savefig("3_Time.png")

    print(f'Mean iteration time with MLP Model: {1000*np.mean(delta_t):.1f}ms -- {1/np.mean(delta_t):.0f}Hz)')



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
