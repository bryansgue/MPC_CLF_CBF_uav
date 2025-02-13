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
from casadi import jacobian
from casadi import sqrt
from std_msgs.msg import Float64MultiArray
import rospy
from scipy.spatial.transform import Rotation as R
from nav_msgs.msg import Odometry
#from c_generated_code.acados_ocp_solver_pyx import AcadosOcpSolverCython
from geometry_msgs.msg import TwistStamped
import math
from scipy.io import savemat
import os

# CARGA FUNCIONES DEL PROGRAMA
from fancy_plots import plot_pose, plot_error, plot_time
from Functions_SimpleModel import f_system_simple_model
from Functions_SimpleModel import f_d, odometry_call_back, get_odometry_simple, send_velocity_control, pub_odometry_sim, calc_J
import P_UAV_simple


def publish_matrix(matrix_data, topic_name='/nombre_del_topico'):
    """
    Publica una matriz en un tópico ROS.

    Args:
        matrix_data (numpy.ndarray): La matriz a publicar.
        topic_name (str): El nombre del tópico ROS en el que se publicará la matriz.
    """
    # Inicializa el nodo ROS si aún no está inicializado
   

    # Crea una instancia del mensaje Float64MultiArray
    matrix_msg = Float64MultiArray()

    # Convierte la matriz NumPy en una lista plana
    matrix_data_flat = matrix_data.flatten().tolist()

    # Asigna los datos de la matriz al mensaje
    matrix_msg.data = matrix_data_flat

    # Crea un publicador para el tópico deseado
    matrix_publisher = rospy.Publisher(topic_name, Float64MultiArray, queue_size=10)

    # Publica el mensaje en el tópico
    matrix_publisher.publish(matrix_msg)

def create_ocp_solver_description(x0, N_horizon, t_horizon, zp_max, zp_min, phi_max, phi_min, theta_max, theta_min, psi_max, psi_min) -> AcadosOcp:
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    model, f_system, f_x, g_x = f_system_simple_model()
    ocp.model = model
    ocp.p = model.p
    ocp.xdot = model.xdot
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ny = nx + nu

    # set dimensions
    ocp.dims.N = N_horizon

    Q_mat = MX.zeros(8, 8)
    Q_mat[0, 0] = 2
    Q_mat[1, 1] = 2
    Q_mat[2, 2] = 2
    Q_mat[3, 3] = 2

    R_mat = MX.zeros(4, 4)
    R_mat[0, 0] = 1.3*(1/2)
    R_mat[1, 1] = 1.3*(1/2)
    R_mat[2, 2] = 1.3*(1/2)
    R_mat[3, 3] = 1.3*(1/2)
    
    ocp.parameter_values = np.zeros(ny)

    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    error_pose = ocp.p[0:8] - model.x[0:8]
    ocp.model.cost_expr_ext_cost = model.u.T @ R_mat @ model.u 
    ocp.model.cost_expr_ext_cost_e = error_pose.T @ Q_mat @ error_pose

    ##·······································
    u = model.u
    x = model.x[0:8]
    x_e = error_pose

    x_p = f_x + g_x@u

    V = (1/2)*x_e.T@ Q_mat @ x_e
    delta_v = jacobian(V, model.x)
    #delta_v = Q_mat @ x_e
    V_p = -x_e.T@ Q_mat@ x_p

    #V_p = delta_v@f_x + delta_v@g_x@u

    obs_x = 2.9      
    obs_y= 0.17
    obs_r = 0.4
    rob_r = 0.4
    B = -sqrt((model.x[0] - obs_x)**2 + (model.x[1] - obs_y)**2) + (rob_r + obs_r)  
    delta_B =  jacobian(B, model.x) 
   
    B_p = delta_B@f_x + delta_B@g_x@u

    print(V_p.shape)

    
    # set constraints
    ocp.constraints.constr_type = 'BGH'

    ocp.constraints.lbu = np.array([-5, -5, -5])
    ocp.constraints.ubu = np.array([5, 5, 5])
    ocp.constraints.idxbu = np.array([0, 1, 2])

    constraints = vertcat(V_p + 0.9*V , B_p + 1*B)
    #constraints = vertcat(B_p + 1*B)
    #constraints = vertcat(model.x[1])
    
    ocp.model.con_h_expr = constraints
    Dim_constraints = 2
    #ocp.model.con_h_expr_e =  ocp.model.con_h_expr


    # We put all constraint cost weights to 0 and only set them at runtime
    cost_weights = np.ones(Dim_constraints)
    ocp.cost.zl = cost_weights
    ocp.cost.Zl = 1e2 *cost_weights
    ocp.cost.Zu = 1e2* cost_weights
    ocp.cost.zu = cost_weights

    ocp.constraints.lh =-1e9* np.ones(Dim_constraints)  #min
    #ocp.constraints.lh_e = -1e9* np.ones(Dim_constraints)
    ocp.constraints.uh = 0.0*np.ones(Dim_constraints) #max
    #ocp.constraints.uh_e = np.zeros(Dim_constraints)
    ocp.constraints.idxsh = np.arange(Dim_constraints)
    
    ocp.constraints.x0 = x0

    # set options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"  # 'GAUSS_NEWTON', 'EXACT'
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"  # SQP_RTI, SQP
    ocp.solver_options.tol = 1e-4

    # set prediction horizon
    ocp.solver_options.tf = t_horizon

    return ocp

def main(vel_pub, vel_msg, odom_sim_pub, odom_sim_msg):
    # Initial Values System
    # Simulation Time
    t_final = 60
    # Sample time
    frec= 30
    t_s = 1/frec
    # Prediction Time
    N_horizont = 60
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
    x = np.zeros((8, t.shape[0]+1-N_prediction), dtype = np.double)
    x_sim = np.zeros((8, t.shape[0]+1-N_prediction), dtype = np.double)

    # Read Real data
    x[:, 0] = get_odometry_simple()

    #TAREA DESEADA
    value = 9
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

    # Reference Signal of the system
    xref = np.zeros((12, t.shape[0]), dtype = np.double)
    xref[0,:] = hxd 
    xref[1,:] = hyd
    xref[2,:] = hzd  
    xref[3,:] = 0*psid 
    xref[4,:] = 0
    xref[5,:] = 0 
    xref[6,:] = 0 
    xref[7,:] = 0 
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
    
    # Simulation System
    ros_rate = 30  # Tasa de ROS en Hz
    rate = rospy.Rate(ros_rate)  # Crear un objeto de la clase rospy.Rate

    #P_UAV_simple.main(vel_pub, vel_msg )

    #INICIALIZA LECTURA DE ODOMETRIA
    for k in range(0, 10):
        # Read Real data
        x[:, 0] = get_odometry_simple()
        # Loop_rate.sleep()
        rate.sleep() 
        print("Init System")

    # Create Optimal problem
    model, f, f_x, g_x= f_system_simple_model()

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
        acados_ocp_solver.set(stage, "x", 0.0 * np.ones(x[:,0].shape))
    for stage in range(N_prediction):
        acados_ocp_solver.set(stage, "u", np.zeros((nu,)))

    # Errors of the system
    Error = np.zeros((4, t.shape[0]-N_prediction), dtype = np.double)

    for k in range(0, t.shape[0]-N_prediction):
              
        Error[:,k] = xref[0:4, k] - x[0:4, k]

        # Control Law Section
        acados_ocp_solver.set(0, "lbx", x[:,k])
        acados_ocp_solver.set(0, "ubx", x[:,k])

        # SET REFERENCES
        for j in range(N_prediction):
            yref = xref[:,k+j]
            acados_ocp_solver.set(j, "p", yref)

        yref_N = xref[:,k+N_prediction]
        acados_ocp_solver.set(N_prediction, "p", yref_N)
        
        
        # Get Computational Time
        tic = time.time()
        status = acados_ocp_solver.solve()
        toc_solver = time.time()- tic

        # get solution
        for i in range(N_prediction):
            simX[:,i] = acados_ocp_solver.get(i, "x")
            simU[:,i] = acados_ocp_solver.get(i, "u")
        simX[:,N_prediction] = acados_ocp_solver.get(N_prediction, "x")

        publish_matrix(simX[0:3, 0:N_prediction], '/Prediction')
        print(simX[0:3, 0:N_prediction].shape)

        u_control[:, k] = simU[:,0]
        #u_control[:, k] = [0.0,0.0,0,0.5]
        send_velocity_control(u_control[:, k], vel_pub, vel_msg)

        # System Evolution
        J = np.zeros((4, 4))
        J[0, 0] = np.cos(x[3,k])
        J[0, 1] = -np.sin(x[3,k])
        J[1, 0] = np.sin(x[3,k])
        J[1, 1] = np.cos(x[3,k])
        J[2, 2] = 1
        J[3, 3] = 1

        q_p = J@x[4:8,k]
        q_e = Error[:,k]

        Q = np.zeros((4, 4))
        Q[0, 0] = 2
        Q[1, 1] = 2
        Q[2, 2] = 2
        Q[3, 3] = 2
        Vp = -q_e.T@Q@q_p
        print(Vp)

        opcion = "Sim"  # Valor que quieres evaluar

        if opcion == "Real":
            x[:, k+1] = get_odometry_simple()
        elif opcion == "Sim":
            x[:, k+1] = f_d(x[:, k], u_control[:, k], t_s, f)
            pub_odometry_sim(x[:, k+1], odom_sim_pub, odom_sim_msg)
        else:
            print("Opción no válida")
        
        delta_t[:, k] = toc_solver
        

        
        
        print("x:", " ".join("{:.2f}".format(value) for value in np.round(x[0:12, k], decimals=2)))
        
        rate.sleep() 
        toc = time.time() - tic 
        
        
    send_velocity_control([0, 0, 0, 0], vel_pub, vel_msg)

    fig1 = plot_pose(x, xref, t)
    fig1.savefig("1_pose.png")
    fig2 = plot_error(Error, t)
    fig2.savefig("2_error_pose.png")
    fig3 = plot_time(t_sample, delta_t , t)
    fig3.savefig("3_Time.png")

    #For MODEL TESTS
    x_data = {"states_MPC": x, "label": "x"}
    xref_data = {"ref_MPC": xref, "label": "xref"}
    t_data = {"t_MPC": t, "label": "time"}

    # Ruta que deseas verificar
    pwd = "/home/bryansgue/Doctoral_Research/Matlab/MPCC_uav_simple"

    # Verificar si la ruta no existe
    if not os.path.exists(pwd) or not os.path.isdir(pwd):
        print(f"La ruta {pwd} no existe. Estableciendo la ruta local como pwd.")
        pwd = os.getcwd()  # Establece la ruta local como pwd

    Test = "Real"

    if Test == "MiL":
        savemat(os.path.join(pwd, "x_MiL_MPC.mat"), x_data)
        savemat(os.path.join(pwd, "xref_MiL_MPC.mat"), xref_data)
        savemat(os.path.join(pwd, "t_MiL_MPC.mat"), t_data)
    elif Test == "HiL":
        savemat(os.path.join(pwd, "x_HiL_MPC.mat"), x_data)
        savemat(os.path.join(pwd, "xref_HiL_MPC.mat"), xref_data)
        savemat(os.path.join(pwd, "t_HiL_MPC.mat"), t_data)
    elif Test == "Real":
        savemat(os.path.join(pwd, "x_Real_MPC.mat"), x_data)
        savemat(os.path.join(pwd, "xref_Real_MPC.mat"), xref_data)
        savemat(os.path.join(pwd, "t_Real_MPC.mat"), t_data)


    print(f'Mean iteration time with MLP Model: {1000*np.mean(delta_t):.1f}ms -- {1/np.mean(delta_t):.0f}Hz)')



if __name__ == '__main__':
    try:
        # Node Initialization
        rospy.init_node("Acados_controller",disable_signals=True, anonymous=True)

        # SUCRIBER
        velocity_subscriber = rospy.Subscriber("/dji_sdk/odometry", Odometry, odometry_call_back)
        
        # PUBLISHER
        velocity_message = TwistStamped()
        velocity_publisher = rospy.Publisher("/m100/velocityControl", TwistStamped, queue_size=10)

        odometry_sim_msg = Odometry()
        odom_sim_pub = rospy.Publisher('/dji_sdk/odometry', Odometry, queue_size=10)
    
        
    

        main(velocity_publisher, velocity_message, odom_sim_pub, odometry_sim_msg)
    except(rospy.ROSInterruptException, KeyboardInterrupt):
        print("\nError System")
        send_velocity_control([0, 0, 0, 0], velocity_publisher, velocity_message)
        pass
    else:
        print("Complete Execution")
        pass
