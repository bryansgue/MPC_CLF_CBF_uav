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
from fancy_plots import fancy_plot
import rospy
from scipy.spatial.transform import Rotation as R
from nav_msgs.msg import Odometry
#from c_generated_code.acados_ocp_solver_pyx import AcadosOcpSolverCython
from geometry_msgs.msg import TwistStamped
import math

# CARGA FUNCIONES DEL PROGRAMA
from fancy_plots import plot_pose, plot_error, plot_time
from Functions_SimpleModel import f_system_simple_model
from Functions_SimpleModel import f_d, odometry_call_back, get_odometry_simple, send_velocity_control, pub_odometry_sim


def create_ocp_solver_description(x0, N_horizon, t_horizon, zp_max, zp_min, phi_max, phi_min, theta_max, theta_min, psi_max, psi_min) -> AcadosOcp:
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    model, f_system, f_x, g_x = f_system_simple_model()
    ocp.model = model
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    
    ny = nx + nu
    ny_e = nx

    # set dimensions
    ocp.dims.N = N_horizon

    # set cost
    
    Q_mat = np.zeros((8, 8))
    Q_mat[0, 0] = 1.1
    Q_mat[1, 1] = 1.1
    Q_mat[2, 2] = 1.1
    Q_mat[3, 3] = 1

    R_mat = np.zeros((4, 4))
    R_mat[0, 0] = 1.3*(1/2)
    R_mat[1, 1] = 1.3*(1/2)
    R_mat[2, 2] = 1.3*(1/2)
    R_mat[3, 3] = 1.3*(1/2)

    ocp.parameter_values = np.zeros(ny)
    
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    

    ocp.cost.W_e = Q_mat
    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)

    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx, :nx] = np.eye(nx)

    Vu = np.zeros((ny, nu))
    Vu[nx : nx + nu, 0:nu] = np.eye(nu)
    ocp.cost.Vu = Vu

    ocp.cost.Vx_e = np.eye(nx)

    ocp.cost.yref = np.zeros((ny,))
    ocp.cost.yref_e = np.zeros((ny_e,))

    # set constraints
    ocp.constraints.lbu = np.array([-2, -2, -2])
    ocp.constraints.ubu = np.array([2, 2, 2])
    ocp.constraints.idxbu = np.array([0, 1, 2])

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

def main(vel_pub, vel_msg, odom_sim_pub, odom_sim_msg):
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


    # Vector Initial conditions
    x = np.zeros((8, t.shape[0]+1-N_prediction), dtype = np.double)
    x_sim = np.zeros((8, t.shape[0]+1-N_prediction), dtype = np.double)

    # Read Real data
    x[:, 0] = get_odometry_simple()

    #TAREA DESEADA
    num = 4
    xd = lambda t: 4 * np.sin(5*0.04*t) + 3
    yd = lambda t: 4 * np.sin(5*0.08*t)
    zd = lambda t: 2.5 * np.sin (0.2* t) + 5  
    xdp = lambda t: 4 * 5 * 0.04 * np.cos(5*0.04*t)
    ydp = lambda t: 4 * 5 * 0.08 * np.cos(5*0.08*t)
    zdp = lambda t: 2.5 * 0.2 * np.cos(0.2 * t)

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
    xref[3,:] = psid 
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

    #INICIALIZA LECTURA DE ODOMETRIA
    for k in range(0, 100):
        # Read Real data
        x[:, 0] = get_odometry_simple()
        # Loop_rate.sleep()
        rate.sleep() 
        print("Init System")

    # Create Optimal problem
    model, f, f_x, g_x = f_system_simple_model()

    ocp = create_ocp_solver_description(x[:,0], N_prediction, t_prediction, zp_ref_max, zp_ref_min, phi_max, phi_min, theta_max, theta_min, psi_max, psi_min)
    #acados_ocp_solver = AcadosOcpSolver(ocp, json_file="acados_ocp_" + ocp.model.name + ".json", build= True, generate= True)

    solver_json = 'acados_ocp_' + model.name + '.json'
    AcadosOcpSolver.generate(ocp, json_file=solver_json)
    AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
    acados_ocp_solver = AcadosOcpSolver.create_cython_solver(solver_json)
    #acados_ocp_solver = AcadosOcpSolverCython(ocp.model.name, ocp.solver_options.nlp_solver_type, ocp.dims.N)

    nx = ocp.model.x.size()[0]
    nu = ocp.model.u.size()[0]

    # Initial States Acados
    for stage in range(N_prediction + 1):
        acados_ocp_solver.set(stage, "x", 0.0 * np.ones(x[:,0].shape))
    for stage in range(N_prediction):
        acados_ocp_solver.set(stage, "u", np.zeros((nu,)))

    # Errors of the system
    Error = np.zeros((3, t.shape[0]-N_prediction), dtype = np.double)

    

    for k in range(0, t.shape[0]-N_prediction):
        tic = time.time()

        Error[:,k] = xref[0:3, k] - x[0:3, k]

        # Control Law Section
        acados_ocp_solver.set(0, "lbx", x[:,k])
        acados_ocp_solver.set(0, "ubx", x[:,k])

        # update yref
        for j in range(N_prediction):
            yref = xref[:,k+j]
            acados_ocp_solver.set(j, "yref", yref)
        
        yref_N = xref[:,k+N_prediction]
        acados_ocp_solver.set(N_prediction, "yref", yref_N[0:8])

        # Get Computational Time
        status = acados_ocp_solver.solve()

        toc_solver = time.time()- tic

        # Get Control Signal
        u_control[:, k] = acados_ocp_solver.get(0, "u")
        #u_control[:, k] = [0.0,0.0,0,0.0]
        send_velocity_control(u_control[:, k], vel_pub, vel_msg)

        # System Evolution
        opcion = "Sim"  # Valor que quieres evaluar

        if opcion == "Real":
            x[:, k+1] = get_odometry_simple()
        elif opcion == "Sim":
            x[:, k+1] = f_d(x[:, k], u_control[:, k], t_s, f)
            pub_odometry_sim(x[:, k+1], odom_sim_pub, odom_sim_msg)
        else:
            print("Opción no válida")
        

        delta_t[:, k] = toc_solver
        
        print("v_real:", " ".join("{:.2f}".format(value) for value in np.round(x[0:12, k], decimals=2)))
        
        rate.sleep() 
        toc = time.time() - tic 
        #print(1/toc)
        
    
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
