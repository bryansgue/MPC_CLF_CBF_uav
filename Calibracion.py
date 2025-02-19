#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Point, Pose2D, Twist, Pose, TwistStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Int32MultiArray
import matplotlib.pyplot as plt
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Joy
import math

from Functions_SimpleModel import odometry_call_back, get_odometry_simple, send_velocity_control
from Functions_DinamicControl import calc_G, calc_C, calc_M, calc_J, limitar_angulo
from fancy_plots import plot_pose, plot_error, plot_time


# Tiempo de inicio cuando se detecta la condición



def main(vel_pub, vel_msg ):

    print("OK, controller is running!!!")
    
    timerun = 60  # Segundos
    hz = 30  # Frecuencia de actualización
    ts = 1 / hz
    samples = timerun * hz  # datos de muestreo totales
    
    # Inicialización de matrices
    t = np.arange(0, samples * ts, ts)
    x = np.zeros((8, samples))
    uc = np.zeros((4, samples))
    uref = np.zeros((4, samples))
    he = np.zeros((4, samples))
    ue = np.zeros((4, samples))
    psidp = np.zeros(samples)

    #GANANCIAS DEL CONTROLADOR
    K1 = np.diag([1,1,1,1])  
    K2 = np.diag([1,1,1,1])  
    K3 = np.diag([1,1,1,1])  
    K4 = np.diag([1,1,1,1])  
    
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
    psidp = np.gradient(psid, ts)

    # Vector Initial conditions
    a = 0
    b = 0
    
    # Reference Signal of the system
    xref = np.zeros((12, t.shape[0]), dtype = np.double)
    xref[0,:] = 0
    xref[1,:] = 0
    xref[2,:] = 6.5
    xref[3,:] = 90* (math.pi / 180)
    xref[4,:] = 0
    xref[5,:] = 0
    xref[6,:] = 0
    xref[7,:] = 0

    # Simulation System
    ros_rate = 30  # Tasa de ROS en Hz
    rate = rospy.Rate(ros_rate)  # Crear un objeto de la clase rospy.Rate

    #INICIALIZA LECTURA DE ODOMETRIA
    for k in range(0, 10):
        # Read Real data
        x[:, 0] = get_odometry_simple()
        # Loop_rate.sleep()
        rate.sleep() 
        print("Init System Position")

    start_time = None  
    
    for k in range(samples-1):
        #INICIO DEL TIEMPO DE BUCLE
        tic = time.time()

        # MODELO CINEMATICO Y DINAMICO
        chi = [0.6756,    1.0000,    0.6344,    1.0000,    0.4080,    1.0000,    1.0000,    1.0000,    0.2953,    0.5941,   -0.8109,    1.0000,    0.3984,    0.7040,    1.0000,    0.9365,    1.0000, 1.0000,    0.9752]# Position
        
        J = calc_J(x[:, k])
        M = calc_M(chi, a, b)
        C = calc_C(chi, a, b, x[:,k])
        G = calc_G()
      
        # CONTROLADOR CINEMATICO
        he[:, k] = xref[0:4, k] - x[0:4, k]
        he[3, k] =  limitar_angulo(he[3, k])
        #uc[:, k] = np.linalg.pinv(J) @ (xref[4:8, k] + K1 @ np.tanh(K2 @ he[:, k]))
        uc[:, k] = np.linalg.pinv(J) @ (K1 @ np.tanh(K2 @ he[:, k]))
  
        if k > 0:
            vcp = (uc[:, k] - uc[:, k - 1]) / ts
        else:
            vcp = uc[:, k] / ts
     
        #COMPENSADOR DINAMICO
        ue[:, k] = uc[:, k] - x[4:8, k]
        control = 0*vcp + K3 @ np.tanh(np.linalg.inv(K3) @ K4 @ ue[:, k])
        uref[:, k] = M @ control + C @ uc[:, k]

        #ENVIO DE VELOCIDADES AL DRON
        send_velocity_control(uc[:, k], vel_pub, vel_msg )

        #LECTURA DE ODOMETRIA MODELO SIMPLIFICADO
        x[:, k+1] = get_odometry_simple()

        rate.sleep() 
        toc = time.time() - tic 

        #Condicion para romper:
        umbral = 0.03

        # Calcula la norma 2 de toda la columna
        norma_2_columna = np.linalg.norm(he[:, k])  # Calcula la norma 2 de la columna k

 

        if norma_2_columna < umbral:
            if start_time is None:
                start_time = time.time()  # Inicia el temporizador solo la primera vez
            
            elapsed_time_s = time.time() - start_time  # Tiempo en segundos
            print(f"Condición activa por {elapsed_time_s:.2f} s", end='\r')  # Mostrar en segundos con 2 decimales

            # Si la condición se mantiene por al menos 0.5 segundos, se detiene
            if elapsed_time_s >= 0.5:
                send_velocity_control([0, 0, 0, 0], vel_pub, vel_msg)
                print("\nPose Init:", [round(element, 2) for element in x[:, k+1]])
                break
        else:
            start_time = None  # Reiniciar el temporizador si la condición deja de cumplirse

                
        print("Error:", " ".join("{:.2f}".format(value) for value in np.round(he[:, k], decimals=2)), end='\r')

    #send_velocity_control([0, 0, 0, 0], velocity_publisher, velocity_message)
 
    fig1 = plot_pose(x, xref, t)
    fig1.savefig("1_pose.png")
    fig2 = plot_error(he, t)
    fig2.savefig("2_error_pose.png")


if __name__ == '__main__':
    try:
        # Node Initialization
        rospy.init_node("Acados_controller",disable_signals=True, anonymous=True)

        # SUCRIBER
        velocity_subscriber = rospy.Subscriber("/dji_sdk/odometry", Odometry, odometry_call_back)
        
        # PUBLISHER
        velocity_message = TwistStamped()
        velocity_publisher = rospy.Publisher("/m100/velocityControl", TwistStamped, queue_size=10)

        main(velocity_publisher, velocity_message)
    except(rospy.ROSInterruptException, KeyboardInterrupt):
        print("Error System")
        
        send_velocity_control([0, 0, 0, 0], velocity_publisher, velocity_message)
        pass
    else:
        print("Complete Execution")
        pass
