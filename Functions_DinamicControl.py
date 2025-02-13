import numpy as np

def limitar_angulo(ErrAng):

    if ErrAng >= 1 * np.pi:
        while ErrAng >= 1 * np.pi:
            ErrAng -= 2 * np.pi
        return ErrAng

    if ErrAng <= -1 * np.pi:
        while ErrAng <= -1 * np.pi:
            ErrAng += 2 * np.pi
        return ErrAng
    
    return ErrAng
 
def calc_M(chi, a, b):
    
    # INERTIAL MATRIchi
    M11 = chi[0]
    M12 = 0
    M13 = 0
    M14 = b * chi[1]
    M21 = 0
    M22 = chi[2]
    M23 = 0
    M24 = a* chi[3]
    M31 = 0
    M32 = 0
    M33 = chi[4]
    M34 = 0
    M41 = b*chi[5]
    M42 = a* chi[6]
    M43 = 0
    M44 = chi[7]*(a**2+b**2) + chi[8]

    M = np.array([[M11, M12, M13, M14],
                [M21, M22, M23, M24],
                [M31, M32, M33, M34],
                [M41, M42, M43, M44]])
    
    return M

def calc_C(chi, a, b, x):
    w = x[7]

    # CENTRIOLIS MATRIchi
    C11 = chi[9]
    C12 = w*chi[10]
    C13 = 0
    C14 = a * w * chi[11]
    C21 = w*chi[12]
    C22 = chi[13]
    C23 = 0
    C24 = b * w * chi[14]
    C31 = 0
    C32 = 0
    C33 = chi[15]
    C34 = 0
    C41 = a *w* chi[16]
    C42 = b * w * chi[17]
    C43 = 0
    C44 = chi[18]

    C = np.array([[C11, C12, C13, C14],
                [C21, C22, C23, C24],
                [C31, C32, C33, C34],
                [C41, C42, C43, C44]])

    return C

def calc_G():
    # GRAVITATIONAL MATRIchi
    G11 = 0
    G21 = 0
    G31 = 0
    G41 = 0

    G = np.array([[G11],
                [G21],
                [G31],
                [G41]])

def calc_J(x):
    
    psi = x[3]

    J = np.zeros((4, 4))

    J[0, 0] = np.cos(psi)
    J[0, 1] = -np.sin(psi)
    J[1, 0] = np.sin(psi)
    J[1, 1] = np.cos(psi)
    J[2, 2] = 1
    J[3, 3] = 1

    return J