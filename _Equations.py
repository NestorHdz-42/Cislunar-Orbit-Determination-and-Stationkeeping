import numpy as np
import scipy as sci
#gravitational constant km3/kg-s2
G1 = 6.674e-20

#defining main three body parameters for a few two body systems
def Earth_Moon():
    m1 =  5.974e24 # kg
    m2 =  7.34767309e22# kg
    r12 = 3.844e5  # km
    n = np.sqrt(G1*(m1+m2)/(r12**3)) # rad per sec
    pi1 = m1 / (m1 + m2)
    pi2 = m2 / (m1 + m2)
    # pi2 = 0.012150585
    return n, m1, m2, r12, pi1, pi2
def Sun_Earth():
    m1 = 1.989e30  # kg
    m2 = 5.974e24  # kg
    r12 = 148605215.62  # km
    n = np.sqrt(G1 * (m1 + m2) / (r12 ** 3))
    pi1 = m1 / (m1 + m2)
    pi2 = m2 / (m1 + m2)
    return n, m1, m2, r12, pi1, pi2
def Sun_Jupiter():
    m1 = 1.989e30  # kg
    m2 = 1.898e27 #kg
    r12 = 778.5e6 #km
    n = np.sqrt(G1 * (m1 + m2) / (r12 ** 3))
    pi1 = m1 / (m1 + m2)
    pi2 = m2 / (m1 + m2)
    return n, m1, m2, r12, pi1, pi2

def Mars_Phobos():
    m1 = 6.39e23  # kg
    m2 = 1.060e16 #kg
    r12 = 9377 #km
    n = np.sqrt(G1 * (m1 + m2) / (r12 ** 3))
    pi1 = m1 / (m1 + m2)
    pi2 = m2 / (m1 + m2)
    return n, m1, m2, r12, pi1, pi2

#choosing two body system in which to propagate
Twobodysys = 'Earth_Moon'
#Twobodysys = 'Sun_Earth'
#Twobodysys = 'Sun_Jupiter'
#Twobodysys = 'Mars_Phobos'

if Twobodysys == 'Earth_Moon':
    n, m1, m2, r12, pi1, pi2 = Earth_Moon()
elif Twobodysys == 'Sun_Earth':
    n, mi, m2, r12, pi1, pi2 = Sun_Earth()
elif Twobodysys == 'Sun_Jupiter':
    n, mi, m2, r12, pi1, pi2 = Sun_Jupiter()
elif Twobodysys == 'Mars_Phobos':
    n, mi, m2, r12, pi1, pi2 = Mars_Phobos()

#circular restricted three body problem
def CRTBP(rv):
    x, y, z, xdot, ydot, zdot = rv
    pi1 = m1 / (m1 + m2)
    pi2 = m2 / (m1 + m2)
    nu1 = G1*m1
    nu2 = G1*m2
    r1 = np.linalg.norm([x + pi2*r12, y, z])

    r2 = np.linalg.norm([x - pi1*r12, y, z])

    xddot = 2*n*ydot + (n**2)*x -(nu1/ (r1**3))*(x + pi2*r12) - ( nu2 / (r2**3))*(x-pi1*r12)
    yddot = (n**2)*y - 2*n*xdot - (nu1/(r1**3) + nu2/(r2**3))*y
    zddot = -((nu1/(r1**3))+(nu2/(r2**3)))*z
    return [xddot, yddot, zddot]


#nondimensional CRTBP first order .
def nondCRTBP_1(t, rv):
    x = rv[0]
    y = rv[1]
    z = rv[2]
    xdot = rv[3]
    ydot = rv[4]
    zdot = rv[5]
    pi2 = m2/(m1 + m2)
    sig = np.linalg.norm(np.array((x+pi2, y, z)))
    psi = np.linalg.norm(np.array((x-1+pi2, y, z)))
    vx = xdot
    vy = ydot
    vz = zdot
    vxdot = 2*ydot + x - ((1-pi2)/ (sig**3))*(x + pi2) - (pi2/(psi**3))*(x -1 + pi2)
    vydot = -2*xdot + y - ((1-pi2)/(sig**3) + (pi2/(psi**3)))*y
    vzdot = -(((1-pi2)/(sig**3)) + (pi2/(psi**3)))*z
    return np.array((vx, vy, vz, vxdot, vydot, vzdot))


#converting coordinates to inertial frame
def rot2in(x, t):
    rv = x.copy()

    n_pts = len(t)
    # print(t)
    n=1
    for i in range(0,n_pts):
        # print(np.sin(n*t[i]))
        rotMat = np.array(((np.cos(n*t[i]), -np.sin(n*t[i]), 0, 0, 0, 0),
                      (np.sin(n*t[i]), np.cos(n*t[i]), 0, 0, 0, 0),
                      (0, 0, 1, 0, 0, 0),
                      (-n*np.sin(n*t[i]), -n*np.cos(n*t[i]), 0, np.cos(n*t[i]), -np.sin(n*t[i]), 0),
                      (n*np.cos(n*t[i]), -n*np.sin(n*t[i]), 0, np.sin(n*t[i]), np.cos(n*t[i]), 0),
                      (0, 0, 0, 0, 0, 1)))
        # print(rotMat)
        # print(rv[i,:])
        rv[i,:] = (rotMat@rv[i,:].reshape(6,1)).reshape(6)
        # print(rv[i,:])
    return rv

#orbit-attitude equation of motion
#for attitude: euler parameters (quaternions) are used where q[0] is related to the rotation angle
I = np.array(([6.5534, -2.6317, 0.5305],
              [-2.6317, 6.9654, 1.1624],
              [0.5173, 1.1624, 5.5538]))*1e5 # kg-m2
# S, I, V = np.linalg.svd(I)
I, V = np.linalg.eig(I)
I = np.diag(I)
def ao_eqn(t,X):
    x = X[0]
    y = X[1]
    z = X[2]
    xdot = X[3]
    ydot = X[4]
    zdot = X[5]
    om = X[6:9]
    q = X[9:]/np.linalg.norm(X[9:])
    # q_bar = np.array(([0, -q[2], q[1]],
    #                   [q[2], 0, -q[0]],
    #                   [-q[1], q[0], 0]))
    q_bar = np.array(([-q[1], -q[2], -q[3]],
                      [q[0], -q[3], q[2]],
                      [q[3], q[0], -q[1]],
                      [-q[2], q[1], q[0]]))
    q_cross = np.array(([0, -q[3], q[2]],
                      [q[3], 0, -q[1]],
                      [-q[2], q[1], 0],))
    om_bar = np.array(([0, -om[2], om[1]],
                      [om[2], 0, -om[0]],
                      [-om[1], om[0], 0]))

    # I = np.array(([5, 0, 0],
    #               [0, 8, 0],
    #               [0, 0, 12])) * 1e6  # kg-km2

    # I = np.array(([1.5, -0.1, 0.06],
    #               [-0.1, 3, -0.06],
    #               [0.06, -0.06, 4])) * 1e6  # kg-km2
    # C = np.eye(3,3) - (2/(1+ np.vdot(q,q)))*q_bar + (2/(1+ np.vdot(q,q)))* q_bar @ q_bar
    C = np.eye(3, 3) - 2 * q[0]*q_cross + 2 * q_cross @ q_cross
    # C = np.array(([q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2, 2*(q[1]*q[2]+q[0]*q[3]), 2*(q[1]*q[3]+q[0]*q[2])],
    #               [2*(q[1]*q[2]-q[0]*q[3]), q[0]**2 - q[1]**2 + q[2]**2 - q[3]**2, 2*(q[2]*q[3]+q[0]*q[1])],
    #               [2*(q[1]*q[3]-q[0]*q[2]), 2*(q[2]*q[3]-q[0]*q[1]), q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2]))
    # print(C)
    pi2 = m2 / (m1 + m2)
    sig = np.linalg.norm(np.array((x + pi2, y, z)))
    psi = np.linalg.norm(np.array((x - 1 + pi2, y, z)))
    vx = xdot
    vy = ydot
    vz = zdot
    vxdot = 2 * ydot + x - ((1 - pi2) / (sig ** 3)) * (x + pi2) - (pi2 / (psi ** 3)) * (x - 1 + pi2)
    vydot = -2 * xdot + y - ((1 - pi2) / (sig ** 3) + (pi2 / (psi ** 3))) * y
    vzdot = -(((1 - pi2) / (sig ** 3)) + (pi2 / (psi ** 3))) * z
    x_moon_inertial = rot2in(np.array(([[pi1, 0, 0, 0, 0, 0]])), [t])
    x_earth_inertial = rot2in(np.array(([[-pi2, 0, 0, 0, 0, 0]])), [t])
    x_inertial = rot2in(X[:6].reshape(1,6),[t])
    x_mci = x_inertial - x_moon_inertial
    x_eci = x_inertial - x_earth_inertial
    # print(x_mci[:3])
    rbar_m = x_mci[0,:3]/np.linalg.norm(x_mci[0,:3])
    rbar_m = C @ rbar_m.reshape(3,1)

    rbar_e = x_eci[0,:3]/np.linalg.norm(x_eci[0,:3])
    rbar_e = C @ rbar_e.reshape(3,1)
    # print((np.linalg.norm(x_mci[0,:3])*3.85e5))
    # print(G1*m2)
    l_grav_moon = (n**(-2))*3*G1*m2*1e9*(np.cross(rbar_m.reshape(1,3), (I@rbar_m.reshape(3,1)).T)).T/((np.linalg.norm(x_mci[0,:3])*3.85e8)**3)#kg-m2
    l_grav_earth = (n**(-2))*3*G1*m1*1e9*(np.cross(rbar_e.reshape(1,3), (I@rbar_e.reshape(3,1)).T)).T/((np.linalg.norm(x_eci[0,:3])*3.85e8)**3)#kg-m2
    # l_grav = l_grav_earth+l_grav_moon
    l_grav = l_grav_earth
    # print(l_grav)
    h_bod = I@om.reshape(3,1)
    # h_wh = np.array(([0],[100],[100]))/n
    # h_sc = h_bod + h_wh
    h_sc = h_bod
    # omdot = np.linalg.inv(I)@(-om_bar@I@om.reshape(3,1))
    omdot = np.linalg.inv(I)@(l_grav-om_bar@h_sc)
    # omdot = np.linalg.inv(I)@(-om_bar@h_sc)
    # print(np.linalg.inv(I)@l_grav)
    # print(om_bar@I@om.reshape(3,1))
    # print(l_grav-om_bar@I@om.reshape(3,1))

    # A = 0.5*(np.eye(3,3)+q_bar + q.reshape(3,1)@q.reshape(1,3))

    # qdot = A@om.reshape(3,1)
    qdot = 0.5*q_bar@om.reshape(3,1)
    # print(qdot)
    return np.hstack((vx, vy, vz, vxdot, vydot, vzdot, omdot.flatten(), qdot.flatten()))

def impulse(t):
    if t == 0 :
        d = 1
    elif t > 0:
        d = 0
    return d

def aoctrl_eqn(t,X):
    x = X[0]
    y = X[1]
    z = X[2]
    xdot = X[3]
    ydot = X[4]
    zdot = X[5]
    om = X[6:9]
    q = X[9:13]/np.linalg.norm(X[9:13])
    h_rw = X[13:]

    # q_bar = np.array(([0, -q[2], q[1]],
    #                   [q[2], 0, -q[0]],
    #                   [-q[1], q[0], 0]))
    q_bar = np.array(([-q[1], -q[2], -q[3]],
                      [q[0], -q[3], q[2]],
                      [q[3], q[0], -q[1]],
                      [-q[2], q[1], q[0]]))
    q_cross = np.array(([0, -q[3], q[2]],
                      [q[3], 0, -q[1]],
                      [-q[2], q[1], 0],))
    om_bar = np.array(([0, -om[2], om[1]],
                      [om[2], 0, -om[0]],
                      [-om[1], om[0], 0]))

    # I = np.array(([5, 0, 0],
    #               [0, 8, 0],
    #               [0, 0, 12])) * 1e6  # kg-km2

    # I = np.array(([1.5, -0.1, 0.06],
    #               [-0.1, 3, -0.06],
    #               [0.06, -0.06, 4])) * 1e6  # kg-km2
    # C = np.eye(3,3) - (2/(1+ np.vdot(q,q)))*q_bar + (2/(1+ np.vdot(q,q)))* q_bar @ q_bar
    C = np.eye(3, 3) - 2 * q[0]*q_cross + 2 * q_cross @ q_cross
    # C = np.array(([q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2, 2*(q[1]*q[2]+q[0]*q[3]), 2*(q[1]*q[3]+q[0]*q[2])],
    #               [2*(q[1]*q[2]-q[0]*q[3]), q[0]**2 - q[1]**2 + q[2]**2 - q[3]**2, 2*(q[2]*q[3]+q[0]*q[1])],
    #               [2*(q[1]*q[3]-q[0]*q[2]), 2*(q[2]*q[3]-q[0]*q[1]), q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2]))
    # print(C)
    pi2 = m2 / (m1 + m2)
    sig = np.linalg.norm(np.array((x + pi2, y, z)))
    psi = np.linalg.norm(np.array((x - 1 + pi2, y, z)))
    vx = xdot
    vy = ydot
    vz = zdot
    vxdot = 2 * ydot + x - ((1 - pi2) / (sig ** 3)) * (x + pi2) - (pi2 / (psi ** 3)) * (x - 1 + pi2)
    vydot = -2 * xdot + y - ((1 - pi2) / (sig ** 3) + (pi2 / (psi ** 3))) * y
    vzdot = -(((1 - pi2) / (sig ** 3)) + (pi2 / (psi ** 3))) * z
    x_moon_inertial = rot2in(np.array(([[pi1, 0, 0, 0, 0, 0]])), [t])
    x_inertial = rot2in(X[:6].reshape(1,6),[t])
    x_mci = x_inertial - x_moon_inertial
    # print(x_mci[:3])
    rbar_m = x_mci[0,:3]/np.linalg.norm(x_mci[0,:3])
    rbar_m = C @ rbar_m.reshape(3,1)
    # print((np.linalg.norm(x_mci[0,:3])*3.85e5))
    # print(G1*m2)
    l_grav = (n**(-2))*3*G1*m2*1e9*(np.cross(rbar_m.reshape(1,3), (I@rbar_m.reshape(3,1)).T)).T/((np.linalg.norm(x_mci[0,:3])*3.85e8)**3)#kg-m2
    # print(l_grav)
    h_sc = I@om.reshape(3,1)
    # h_wh = np.array(([0],[100],[100]))/n
    h_tot = h_sc + h_rw.reshape(3,1)
    # h_sc = h_bod
    # omdot = np.linalg.inv(I)@(-om_bar@I@om.reshape(3,1))
    h_rwdot = np.array((-1*h_rw[0], 0, 0))/(n)
    # print(t)
    omdot = np.linalg.inv(I)@(-h_rwdot.reshape(3,1)+l_grav-om_bar@h_tot)
    # omdot = np.linalg.inv(I)@(-om_bar@h_sc)
    # print(np.linalg.inv(I)@l_grav)
    # print(om_bar@I@om.reshape(3,1))
    # print(l_grav-om_bar@I@om.reshape(3,1))

    # A = 0.5*(np.eye(3,3)+q_bar + q.reshape(3,1)@q.reshape(1,3))

    # qdot = A@om.reshape(3,1)
    qdot = 0.5*q_bar@om.reshape(3,1)
    # print(qdot)
    return np.hstack((vx, vy, vz, vxdot, vydot, vzdot, omdot.flatten(), qdot.flatten(), h_rwdot))

#this is the ode used for the differential correction. Propagates both crtbp and state transition matrix
def phidot(t, phi):
    # creating df jacobian
    # print(phi)
    x = phi[:6]
    d_1 = np.linalg.norm(np.array((x[0] + pi2, x[1], x[2])))
    d_2 = np.linalg.norm(np.array((x[0] - pi1, x[1], x[2])))

    u_xx = 1+(pi1/(d_1**5))*(3*((x[0]+pi2)**2))+(pi2/(d_2**5))*(3*((x[0]-pi1)**2))-((pi1/(d_1**3))+(pi2/(d_2**3)))
    u_yy = 1+(pi1/(d_1**5))*(3* (x[1]**2)    )+(pi2/(d_2**5))*(3* (x[1]**2)     )-((pi1/(d_1**3))+(pi2/(d_2**3)))
    u_zz = (pi1 / (d_1 ** 5)) * (3 * (x[2] ** 2)) + (pi2 / (d_2 ** 5)) * (3 * (x[2] ** 2)) - (
                (pi1 / (d_1 ** 3)) + (pi2 / (d_2 ** 3)))

    u_xy = 3*x[1]*     ((pi1*(x[0]+pi2)/(d_1**5))+(pi2*(x[0]-pi1)/(d_2**5)))
    u_xz = 3*x[2]*     ((pi1*(x[0]+pi2)/(d_1**5))+(pi2*(x[0]-pi1)/(d_2**5)))
    u_yz = 3*x[1]*x[2]*((pi1          /(d_1**5))+(pi2           /(d_2**5)))
    zero = np.zeros((3, 3))

    eye = np.eye(3)
    Om = np.zeros((3,3))
    Om[0, 1] = 2
    Om[1, 0] = -2
    U = np.array([[u_xx, u_xy, u_xz],
                  [u_xy, u_yy, u_yz],
                  [u_xz, u_yz, u_zz]])

    Df = np.hstack((np.vstack((zero, U)), np.vstack((eye, Om))))
    phidot = np.matmul(Df, np.reshape(phi[6:],(6,6)))
    phidot = np.reshape(phidot, (36,))
    phidot = np.hstack((nondCRTBP_1(t, x), phidot))
    return phidot

def potential(pi2, x):
    pi1 = 1-pi2
    x = x[:,:3]
    d = np.linalg.norm(np.array((x[:,0], x[:,1])), axis= 0)
    d_1 = np.linalg.norm(np.array((x[:,0] + pi2, x[:,1], x[:,2])), axis= 0)
    d_2 = np.linalg.norm(np.array((pi1-x[:,0], x[:,1], x[:,2])), axis= 0)
    return 0.5*(d**2)+ pi1/d_1 + pi2/d_2
    #return 0.5*(pi1*(d_1**2) + pi2*(d_2**2)) + pi1/d_1 + pi2/d_2
    # return (pi1/d_1.T + pi2/d_2.T + 0.5*(d**2 + pi1*pi2))

def cr3bp_jacobian(x):
    pi1 = 1-pi2
    d_1 = np.linalg.norm(np.array((x[0] + pi2, x[1], x[2])))
    d_2 = np.linalg.norm(np.array((x[0] - pi1, x[1], x[2])))
    u_xx = 1+(pi1/(d_1**5))*(3*((x[0]+pi2)**2))+(pi2/(d_2**5))*(3*((x[0]-pi1)**2))-((pi1/(d_1**3))+(pi2/(d_2**3)))
    u_yy = 1+(pi1/(d_1**5))*(3* (x[1]**2)    )+(pi2/(d_2**5))*(3* (x[1]**2)     )-((pi1/(d_1**3))+(pi2/(d_2**3)))
    u_zz = (pi1 / (d_1 ** 5)) * (3 * (x[2] ** 2)) + (pi2 / (d_2 ** 5)) * (3 * (x[2] ** 2)) - (
                (pi1 / (d_1 ** 3)) + (pi2 / (d_2 ** 3)))

    u_xy = 3*x[1]*     ((pi1*(x[0]+pi2)/(d_1**5))+(pi2*(x[0]-pi1)/(d_2**5)))
    u_xz = 3*x[2]*     ((pi1*(x[0]+pi2)/(d_1**5))+(pi2*(x[0]-pi1)/(d_2**5)))
    u_yz = 3*x[1]*x[2]*((pi1          /(d_1**5))+(pi2           /(d_2**5)))
    zero = np.zeros((2, 2))

    eye = np.eye(2)
    Om = np.zeros((2,2))
    Om[0, 1] = 2
    Om[1, 0] = -2
    U = np.array([[u_xx, u_xy, u_xz],
                  [u_xy, u_yy, u_yz],
                  [u_xz, u_yz, u_zz]])
    U = np.array([[u_xx, u_xy],
                  [u_xy, u_yy]])
    Df = np.hstack((np.vstack((zero, U)), np.vstack((eye, Om))))
    return Df

def cr3bp_jacobian3(x):
    pi1 = 1-pi2
    d_1 = np.linalg.norm(np.array((x[0] + pi2, x[1], x[2])))
    d_2 = np.linalg.norm(np.array((x[0] - pi1, x[1], x[2])))
    # print(d_2)
    u_xx = 1+(pi1/(d_1**5))*(3*((x[0]+pi2)**2))+(pi2/(d_2**5))*(3*((x[0]-pi1)**2))-((pi1/(d_1**3))+(pi2/(d_2**3)))
    u_yy = 1+(pi1/(d_1**5))*(3* (x[1]**2)    )+(pi2/(d_2**5))*(3* (x[1]**2)     )-((pi1/(d_1**3))+(pi2/(d_2**3)))
    u_zz = (pi1 / (d_1 ** 5)) * (3 * (x[2] ** 2)) + (pi2 / (d_2 ** 5)) * (3 * (x[2] ** 2)) - (
                (pi1 / (d_1 ** 3)) + (pi2 / (d_2 ** 3)))

    u_xy = 3*x[1]*     ((pi1*(x[0]+pi2)/(d_1**5))+(pi2*(x[0]-pi1)/(d_2**5)))
    u_xz = 3*x[2]*     ((pi1*(x[0]+pi2)/(d_1**5))+(pi2*(x[0]-pi1)/(d_2**5)))
    u_yz = 3*x[1]*x[2]*((pi1          /(d_1**5))+(pi2           /(d_2**5)))
    zero = np.zeros((3, 3))

    eye = np.eye(3)
    Om = np.zeros((3,3))
    Om[0, 1] = 2
    Om[1, 0] = -2
    U = np.array([[u_xx, u_xy, u_xz],
                  [u_xy, u_yy, u_yz],
                  [u_xz, u_yz, u_zz]])
    Df = np.hstack((np.vstack((zero, U)), np.vstack((eye, Om))))
    return Df