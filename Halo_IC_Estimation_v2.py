import numpy as np
import matplotlib.pyplot as plt
import CR3BP_Propagator._Equations as sys
from CR3BP_Propagator._Jacobi import jacobi_int
from scipy.integrate import solve_ivp
np.set_printoptions(linewidth= 200)

def RM(seq, theta1=0, theta2=0, theta3=0):
    theta = [theta1, theta2, theta3]
    I = np.eye(3)
    R = np.array([I, I, I])
    for i in np.arange(0,len(seq),1):
        if int(seq[i]) == 1:
            R[i] = np.array([[1, 0, 0],
                             [0, np.cos(np.radians(theta[i])), np.sin(np.radians(theta[i]))],
                             [0, -np.sin(np.radians(theta[i])), np.cos(np.radians(theta[i]))]])
        elif int(seq[i]) == 2:
            R[i] = np.array([[np.cos(np.radians(theta[i])), 0, -np.sin(np.radians(theta[i]))],
                             [0, 1, 0],
                             [np.sin(np.radians(theta[i])), 0, np.cos(np.radians(theta[i]))]])
        elif int(seq[i]) == 3:
            R[i] = np.array([[np.cos(np.radians(theta[i])), np.sin(np.radians(theta[i])), 0],
                             [-np.sin(np.radians(theta[i])), np.cos(np.radians(theta[i])), 0],
                             [0, 0, 1]])

    # DCM = np.matmul(R[2], np.matmul(R[1], R[0]))
    DCM = R[2]@R[1]@R[0]
    return DCM


def Earth_observer( long, lat, t):
    R = 6378*np.array([[1],[0],[0]])/sys.r12
    Of = RM('32', long, -lat).T
    R = Of@R
    p_e = Of@R
    we = 7.272205217e-5
    wm = (2*np.pi)/2.361e6
    wev = we*np.array([0,0,1])
    wmv = wm*np.array([0,0,1])
    P = np.empty((len(t),3))
    P[0] = p_e.T
    Pdot = np.empty((len(t),3))
    C2 = RM('2', 28.58).T #inertial 2 to 1
    for i in np.arange(0,len(t)):
        # print(euler_seq('3',we*t[i], 0, 0))
        # wev = p_e@wev
        # print(wev, p_e[:,0:1])
        # print(P[-1].T)
        C1 = RM('3', np.degrees((we) * t[i])).T #ecef to inertial 2
        C3 = RM('3', np.degrees(wm*t[i])) #inertial 1 to cr3bpf


        rpi = R
        rpi =  C2 @ C1@ rpi
        #
        rpi = C3 @rpi
        wevi = C3@C2@wev.reshape((3,1))
        Pdot[i] = np.cross(wevi[:,0].T-wmv, rpi.T)
        rpi[0,0] -= sys.pi2
        P[i] = rpi.T


    return np.hstack((P,Pdot/1.025))

def range(r1, r2):
    rho = r2 - r1
    # print(r1, r2)
    # print(np.linalg.norm(rho, axis =1).reshape((len(rho),1)))
    return (rho, np.linalg.norm(rho, axis =1).reshape((len(rho),1)))

def range_rate(range, v1, v2):
    rho_dot = v2 - v1
    range_vec, range = range
    vr = np.sum(range_vec*rho_dot, axis = 1).reshape((len(range),1))
    return (rho_dot, vr/range)

def Jac(rho, rhodot, phi):

    r_vec, d = rho
    r_rate_vec, r_rate = rhodot
    drho_drho = r_vec/(np.repeat(d,3, axis = 1))
    drho_drhodot = np.zeros((len(d),3))
    drhodot_drho = (r_rate_vec/np.repeat(d, 3, axis= 1)) - (r_rate/(d**2))*r_vec
    drhodot_drhodot = r_vec/np.repeat(d, 3, axis = 1)
    jac = np.empty((2*len(d),6))
    for i in np.arange(0,len(d),1):
        stm = phi[i,:].reshape(6, 6)
        jac[2*i:2*i+2,:] = np.matmul(np.block([[drho_drho[i,:], drho_drhodot[i,:]],[drhodot_drho[i,:], drhodot_drhodot[i,:]]]), stm)
    return jac

def halo_od(r, rr, W, T, x0c):
    i = 0
    dJ = 1;
    J = 1;
    dx0c = 1
    while abs(dJ) > 1e-12:
        prop = solve_ivp(sys.phidot, [0, T], np.hstack((x0c, np.eye(6).reshape(36,))), 'Radau', t_eval= t)
        OE = prop.y.T
        r_estimate = OE[:,0:3]
        v_estimate = OE[:,3:6]
        range_estimate_vec, range_estimate = range(RM[:, :3] - np.array([[sys.pi2, 0, 0]]), r_estimate)
        range_rate_estimate_vec, range_rate_estimate = range_rate((range_estimate_vec, range_estimate), RM[:, 3:], v_estimate)
        dy = np.empty((2*len(t),1))
        dy[0::2] = range_msd-range_estimate
        dy[1::2] = range_rt_msd-range_rate_estimate
        H = Jac((range_estimate_vec, range_estimate), (range_rate_estimate_vec, range_rate_estimate), OE[:,6:])
        dx0c = np.matmul(np.matmul(np.matmul(np.linalg.inv(np.matmul(np.matmul(H.T, W), H)),H.T), W), dy)
        x0c += dx0c[:,0]
        J = np.hstack((J,0.5*np.matmul(dy.T, dy)[0,0]))
        dJ = J[i+1]-J[i]
        i += 1
        if i > 200:
            print('Did not converge')
            break

    return x0c, OE, J[-1], i, H


X0 = np.array(([0.831701735, 0, 0.1241865, 0, 0.23857984513899108, 0]))
J = jacobi_int(sys.pi2, X0.reshape((1,6)))
T = 1.391852123231276
l1 = 0.836915125819713



#propagate reference orbit and stm/ get true range and range rate measurements
t = np.linspace(0, 2*T, 50)
prop = solve_ivp(sys.phidot, [0, 2*T], np.hstack((X0, np.eye(6).reshape(36,))), 'Radau', t_eval= t)
Halo_orbit = prop.y.T[:,:6]
r = Halo_orbit[:,0:3]
v = Halo_orbit[:,3:]

#measurement position and velocity
# rm = np.zeros((1,3))
# rmdot = np.zeros((1,3))
# rm = np.hstack((rm, rmdot))
RM = Earth_observer(0, 45, (2.361e6 / (2 * np.pi)) * t)
# print(rm)
# plt.plot(rm[:,0], rm[:,1])
# plt.show()
#
range_vec_true, range_true = range(RM[:, :3] - np.array([[sys.pi2, 0, 0]]), r)
range_vec_rt_true, range_rt_true = range_rate((range_vec_true, range_true), RM[:, 3:], v)
range_std = .005/3.850e5
range_rt_std = 5e-3/1025
R = np.array((range_std**2, range_rt_std**2))
W = np.linalg.inv(np.diag(np.tile(R,len(t))))

x0c = np.array((0.83, 0, 0.124, 0, 0.23, 0))
M = 100
sigma_pos = np.empty((len(t), 3, M))
sigma_vel = np.empty((len(t), 3, M))
error = np.empty((len(t), 6, M))
for m in np.arange(0,M,1):
    range_msd = range_true + np.random.normal(0, range_std, (len(t), 1))
    range_rt_msd = range_rt_true + np.random.normal(0, range_rt_std, (len(t), 1))
    x0c, Orbit_estimate, Jmin, i, H = halo_od(range_msd, range_rt_msd, W, 2*T, x0c)
    P0 = np.linalg.inv(np.matmul(np.matmul(H.T,W),H))
    error[:,:,m] = Halo_orbit - Orbit_estimate[:,:6]
    for i in np.arange(0,len(t), 1) :
        stm = Orbit_estimate[i,6:].reshape(6,6)
        P = np.matmul(np.matmul(stm, P0), stm.T)
        sigma_pos[i,:,m] = np.diag(P[:3,:3])
        sigma_vel[i,:,m] = np.diag(P[3:,3:])
    print(m, x0c)
sigma_pos_avg = np.mean(sigma_pos, axis = 2)
sigma_vel_avg = np.mean(sigma_vel, axis = 2)
error_std = np.std(error, axis =2)

# print(sigma_pos_avg.shape)
# print(sigma_vel_avg.shape)
print(error_std.shape)
# plt.figure(1)
# plt.title('Range and Range Rate error from true')
# plt.subplot(211)
# plt.plot(t, range_true- range_estimate)
# plt.ylabel('Range error (ND)')
# plt.grid()
# plt.legend()
# plt.subplot(212)
# plt.plot(t, range_rt_true- range_rate_estimate)
# plt.xlabel('Time (ND)')
# plt.ylabel('Range Rate error (ND)')
# plt.grid()
# plt.legend()
#
plt.figure(1)
plt.suptitle('State component errors from true')
plt.subplot(321)
plt.plot(t, error[:,0,:], color = 'c', linewidth = 0.85)
plt.plot(t, 3*np.sqrt(sigma_pos_avg[:,0]), color = 'k', linestyle = '--')
plt.plot(t, -3*np.sqrt(sigma_pos_avg[:,0]),color = 'k', linestyle = '--', label ='$\sigma_x$')
plt.ylabel('X Error')
plt.grid()
plt.legend()
plt.subplot(323)
plt.plot(t, error[:,1,:], color = 'c', linewidth = 0.85)
plt.plot(t, 3*np.sqrt(sigma_pos_avg[:,1]), color = 'k', linestyle = '--')
plt.plot(t, -3*np.sqrt(sigma_pos_avg[:,1]),color = 'k', linestyle = '--', label ='$\sigma_y$')
plt.ylabel('Y Error')
plt.grid()
plt.legend()
plt.subplot(325)
plt.plot(t, error[:,2,:], color = 'c', linewidth = 0.85)
plt.plot(t, 3*np.sqrt(sigma_pos_avg[:,2]), color = 'k', linestyle = '--')
plt.plot(t, -3*np.sqrt(sigma_pos_avg[:,2]),color = 'k', linestyle = '--', label ='$\sigma_z$')
plt.xlabel('Time (ND)')
plt.ylabel('Z Error')
plt.grid()
plt.legend()

plt.subplot(322)
plt.plot(t, error[:,3,:], color = 'c', linewidth = 0.85)
plt.plot(t, 3*np.sqrt(sigma_vel_avg[:,0]), color = 'k', linestyle = '--')
plt.plot(t, -3*np.sqrt(sigma_vel_avg[:,0]),color = 'k', linestyle = '--', label ='$\sigma_\dot{x}$')
plt.ylabel('$\dot{X}$ Error')
plt.grid()
plt.legend()
plt.subplot(324)
plt.plot(t, error[:,4,:], color = 'c', linewidth = 0.85)
plt.plot(t, 3*np.sqrt(sigma_vel_avg[:,1]), color = 'k', linestyle = '--')
plt.plot(t, -3*np.sqrt(sigma_vel_avg[:,1]),color = 'k', linestyle = '--', label ='$\sigma_\dot{y}$')
plt.ylabel('$\dot{Y}$ Error')
plt.grid()
plt.legend()
plt.subplot(326)
plt.plot(t, error[:,5,:], color = 'c', linewidth = 0.85)
plt.plot(t, 3*np.sqrt(sigma_vel_avg[:,2]), color = 'k', linestyle = '--')
plt.plot(t, -3*np.sqrt(sigma_vel_avg[:,2]),color = 'k', linestyle = '--', label ='$\sigma_\dot{z}$')
plt.xlabel('Time (ND)')
plt.ylabel('$\dot{Z}$ Error')
plt.grid()
plt.legend()

plt.figure(2)
plt.suptitle('State component error std\'s vs average std\'s from true (Monte carlo)')
plt.subplot(321)
plt.plot(t, error_std[:,0], color = 'b', label = 'Error std')
plt.plot(t, np.sqrt(sigma_pos_avg[:,0]),color = 'k', linestyle = '--', label ='$\sigma_x$')
plt.ylabel('X Error')
plt.grid()
plt.legend()
plt.subplot(323)
plt.plot(t, error_std[:,1], color = 'b', label = 'Error std')
plt.plot(t, np.sqrt(sigma_pos_avg[:,1]),color = 'k', linestyle = '--', label ='$\sigma_y$')
plt.ylabel('Y Error')
plt.grid()
plt.legend()
plt.subplot(325)
plt.plot(t, error_std[:,2], color = 'b', label = 'Error std')
plt.plot(t, np.sqrt(sigma_pos_avg[:,2]),color = 'k', linestyle = '--', label ='$\sigma_z$')
plt.xlabel('Time (ND)')
plt.ylabel('Z Error')
plt.grid()
plt.legend()

plt.subplot(322)
plt.plot(t, error_std[:,3], color = 'b', label = 'Error std')
plt.plot(t, np.sqrt(sigma_vel_avg[:,0]),color = 'k', linestyle = '--', label ='$\sigma_\dot{x}$')
plt.ylabel('$\dot{X}$ Error')
plt.grid()
plt.legend()
plt.subplot(324)
plt.plot(t, error_std[:,4], color = 'b', label = 'Error std')
plt.plot(t, np.sqrt(sigma_vel_avg[:,1]),color = 'k', linestyle = '--', label ='$\sigma_\dot{y}$')
plt.ylabel('$\dot{Y}$ Error')
plt.grid()
plt.legend()
plt.subplot(326)
plt.plot(t, error_std[:,5], color = 'b', label = 'Error std')
plt.plot(t, np.sqrt(sigma_vel_avg[:,2]),color = 'k', linestyle = '--', label ='$\sigma_\dot{z}$')
plt.xlabel('Time (ND)')
plt.ylabel('$\dot{Z}$ Error')
plt.grid()
plt.legend()
plt.show()
