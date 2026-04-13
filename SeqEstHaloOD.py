import numpy as np
import matplotlib.pyplot as plt
import CR3BP_Propagator._Equations as sys
from CR3BP_Propagator._Jacobi import jacobi_int
from scipy.integrate import solve_ivp
np.set_printoptions(linewidth= 1000)

class seq_est:
    def __init__(self, X0, tspan):
        # propagate reference orbit and stm/ get true range and range rate measurements
        self.x0 = X0
        self.t = tspan
        self.dt = self.t[1]
        prop = solve_ivp(sys.phidot, [0, tspan[-1]], np.hstack((X0, np.eye(6).reshape(36, ))), max_step = .01, method = 'LSODA', rtol = 1e-12, atol = 1e-12, t_eval=self.t)
        self.Halo_orbit = prop.y.T[:, :6]
        r = self.Halo_orbit[:, 0:3]
        v = self.Halo_orbit[:, 3:]

        # measurement position and velocity
        # rm = np.zeros((1,3))
        # rmdot = np.zeros((1,3))
        # rm = np.hstack((rm, rmdot))
        self.R_obs = Earth_observer(0, 0, (2.361e6 / (2 * np.pi)) * self.t)
        # print(rm)
        # plt.plot(rm[:,0], rm[:,1])
        # plt.show()
        #
        range_vec_true, self.range_true = range_msd(self.R_obs[:, :3] , r)
        range_vec_rt_true, self.range_rt_true = range_rate_msd((range_vec_true, self.range_true), self.R_obs[:, 3:], v)
        range_std = .005 / 3.850e5
        range_rt_std = 5e-3 / 1025
        self.Rk = np.diag((range_std ** 2, range_rt_std ** 2))
        self.t_plot = np.zeros((2*len(self.t),))
        self.t_plot[::2] = self.t
        self.t_plot[1::2] =self.t
        # W = np.linalg.inv(np.diag(np.tile(R, len(t))))

    def ekf(self, x0pos, p0pos, ymeas, r_obs, dt):
        # print(r_obs[:,:3].shape)
    #process noise
        G = np.eye(6,6)
        Q = np.diag([1e-16, 1e-16, 1e-16, 1e-16, 1e-16, 1e-16])
        # Q = 1e-12*np.eye(6,6)
        Q = np.zeros((6,6))
        if dt == 0:
            x1pre = x0pos
            p1pre = p0pos

        else:

            prop = solve_ivp(sys.phidot, [0,dt], np.hstack((x0pos.reshape(6,), np.eye(6,6).reshape(36,))),  'LSODA',rtol = 1e-12, atol = 1e-12)
            # print(prop.y.shape)
            x1pre = prop.y[:6,-1].reshape(6,1)
            stm1 = prop.y[6:,-1].reshape(6,6)
            p1pre = stm1@p0pos@stm1.T + G@Q@G.T
        # print(r_obs[:,:3].shape)
        rng = range_msd(r_obs[:,:3], x1pre[:3,:].T)
        rng_dt = range_rate_msd(rng, r_obs[:,3:], x1pre[3:,:].T)
        # print(rng_dt[1][0])
        h_xt = np.array([[rng[1][0]],[rng_dt[1][0]]])
        # print(h_xt.shape)
        Hk = H(rng,rng_dt)
        Kk = p1pre @ Hk.T @ np.linalg.inv(Hk @ p1pre @ Hk.T + self.Rk)

        x1pos = x1pre + Kk@(ymeas-h_xt)
        # print(x1pos.shape)
        p1pos = (np.eye(6)-Kk@Hk)@p1pre@(np.eye(6)-Kk@Hk).T + Kk@self.Rk@Kk.T

        return x1pre, p1pre, x1pos, p1pos

    def run(self):
        P0 = 1e-12*np.eye(6,6)
        x0 = self.x0.reshape(6,1) + np.sqrt(P0)@np.random.normal(0, 1, (6,1))
        self.xhat = x0

        self.P = np.zeros((2*len(self.t),6,6))
        # print(self.P.shape)
        for k in range(0, len(self.t)):
            y_measured = np.array(([[self.range_true[k]],[self.range_rt_true[k]]])) + self.Rk@np.random.normal(0,1, (2,1))
            # print(y_measured)
            r_o = self.R_obs[k,:].reshape(1,6)
            if k == 0:
                x1pre, p1pre, x1pos, p1pos = self.ekf(x0, P0, y_measured, r_o, k)
            else:
                x1pre, p1pre, x1pos, p1pos = self.ekf(x0, P0, y_measured, r_o, self.dt)
            x0 = x1pos
            P0 = p1pos
            # print(p1pos, x1pos)
            self.xhat = np.hstack((self.xhat, x1pre, x1pos))
            self.P[2*k:2*k+2,:,:] = np.stack((p1pre, p1pos), axis = 0)
        self.xhat = self.xhat[:,1:]
        self.error = self.Halo_orbit-self.xhat.T[1::2,:]
        self.sigma = np.sqrt(np.diagonal(self.P[1::2,:,:], axis1 = 1, axis2 = 2))

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

def range_msd(r1, r2):
    rho = r2 - r1
    # print(r1, r2)
    # print(np.linalg.norm(rho, axis =1))
    return (rho, np.linalg.norm(rho, axis = 1))

def range_rate_msd(range, v1, v2):
    rho_dot = v2 - v1
    range_vec, range = range
    vr = np.sum(range_vec*rho_dot, axis = 1)
    return (rho_dot, vr/range)

def H(rho, rhodot):
    r_vec, d = rho
    r_rate_vec, r_rate = rhodot
    jac = np.empty((2,6))
    jac[0,:3] = r_vec/(np.repeat(d,3, axis = 0)) #drho_drho
    jac[0,3:] = np.zeros((1,3)) #drho_drhodot
    jac[1,:3] = (r_rate_vec/np.repeat(d, 3, axis= 0)) - (r_rate/(d**2))*r_vec # drhodot_drho
    jac[1,3:] = r_vec/np.repeat(d, 3, axis = 0) #drhodot_drhodot
    # print(jac)
    return jac

X0 = np.array(([0.831701735, 0, 0.1241865, 0, 0.23857984513899108, 0]))
J = jacobi_int(sys.pi2, X0.reshape((1,6)))
T = 1.391852123231276
est = seq_est(X0, np.linspace(0,T,100))
M = 100
mc_error = np.zeros((M, len(est.t), 6))
mc_sigma = np.zeros((M, len(est.t), 6))
t = est.t
for m in range(0,M):
    est.run()
    mc_error[m, :, :] = est.error
    # print(est.sigma)
    mc_sigma[m, :, :] = est.sigma
error_avg = np.mean(mc_error, axis = 0)
error_std = np.std(mc_error, axis = 0)
sigma_avg = np.mean(mc_sigma, axis = 0)
sigma_std = np.std(mc_sigma, axis = 0)
error = mc_error
# print(error[:,:,0])
sigma_pos_avg = sigma_avg[:,:3]
sigma_vel_avg = sigma_avg[:,3:]
plt.figure(1)
plt.suptitle('State component errors from true')
plt.subplot(321)
plt.plot(t, error[:,:,0].T, color = 'c', linewidth = 0.85)
plt.plot(t, 3*sigma_pos_avg[:,0], color = 'k', linestyle = '--')
plt.plot(t, -3*sigma_pos_avg[:,0],color = 'k', linestyle = '--', label ='3$\sigma_x$')
plt.ylabel('X Error')
plt.grid()
plt.legend()
plt.subplot(323)
plt.plot(t, error[:,:,1].T, color = 'c', linewidth = 0.85)
plt.plot(t, 3*sigma_pos_avg[:,1], color = 'k', linestyle = '--')
plt.plot(t, -3*sigma_pos_avg[:,1],color = 'k', linestyle = '--', label ='3$\sigma_y$')
plt.ylabel('Y Error')
plt.grid()
plt.legend()
plt.subplot(325)
plt.plot(t, error[:,:,2].T, color = 'c', linewidth = 0.85)
plt.plot(t, 3*sigma_pos_avg[:,2], color = 'k', linestyle = '--')
plt.plot(t, -3*sigma_pos_avg[:,2],color = 'k', linestyle = '--', label ='3$\sigma_z$')
plt.xlabel('Time (ND)')
plt.ylabel('Z Error')
plt.grid()
plt.legend()

plt.subplot(322)
plt.plot(t, error[:,:,3].T, color = 'c', linewidth = 0.85)
plt.plot(t, 3*sigma_vel_avg[:,0], color = 'k', linestyle = '--')
plt.plot(t, -3*sigma_vel_avg[:,0],color = 'k', linestyle = '--', label ='3$\sigma_\dot{x}$')
plt.ylabel('$\dot{X}$ Error')
plt.grid()
plt.legend()
plt.subplot(324)
plt.plot(t, error[:,:,4].T, color = 'c', linewidth = 0.85)
plt.plot(t, 3*sigma_vel_avg[:,1], color = 'k', linestyle = '--')
plt.plot(t, -3*sigma_vel_avg[:,1],color = 'k', linestyle = '--', label ='3$\sigma_\dot{y}$')
plt.ylabel('$\dot{Y}$ Error')
plt.grid()
plt.legend()
plt.subplot(326)
plt.plot(t, error[:,:,5].T, color = 'c', linewidth = 0.85)
plt.plot(t, 3*sigma_vel_avg[:,2], color = 'k', linestyle = '--')
plt.plot(t, -3*sigma_vel_avg[:,2],color = 'k', linestyle = '--', label ='3$\sigma_\dot{z}$')
plt.xlabel('Time (ND)')
plt.ylabel('$\dot{Z}$ Error')
plt.grid()
plt.legend()

plt.figure(2)
plt.suptitle('State component error std\'s vs average std\'s from true (Monte carlo)')
plt.subplot(321)
plt.plot(t, error_std[:,0], color = 'b', label = 'Error std')
plt.plot(t, sigma_pos_avg[:,0],color = 'k', linestyle = '--', label ='$\sigma_x$')
plt.ylabel('X Error')
plt.grid()
plt.legend()
plt.subplot(323)
plt.plot(t, error_std[:,1], color = 'b', label = 'Error std')
plt.plot(t, sigma_pos_avg[:,1],color = 'k', linestyle = '--', label ='$\sigma_y$')
plt.ylabel('Y Error')
plt.grid()
plt.legend()
plt.subplot(325)
plt.plot(t, error_std[:,2], color = 'b', label = 'Error std')
plt.plot(t, sigma_pos_avg[:,2],color = 'k', linestyle = '--', label ='$\sigma_z$')
plt.xlabel('Time (ND)')
plt.ylabel('Z Error')
plt.grid()
plt.legend()

plt.subplot(322)
plt.plot(t, error_std[:,3], color = 'b', label = 'Error std')
plt.plot(t, sigma_vel_avg[:,0],color = 'k', linestyle = '--', label ='$\sigma_\dot{x}$')
plt.ylabel('$\dot{X}$ Error')
plt.grid()
plt.legend()
plt.subplot(324)
plt.plot(t, error_std[:,4], color = 'b', label = 'Error std')
plt.plot(t, sigma_vel_avg[:,1],color = 'k', linestyle = '--', label ='$\sigma_\dot{y}$')
plt.ylabel('$\dot{Y}$ Error')
plt.grid()
plt.legend()
plt.subplot(326)
plt.plot(t, error_std[:,5], color = 'b', label = 'Error std')
plt.plot(t, sigma_vel_avg[:,2],color = 'k', linestyle = '--', label ='$\sigma_\dot{z}$')
plt.xlabel('Time (ND)')
plt.ylabel('$\dot{Z}$ Error')
plt.grid()
plt.legend()
plt.show()
# plt.show()