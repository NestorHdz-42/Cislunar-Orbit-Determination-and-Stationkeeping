#Propagate motion for the specified amount of time
import numpy as np
from CR3BP_Propagator._CPIM import CPIM_2
from CR3BP_Propagator._Equations import rot2in, r12, pi1, pi2, n, nondCRTBP_1, phidot

#defining main propagating function for a speficified time interval tf-t0 and initial state x0
#uses Chevyshev Picard Iteration Method (CPIM) to integrate equations of motion up to a certain tolerance
#frame - 'rot' (rotating frame) or 'in' (inertial frame)
#unit - 'nondim' (nondimensional values) or 'dim' (dimensional units in SI units) ###STILL NEED TO VERIFY

#This function uses the Chebyshev Picard Iteration Method (CPIM)as integrator to be able to propagate the
#equations over a larger interval time. However, for the nondimensional equations, even an interval of 1 may be
#too large since time is scaled with the orbital period of the two body system. For this I choose a small enough value and
#set the interval constant basically doing a picewise approach. The interval 'should' be equivalent to a week in the
#nondimensial time and is just calculated from the circular orbit mean motion. The motion will also vary for different
#mass parameter and this interval was tuned for the two-body systems defined in the Equations module,
#so if you have defined another system, keep in mind the interval may still be too large and it may need to be reduced
#in order for it to converge.
def three_b_prop1(t0, tf, x0, frame = 'rot', unit = 'nondim', stm = False):
    H = tf-t0
    N = 20 #degree of chebyshev polynomials
    h = 7*24*3600*n/(40*np.pi)
    I_N = abs(int(H/h))
    if stm:
        phi0 = np.eye(6).reshape((36,))
        x0 = np.hstack((x0, phi0))
    # print(x0.shape)
    t_x = np.zeros((1, len(x0) + 1))
    t_x[0,1:] = x0
    t1 = t0
    t2 = t1 + h
    if H < 0 :
        t1 = t0
        t2 = t0 - h
    if I_N == 0:
        if stm :
            # phi0 = np.eye(6).reshape((36,))
            # x0 = np.hstack((x0,phi0))
            CPI = CPIM_2(phidot, N, (t0, tf), x0)
        elif stm == False:
            CPI = CPIM_2(nondCRTBP_1, N, (t0, tf), x0)

        x_t = CPI[:, 1:4]
        v_t = CPI[:, 4:7]
        if H > 0:

            t_x = np.vstack((t_x, CPI[1:, :]))
            x0 = t_x[-1, 1:]
            t1 = t2
            t2 = t2 + h

        elif H < 0:

            t_x = np.vstack((CPI[:-1, :], t_x))
            x0 = t_x[0, 1:]
            t1 = t2
            t2 = t2 - h
    elif I_N > 0:
        for i in range(I_N):
            if stm:
                # phi0 = np.eye(6).reshape((36,))
                # x0 = np.hstack((x0, phi0))
                CPI = CPIM_2(phidot, N, (t1, t2), x0)
            elif stm == False:
                CPI = CPIM_2(nondCRTBP_1, N, (t1, t2), x0)
            x_t = CPI[:, 1:4]
            v_t = CPI[:, 4:7]
            if H > 0:

                t_x = np.vstack((t_x, CPI[1:,:]))
                x0 = t_x[-1,1:]
                t1 = t2
                t2 = t2 + h

            elif H <0 :

                t_x = np.vstack((CPI[:-1,:], t_x))
                x0 = t_x[0,1:]
                t1 = t2
                t2 = t2 - h
        #CPI = CPIM_2(nondCRTBP_1, N, (t1, tf), x0)
        if stm :
            # phi0 = np.eye(6).reshape((36,))
            # x0 = np.hstack((x0,phi0))
            CPI = CPIM_2(phidot, N, (t1, tf), x0)
        elif stm == False:
            CPI = CPIM_2(nondCRTBP_1, N, (t1, tf), x0)
        x_t = CPI[:, 1:4]
        v_t = CPI[:, 4:7]
        if H > 0:

            t_x = np.vstack((t_x, CPI[1:, :]))

        elif H < 0:

            t_x = np.vstack((CPI[:-1, :], t_x))

    if unit == 'dim':
        t_x[:,0] = t_x[:,0]* 2 * np.pi / n
        t_x[:,1:4] = t_x[:,1:4] * r12
        t_x[:,4:] = t_x[:,4:] * r12 * n / (2 * np.pi)

    if frame == 'in' :
        for k in range(len(t_x[:,0])):
            rv_rot = t_x[k,1:]
            rv_rot[0] -=  pi2*r12
            t = t_x[:,0]
            rv_in = rot2in(rv_rot, n, t[k])
            t_x[k] = rv_in

    return t_x

#function propagating initial state until a specified plane crossing (xy. xz, yz)
#the state ode is defaulted to including the variatonal equations to propagate the state transition matrix
#further work:
    #this could be useful for other trajectory design applications and additional improvements could be implemented
    #if someone finds it useful.
    # Currently this function only works once, if you try passing the final state from one
    #function call into another, the main part of the function will not evaluate. I didn't looked at that in depth
    #since only one call is needed for the differential procedure so it works for this case. I suspect is a simple
    #issue in the conditionals.
    # Another argument could be added to specify a bounded plane or rotated planes . This could be useful for building
    #poincare sections
def prop2planecross(t0, x0, plane, step = 'fwd', ddt = phidot, count = 1):
    N= 20
    h = 7*24*3600*n/(2*np.pi)
    r0 = x0[:3]
    v0 = x0[3:]
    if plane == 'yz':
        index = 0
        plane_error_0 = r0[0]
    elif plane == 'xz':
        index = 1
        plane_error_0 = r0[1]
    elif plane == 'xy':
        index = 2
        plane_error_0 = r0[2]

    if r0[index] == 0:
        plane_error_0 = v0[index]
    i = 0
    if ddt == phidot:
        phi_0 = np.eye(6)
        x0 = np.hstack((x0, np.reshape(phi_0, (36,))))

    T_X = np.zeros((1, len(x0) + 1))
    T_X[0,1:] = x0

    if step == 'bwd':
        h = -h
    # print(h)
    cross = 0
    while abs(r0[index]) > 1e-9 or r0[index] == 0 :
        tf = t0 + h
        #
        while (r0[index] > 0 and plane_error_0 > 0) or (r0[index] < 0 and plane_error_0 < 0) or (r0[index] == 0) :
            # print(r0[index])
            CPI_2 = CPIM_2(ddt, N, (t0,tf), x0)
            if step == 'fwd':
                T_X = np.vstack((T_X, CPI_2[1:,:]))
                x0 = T_X[-1,1:]

            elif step == 'bwd':
                T_X = np.vstack((CPI_2[:-1,:], T_X))
                x0 = T_X[0,1:]

            t0 = tf
            tf = t0 + h

            r0 = x0[:3]
            # print(r0[index])

        cross +=1 #cross < count and
        if r0[0] > 0:
            # print(x0)
            # print(cross)
            plane_error_0 = r0[index]
            continue

        elif cross >= count:
            # print(cross, count)
            # print(x0)
            # delete last interval from the states array
            if step == 'fwd':
                T_X = T_X[:-(N+1),:]
                x0 = T_X[-1,1:]
                t0 = T_X[-1, 0]
            elif step == 'bwd':
                T_X = T_X[(N + 1):, :]
                x0 = T_X[0, 1:]
                t0 = T_X[0,0]

            # redefine initial state at the beginning of the last interval
            r0 = x0[:3]
            h = h / 2

    # print(T_X)


    return T_X
