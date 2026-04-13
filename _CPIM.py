import numpy as np

#create Nxlen(x) array of function evaluations at each node
def G(ddrdtt, t, x):
    #print(x[:].shape)
    g = np.zeros(x[:].shape)
    #print(g.shape)
    #print(g.shape)
    for i in range(len(x[:])):
        g[i] = ddrdtt(t, x[i])
    return g

#evaluate n degree chebyshev polynomial at tau
def T(N,tau):
    t_iminus1 = 1 + tau*0
    t_i = tau
    if N == 0:
        t_i = t_iminus1
    for n in range(2, N+1):
        t_iplus1 = 2*tau*t_i - t_iminus1
        t_iminus1 = t_i
        t_i = t_iplus1
    return t_i

#Chevysheb_Picard Iteration Method
def CPIM(ddrdtt, N, h, x_0, step = 'fwd'):
    r0, v0 = x_0
    # GCL nodes
    tau = np.zeros((1, N + 1))
    for m in range(N + 1):
        tau[0][m] = -np.cos(m * np.pi / N)
    t = (h / 2) * tau + h / 2

    if step == 'bwd':
        t = -np.flip(t)
        tau = np.flip(tau,1)

    # R matrix initialization
    r = np.ones((1, N + 1))
    for n in range(1, N + 1):
        r[0][n] = 1 / (2 * n)
    R = np.diag(r[0])

    # S matrix
    s_1 = np.zeros((1, N + 1))  # first row
    s_1[0][0] = 1
    s_1[0][1] = -1 / 2
    for p in range(2, N):
        s_1[0][p] = ((-1) ** (p + 1)) * ((1 / (p - 1)) - (1 / (p + 1)))
    s_1[0][N] = ((-1) ** (N + 1)) * (1 / (N - 1))
    s_N = np.zeros((N, 1))  # last column below first row
    s_N[-2] = -1
    s_s = np.eye(N) + -np.eye(N, k=2)  # square matrix below first row
    S_b = np.concatenate((s_s, s_N), axis=1)
    S = np.concatenate((s_1, S_b), axis=0)

    # T matrix
    T_mat = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(N + 1):
            T_mat[j][i] = T(i, tau[0][j])

    # V matrix
    v = []
    for y in range(N - 1):
        v += [2 / N]
    V = np.diag([1 / N] + v + [1 / N])

    # W matrix
    w = np.ones((1, N + 1))
    w[0][0] = 1 / 2
    W = np.diag(w[0])

    # Cx
    Cx = np.matmul(T_mat, W)

    # Calpha
    C_alpha = np.matmul(R, np.matmul(S, np.matmul(np.transpose(T_mat), V)))

    # initial conditions
    R0 = np.zeros((N + 1, 3))
    V0 = np.zeros((N + 1, 3))

    R0[0] = 2 * r0
    V0[0] = 2 * v0

    # initial guess
    Beta_0 = np.random.rand(N + 1, 6)
    xv_0 = np.matmul(Cx, Beta_0)
    if step == 'fwd':
        xv_0[0] = np.concatenate((r0, v0))
    elif step == "bwd":
        xv_0[-1] = np.concatenate((r0, v0))
        h = -h
    # position only approximation
    C1 = ((h / 2) ** 2) * np.matmul(Cx, np.matmul(C_alpha, np.matmul(Cx, C_alpha)))
    C2 = (h / 2) * np.matmul(Cx, np.matmul(C_alpha, np.matmul(Cx, V0))) + np.matmul(Cx, R0)

    e = 1
    eTol = 1e-16
    # picard iteration
    i = 0
    while abs(e) > eTol:
        g = G(ddrdtt, xv_0) #function evaluation
        #print(g)
        v_1 = (h/2)*np.matmul(Cx,np.matmul(C_alpha, g)) + np.matmul(Cx, V0)
        x_1 = (h/2)*np.matmul(Cx, np.matmul(C_alpha,v_1)) + np.matmul(Cx, R0) #next guess

        x_t = np.concatenate((x_1, v_1), axis=1)
        e = abs(np.linalg.norm(x_t) - np.linalg.norm(xv_0))
        #print(e)
        xv_0 = x_t
        i+=1
    print(i)
    #print(xv_0)

    #print(x_t-xv_0)
    output = np.concatenate((np.transpose(t), x_t), axis = 1)
    return output

#improved CPI method, converges faster and takes any state
#array. It also takes function arguments to pass into the function evaluation
#based on the picard iteration, the benefit is larger time intervals of integration
#but is also harder to identify at which interval it becomes unstable so a
#reasonable interval must be chosen (not too large), otherwise there is a risk
#of it not converging.
def CPIM_2(drdt, N, dt, x_0, farg = None):

    t0, tf = dt
    h = tf-t0
    # GCL nodes
    tau = np.zeros((1, N + 1))
    for m in range(N + 1):
        tau[0][m] = -np.cos(m * np.pi / N)
    t = (h / 2) * tau + (tf+t0) / 2

    if h < 0:
        t = np.flip(t)
        tau = np.flip(tau, 1)

    # R matrix initialization
    r = np.ones((1, N + 1))
    for n in range(1, N + 1):
        r[0][n] = 1 / (2 * n)
    R = np.diag(r[0])

    # S matrix
    s_1 = np.zeros((1, N + 1))  # first row
    s_1[0][0] = 1
    s_1[0][1] = -1 / 2
    for p in range(2, N):
        s_1[0][p] = ((-1) ** (p + 1)) * ((1 / (p - 1)) - (1 / (p + 1)))
    s_1[0][N] = ((-1) ** (N + 1)) * (1 / (N - 1))
    s_N = np.zeros((N, 1))  # last column below first row
    s_N[-2] = -1
    s_s = np.eye(N) + -np.eye(N, k=2)  # square matrix below first row
    S_b = np.concatenate((s_s, s_N), axis=1)
    S = np.concatenate((s_1, S_b), axis=0)

    # T matrix
    T_mat = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(N + 1):
            T_mat[j][i] = T(i, tau[0][j])

    # V matrix
    v = []
    for y in range(N - 1):
        v += [2 / N]
    V = np.diag([1 / N] + v + [1 / N])

    # W matrix
    w = np.ones((1, N + 1))
    w[0][0] = 1 / 2
    W = np.diag(w[0])

    # Cx
    Cx = np.matmul(T_mat, W)

    # Calpha
    C_alpha = np.matmul(R, np.matmul(S, np.matmul(np.transpose(T_mat), V)))

    # initial state
    R0 = np.zeros((N + 1, len(x_0)))

    R0[0] = 2 * x_0

    #initial guess approximated linearly based on the ode function evaluation
    #could it possibly be approximated as higher order polynomial? Could that make it significantly faster to converge?
    #the way this method of integration works it seems that it could be for highly nonlinear equations,
    #where trajectories over even a small interval cannot be approximated linearly.
    x_g = np.zeros((N + 1, len(x_0)))
    if farg == None:
        g_g = drdt(t0, x_0)
        # print(g_g)
    elif type(farg) == tuple:
        g_g = drdt(t0, x_0, farg)
    for j in range(len(g_g)):
        # print(x_0[j], float(g_g[j]*h))
        x_g[:, j] = np.linspace(x_0[j], float(g_g[j] * h), N + 1)

    e = 1
    eTol = 1e-12
    # picard iteration
    i = 0
    while abs(e) > eTol:
        if farg == None:
            g = G(drdt, t, x_g)  # function evaluation
        elif type(farg) == tuple:
            g = G(drdt,t, x_g, farg)
        x_1 = (h / 2) * np.matmul(Cx, np.matmul(C_alpha, g)) + np.matmul(Cx, R0)
        e = abs(np.linalg.norm(x_1) - np.linalg.norm(x_g))
        print(e)
        x_g = x_1
        i += 1
        print('CPI:',i)
        if i > 100:
            print('CPIM_2 - Risk not converging')

    #print below to check how many iterations it takes to converge
    #if takes more than 50 it would be safer to reduce the interval time
    #or reduce the size of the state ode
    # print(i)
    output = np.concatenate((np.transpose(t), x_1), axis=1)
    return output