import numpy as np
from CR3BP_Propagator._Equations import potential, pi1, pi2
import matplotlib.pyplot as plt
def jacobi_int(pi2, X):
    Xdot = X[:,3:]
    V2 = np.linalg.norm(Xdot, axis=1)**2
    return 2*potential(pi2, X) - V2
    # return 0.5*V2-potential(pi2, X)
def zvc(pi2, C):
    pi1 = 1-pi2
    X = np.hstack((np.arange(-1.5, -0.01, 0.001), np.arange(0.01, 1.5, 0.001)))
    Y = np.hstack((np.arange(-1.5, -0.01, 0.001), np.arange(0.01, 1.5, 0.001)))
    Z = np.zeros((1,))
    x, y, z = np.meshgrid(X,  Y, Z)
    d = np.linalg.norm(np.stack((x, y, z)), axis= 0)
    d_1 = np.linalg.norm(np.stack((x + pi2, y, z)), axis= 0)
    d_2 = np.linalg.norm(np.array((x - pi1, y, z)), axis= 0)
    U = -0.5*(d**2 + 2*pi1/d_1 + 2*pi2/d_2 + pi1*pi2)
    zvc = plt.contour(x[:,:,0], y[:,:, 0], U[:,:,0], C)
    plt.clabel(zvc, inline=True, fontsize=5)

def zvs(pi2,C):
    pi1 = 1-pi2
    X = np.hstack((np.arange(-1.5, -0.01, 0.01),np.arange(0.01, 1.5, 0.01)))
    Y = np.hstack((np.arange(-1.5, -0.01, 0.01),np.arange(0.01, 1.5, 0.01)))
    Z = np.hstack((np.arange(-1.5, -0.01, 0.01),np.arange(0.01, 1.5, 0.01)))
    x, y, z = np.meshgrid(X,  Y, Z)
    d = np.linalg.norm(np.stack((x, y, z)), axis= 0)
    d_1 = np.linalg.norm(np.stack((x + pi2, y, z)), axis= 0)
    d_2 = np.linalg.norm(np.array((x - pi1, y, z)), axis= 0)
    U = -(d**2 + 2*pi1/d_1 + 2*pi2/d_2 + pi1*pi2)
    datapts = np.empty((1,3))
    print(datapts)
    for i in range(159,160):#int(len(U)/2)):
        #print(z[:,i,:])
        # print(x[:, :, 0].shape)
        zv = plt.contour(x[:,:,i], y[:,:,i], U[:,:,i], C)
        if len(zv.allsegs[0]) >0:
            print(zv.allsegs[0])
            data = np.vstack(zv.allsegs[0][0])
            print(data)
            plt.scatter(data[:,0], data[:,1])
            zi = z[i,i,i]*np.ones((len(data),1))
            data = np.hstack((data, zi))
            datapts = np.vstack((datapts, data))
        plt.clf()
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.scatter(datapts[1:,0], datapts[1:,1], datapts[1:,2].reshape(len(datapts),1))
    print(datapts)
