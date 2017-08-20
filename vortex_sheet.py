from vortex2d import *
import matplotlib.pyplot as plt

@np.vectorize
def gamma(y):
    return (4*y)/(1-(4*y*y))**0.5

# PART 1 : POINT VORTICES

def point_vortices(number, time_step):
    '''
    Simulate the flow upto 100*time_step time with #number of vortices
    '''
    points = np.linspace(-0.5,0.5,number+1) + 1.0/(2*number)
    circulations = gamma(points[:-1])*(1.0/number)
    coordinates = [complex(i,j) for i,j in zip(points[:-1],np.zeros(number,))]
    figs1 = []
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    partb = Setup(vortices=[Vortex(*i) for i in zip(coordinates,circulations)])
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    x = [i.location.real for i in partb.vortices]
    y = [i.location.imag for i in partb.vortices]
    ax.plot(x,y)
    plt.xlim((-2,2))
    plt.ylim((-2,2))
    figs1.append(fig)
    for i in range(4):
        for j in range(25):
            partb.update_RK2(time_step)
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        x = [i.location.real for i in partb.vortices]
        y = [i.location.imag for i in partb.vortices]
        ax.plot(x,y)
        plt.xlim((-2,2))
        plt.ylim((-2,2))
        figs1.append(fig)
    return figs1


# PART 2 : SAME DISCRETISATION BUT USING KRASNY BLOB
def krasny_blob_linear(number, time_step, delta):
    points = np.linspace(-0.5,0.5,number+1) + 1.0/(2*number)
    circulations = gamma(points[:-1])*(1.0/number)
    coordinates = [complex(i,j) for i,j in zip(points[:-1],np.zeros(number,))]
    figs2 = []
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    partb = Setup(vortices=[Vortex(*i) for i in zip(coordinates,circulations)])
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    x = [i.location.real for i in partb.vortices]
    y = [i.location.imag for i in partb.vortices]
    ax.plot(x,y)
    plt.xlim((-2,2))
    plt.ylim((-2,2))
    figs2.append(fig)
    for i in range(4):
        for j in range(25):
            partb.update_RK2_krasny(time_step,delta)
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        x = [i.location.real for i in partb.vortices]
        y = [i.location.imag for i in partb.vortices]
        ax.plot(x,y)
        plt.xlim((-2,2))
        plt.ylim((-2,2))
        figs2.append(fig)
    return figs2

# PART 3 : SINUSODIAL DISCRETISATION ALONG WITH KRASNY BLOB
def krasny_blob_sine(number, time_step, delta):
    points = np.sin(np.linspace(-cmath.pi/2,cmath.pi/2,number+1))*0.5
    new_points = []
    circulations = []
    for i in range(len(points)-1):
        new_points.append((points[i+1] + points[i])/2)
        circulations.append(gamma((points[i+1] + points[i])/2)*(points[i+1] - points[i]))

    coordinates = [complex(i,j) for i,j in zip(new_points,np.zeros(number,))]
    partb = Setup(vortices=[Vortex(*i) for i in zip(coordinates,circulations)])
    figs3 = []
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    x = [i.location.real for i in partb.vortices]
    y = [i.location.imag for i in partb.vortices]
    ax.plot(x,y)
    plt.xlim((-2,2))
    plt.ylim((-2,2))
    figs3.append(fig)
    for i in range(4):
        for j in range(25):
            partb.update_RK2_krasny(time_step,delta)
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        x = [i.location.real for i in partb.vortices]
        y = [i.location.imag for i in partb.vortices]
        ax.plot(x,y)
        plt.xlim((-2,2))
        plt.ylim((-2,2))
        figs3.append(fig)
    return figs3
