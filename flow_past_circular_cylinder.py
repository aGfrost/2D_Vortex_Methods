import vortex2d as v
import numpy as np
import matplotlib.pyplot as plt

pts = np.linspace(0,2*np.pi, 51)
pts = [complex(1*np.cos(i), 1*np.sin(i)) for i in pts]

panel = v.Linear_panel(pts, net_circulation=0)
sys = v.Setup(linear_panels=[panel], freestreams=[v.Freestream(1)])
gamma_max = 0.1
rho = 1000.0
Re = 1000.0
u = 1.0
d = 2.0
nu = u*d/Re
lamda = abs(
    sys.linear_panels[0].panel_points[0].location -\
    sys.linear_panels[0].panel_points[1].location
)
delta = lamda/np.pi

I = []
#yet to calculate Cd

def color(gamma):
    if gamma > 0:
        return 'b'
    else:
        return 'r'

def simulate(time):
    time_value = 0
    for iter in range(time):
        time_value += 1
        time_stamp = str(time_value)
        for i in range(10):
            sys.linear_panels[0].solve_chorin(sys)
            sys.update_RK2_chorin(0.1)
            sys.reflect(complex(0,0), 1)
            points = sys.linear_panels[0].panel_points
            for i in range(len(points) - 1):
                relative_vector = points[i+1].location - points[i].location
                unit_vector = relative_vector/abs(relative_vector)
                unit_normal = 1j*unit_vector
                panel_circulation = (points[i].gamma + points[i+1].gamma)*\
                lamda/2
                no_of_vortices = int(abs(panel_circulation/gamma_max))

                sign = panel_circulation/abs(panel_circulation)
                if no_of_vortices > 0:
                    new_vortices = []
                    new_location = points[i].location + relative_vector/2 +\
                     delta*unit_normal
                    for k in range(no_of_vortices):
                        new_vortices.append(v.Vortex(
                                new_location,
                                gamma_max*sign,
                                delta=delta))
                    sys.add_vortices(new_vortices)
            sys.diffuse(nu,0.1)

            sys.reflect(complex(0,0),1)
        panel = sys.linear_panels[0]

        I_t = []
        for i in range(panel.n):

            p1 = panel.panel_points[i]
            p2 = panel.panel_points[i+1]
            circ = (p1.gamma + p2.gamma)*lamda/2
            mdpt = (p1.location + p2.location)/2
            I_t.append(rho*circ*np.array([mdpt.imag, -mdpt.real]))

        #smoothing I
        I_t = (I_t + np.roll(I_t,1, axis=0) + np.roll(I_t,-1, axis=0))/3
        res = I_t.sum(axis=0)

        I.append(abs(complex(*res)))

        x = [i.location.real for i in sys.vortices]
        y = [i.location.imag for i in sys.vortices]
        c = [color(i.circulation) for i in sys.vortices]

        fig_v = plt.figure()
        ax = fig_v.add_axes([0.11,0.11,0.75,0.75])
        ax.scatter(x,y, s=1.2, c=c)
        ax.set_xlabel('x')
        ax.set_xlim(left=-3, right=3)
        ax.set_ylim(bottom=-3, top=3)

        ax.set_ylabel('y')
        ax.set_title('Time = ' + time_stamp + 's')
        ax.plot(
            [i.location.real for i in sys.linear_panels[0].panel_points],
            [i.location.imag for i in sys.linear_panels[0].panel_points], 'g'
        )
        fig_v.savefig('vortices_' + time_stamp + '_second.png')

        fig_s = plt.figure()
        ax = fig_s.add_axes([0.11,0.11,0.75,0.75])
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax.set_title('Time = ' + time_stamp + 's')
        x,y = np.meshgrid(np.linspace(-2,3,30), np.linspace(-2,2,30))
        velocity = sys.get_total_velocity_chorin(x,y)
        ax.quiver(x,y,velocity.real, velocity.imag)
        ax.plot(
            [i.location.real for i in sys.linear_panels[0].panel_points],
            [i.location.imag for i in sys.linear_panels[0].panel_points], 'g'
        )
        fig_s.savefig('quiver_plot_' + time_stamp + '_second')

        fig_s_half = plt.figure()
        ax = fig_s_half.add_axes([0.11,0.11,0.75,0.75])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Time = ' + time_stamp + 's')

        x,y = np.meshgrid(np.linspace(0,3,30), np.linspace(-2,2,30))
        velocity = sys.get_total_velocity_chorin(x,y)
        ax.quiver(x,y,velocity.real, velocity.imag)
        fig_s_half.savefig('quiver_plot_half_' + time_stamp + '_second')

        fig_cnt = plt.figure()
        ax = fig_cnt.add_axes([0.11,0.11,0.75,0.75])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Time = ' + time_stamp + 's')
        ax.plot(
            [i.location.real for i in sys.linear_panels[0].panel_points],
            [i.location.imag for i in sys.linear_panels[0].panel_points], 'r'
        )
        x,y = np.meshgrid(np.linspace(-2,3,30), np.linspace(-2,2,30))
        velocity = sys.get_total_velocity_chorin(x,y)
        cnt = ax.contourf(x,y,abs(velocity))
        cbar = plt.colorbar(cnt, ax=ax)
        cbar.set_label('Vmag')
        fig_cnt.savefig('contour_plot_' + time_stamp + '_second')

if __name__ == '__main__':
    simulate(6)
    cd = abs(np.array(I) - np.roll(I,1))[:-1]
    time = range(1,6)
    cd = cd*2/(rho*u*u*d)
    fig = plt.figure()
    ax = fig.add_axes([0.11,0.11,0.75,0.75])
    ax.set_xlabel('time')
    ax.set_ylabel('Cd')
    ax.plot(time, cd)
    fig.savefig('cd_vs_time.png')
