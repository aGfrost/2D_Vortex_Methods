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

#simulate for 1 second

for i in range(10):
    sys.linear_panels[0].solve_chorin(sys)
    sys.update_RK2_chorin(0.1)
    sys.reflect(complex(0,0), 1)
    points = sys.linear_panels[0].panel_points
    for i in range(len(points) - 1):
        relative_vector = points[i+1].location - points[i].location
        unit_vector = relative_vector/abs(relative_vector)
        unit_normal = 1j*unit_vector
        panel_circulation = (points[i].gamma + points[i+1].gamma)*lamda/2
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

x = [i.location.real for i in sys.vortices]
y = [i.location.imag for i in sys.vortices]
def color(gamma):
    if gamma > 0:
        return 'b'
    else:
        return 'r'
c = [color(i.circulation) for i in sys.vortices]

fig_v = plt.figure()
ax = fig_v.add_axes([0.11,0.11,0.75,0.75])
ax.scatter(x,y, s=1.2, c=c)
ax.set_xlabel('x')
ax.set_xlim(left=-3, right=3)
ax.set_ylim(bottom=-3, top=3)
ax.set_ylabel('y')
ax.set_title('Time = 1s')
ax.plot(
    [i.location.real for i in sys.linear_panels[0].panel_points],
    [i.location.imag for i in sys.linear_panels[0].panel_points], 'g'
)
fig_v.savefig('vortices_1_second.png')

fig_s = plt.figure()
ax = fig_s.add_axes([0.11,0.11,0.75,0.75])
ax.set_xlabel('x')
ax.set_ylabel('y')

ax.set_title('Time = 1s')
x,y = np.meshgrid(np.linspace(-2,3,30), np.linspace(-2,2,30))
velocity = sys.get_total_velocity_chorin(x,y)
ax.quiver(x,y,velocity.real, velocity.imag)
ax.plot(
    [i.location.real for i in sys.linear_panels[0].panel_points],
    [i.location.imag for i in sys.linear_panels[0].panel_points], 'g'
)
fig_s.savefig('quiver_plot_1_second')

fig_s_half = plt.figure()
ax = fig_s_half.add_axes([0.11,0.11,0.75,0.75])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Time = 1s')

x,y = np.meshgrid(np.linspace(0,3,30), np.linspace(-2,2,30))
velocity = sys.get_total_velocity_chorin(x,y)
ax.quiver(x,y,velocity.real, velocity.imag)
fig_s_half.savefig('quiver_plot_half_1_second')

fig_cnt = plt.figure()
ax = fig_cnt.add_axes([0.11,0.11,0.75,0.75])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Time = 1s')

x,y = np.meshgrid(np.linspace(-2,3,30), np.linspace(-2,2,30))
velocity = sys.get_total_velocity_chorin(x,y)
ax.contourf(x,y,abs(velocity))
fig_cnt.savefig('contour_plot_1_second')

## Code is repeated after this.

#simulate for 2 second

for i in range(10):
    sys.linear_panels[0].solve_chorin(sys)
    sys.update_RK2_chorin(0.1)
    sys.reflect(complex(0,0), 1)
    points = sys.linear_panels[0].panel_points
    for i in range(len(points) - 1):
        relative_vector = points[i+1].location - points[i].location
        unit_vector = relative_vector/abs(relative_vector)
        unit_normal = 1j*unit_vector
        panel_circulation = (points[i].gamma + points[i+1].gamma)*lamda/2
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

x = [i.location.real for i in sys.vortices]
y = [i.location.imag for i in sys.vortices]
def color(gamma):
    if gamma > 0:
        return 'b'
    else:
        return 'r'
c = [color(i.circulation) for i in sys.vortices]

fig_v = plt.figure()
ax = fig_v.add_axes([0.11,0.11,0.75,0.75])
ax.scatter(x,y, s=1.2, c=c)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_xlim(left=-3, right=3)
ax.set_ylim(bottom=-3, top=3)
ax.set_title('Time = 2s')
ax.plot(
    [i.location.real for i in sys.linear_panels[0].panel_points],
    [i.location.imag for i in sys.linear_panels[0].panel_points], 'g'
    )
fig_v.savefig('vortices_2_second.png')

fig_s = plt.figure()
ax = fig_s.add_axes([0.11,0.11,0.75,0.75])
ax.set_xlabel('x')
ax.set_ylabel('y')

ax.set_title('Time = 2s')
x,y = np.meshgrid(np.linspace(-2,3,30), np.linspace(-2,2,30))
velocity = sys.get_total_velocity_chorin(x,y)
ax.quiver(x,y,velocity.real, velocity.imag)
ax.plot(
    [i.location.real for i in sys.linear_panels[0].panel_points],
    [i.location.imag for i in sys.linear_panels[0].panel_points], 'g'
    )
fig_s.savefig('quiver_plot_2_second')

fig_s_half = plt.figure()
ax = fig_s_half.add_axes([0.11,0.11,0.75,0.75])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Time = 2s')

x,y = np.meshgrid(np.linspace(0,3,30), np.linspace(-2,2,30))
velocity = sys.get_total_velocity_chorin(x,y)
ax.quiver(x,y,velocity.real, velocity.imag)
fig_s_half.savefig('quiver_plot_half_2_second')

fig_cnt = plt.figure()
ax = fig_cnt.add_axes([0.11,0.11,0.75,0.75])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Time = 2s')

x,y = np.meshgrid(np.linspace(-2,3,30), np.linspace(-2,2,30))
velocity = sys.get_total_velocity_chorin(x,y)
ax.contourf(x,y,abs(velocity))
fig_cnt.savefig('contour_plot_2_second')

#simulate for 3 second

for i in range(10):
    sys.linear_panels[0].solve_chorin(sys)
    sys.update_RK2_chorin(0.1)
    sys.reflect(complex(0,0), 1)
    points = sys.linear_panels[0].panel_points
    for i in range(len(points) - 1):
        relative_vector = points[i+1].location - points[i].location
        unit_vector = relative_vector/abs(relative_vector)
        unit_normal = 1j*unit_vector
        panel_circulation = (points[i].gamma + points[i+1].gamma)*lamda/2
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

x = [i.location.real for i in sys.vortices]
y = [i.location.imag for i in sys.vortices]
def color(gamma):
    if gamma > 0:
        return 'b'
    else:
        return 'r'
c = [color(i.circulation) for i in sys.vortices]

fig_v = plt.figure()
ax = fig_v.add_axes([0.11,0.11,0.75,0.75])
ax.scatter(x,y, s=1.2, c=c)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Time = 3s')
ax.set_xlim(left=-3, right=3)
ax.set_ylim(bottom=-3, top=3)
ax.plot(
    [i.location.real for i in sys.linear_panels[0].panel_points],
    [i.location.imag for i in sys.linear_panels[0].panel_points], 'g'
    )
fig_v.savefig('vortices_3_second.png')

fig_s = plt.figure()
ax = fig_s.add_axes([0.11,0.11,0.75,0.75])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Time = 3s')

x,y = np.meshgrid(np.linspace(-2,3,30), np.linspace(-2,2,30))
velocity = sys.get_total_velocity_chorin(x,y)
ax.quiver(x,y,velocity.real, velocity.imag)
ax.plot(
    [i.location.real for i in sys.linear_panels[0].panel_points],
    [i.location.imag for i in sys.linear_panels[0].panel_points], 'g'
    )
fig_s.savefig('quiver_plot_3_second')

fig_s_half = plt.figure()
ax = fig_s_half.add_axes([0.11,0.11,0.75,0.75])
ax.set_xlabel('x')
ax.set_ylabel('y')

ax.set_title('Time = 3s')
x,y = np.meshgrid(np.linspace(0,3,30), np.linspace(-2,2,30))
velocity = sys.get_total_velocity_chorin(x,y)
ax.quiver(x,y,velocity.real, velocity.imag)
fig_s_half.savefig('quiver_plot_half_3_second')


fig_cnt = plt.figure()
ax = fig_cnt.add_axes([0.11,0.11,0.75,0.75])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Time = 3s')

x,y = np.meshgrid(np.linspace(-2,3,30), np.linspace(-2,2,30))
velocity = sys.get_total_velocity_chorin(x,y)
ax.contourf(x,y,abs(velocity))
fig_cnt.savefig('contour_plot_3_second')
#simulate for 4 second

for i in range(10):
    sys.linear_panels[0].solve_chorin(sys)
    sys.update_RK2_chorin(0.1)
    sys.reflect(complex(0,0), 1)
    points = sys.linear_panels[0].panel_points
    for i in range(len(points) - 1):
        relative_vector = points[i+1].location - points[i].location
        unit_vector = relative_vector/abs(relative_vector)
        unit_normal = 1j*unit_vector
        panel_circulation = (points[i].gamma + points[i+1].gamma)*lamda/2
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

x = [i.location.real for i in sys.vortices]
y = [i.location.imag for i in sys.vortices]
def color(gamma):
    if gamma > 0:
        return 'b'
    else:
        return 'r'
c = [color(i.circulation) for i in sys.vortices]

fig_v = plt.figure()
ax = fig_v.add_axes([0.11,0.11,0.75,0.75])
ax.scatter(x,y, s=1.2, c=c)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_xlim(left=-3, right=3)
ax.set_ylim(bottom=-3, top=3)
ax.set_title('Time = 4s')
ax.plot(
    [i.location.real for i in sys.linear_panels[0].panel_points],
    [i.location.imag for i in sys.linear_panels[0].panel_points], 'g'
    )
fig_v.savefig('vortices_4_second.png')

fig_s = plt.figure()
ax = fig_s.add_axes([0.11,0.11,0.75,0.75])
ax.set_xlabel('x')

ax.set_ylabel('y')
ax.set_title('Time = 4s')
x,y = np.meshgrid(np.linspace(-2,3,30), np.linspace(-2,2,30))
velocity = sys.get_total_velocity_chorin(x,y)
ax.quiver(x,y,velocity.real, velocity.imag)
ax.plot(
    [i.location.real for i in sys.linear_panels[0].panel_points],
    [i.location.imag for i in sys.linear_panels[0].panel_points], 'g'
    )
fig_s.savefig('quiver_plot_4_second')

fig_s_half = plt.figure()
ax = fig_s_half.add_axes([0.11,0.11,0.75,0.75])
ax.set_xlabel('x')

ax.set_ylabel('y')
ax.set_title('Time = 4s')
x,y = np.meshgrid(np.linspace(0,3,30), np.linspace(-2,2,30))
velocity = sys.get_total_velocity_chorin(x,y)
ax.quiver(x,y,velocity.real, velocity.imag)
fig_s_half.savefig('quiver_plot_half_4_second')

fig_cnt = plt.figure()
ax = fig_cnt.add_axes([0.11,0.11,0.75,0.75])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Time = 4s')

x,y = np.meshgrid(np.linspace(-2,3,30), np.linspace(-2,2,30))
velocity = sys.get_total_velocity_chorin(x,y)
ax.contourf(x,y,abs(velocity))
fig_cnt.savefig('contour_plot_4_second')

#simulate for 5 second

for i in range(10):
    sys.linear_panels[0].solve_chorin(sys)
    sys.update_RK2_chorin(0.1)
    sys.reflect(complex(0,0), 1)
    points = sys.linear_panels[0].panel_points
    for i in range(len(points) - 1):
        relative_vector = points[i+1].location - points[i].location
        unit_vector = relative_vector/abs(relative_vector)
        unit_normal = 1j*unit_vector
        panel_circulation = (points[i].gamma + points[i+1].gamma)*lamda/2
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

x = [i.location.real for i in sys.vortices]
y = [i.location.imag for i in sys.vortices]
def color(gamma):
    if gamma > 0:
        return 'b'
    else:
        return 'r'
c = [color(i.circulation) for i in sys.vortices]

fig_v = plt.figure()
ax = fig_v.add_axes([0.11,0.11,0.75,0.75])
ax.scatter(x,y, s=1.2, c=c)
ax.set_xlabel('x')
ax.set_xlim(left=-3, right=3)
ax.set_ylim(bottom=-3, top=3)
ax.set_ylabel('y')
ax.set_title('Time = 5s')
ax.plot(
    [i.location.real for i in sys.linear_panels[0].panel_points],
    [i.location.imag for i in sys.linear_panels[0].panel_points], 'g'
    )
fig_v.savefig('vortices_5_second.png')

fig_s = plt.figure()
ax = fig_s.add_axes([0.11,0.11,0.75,0.75])
ax.set_xlabel('x')
ax.set_ylabel('y')

ax.set_title('Time = 5s')
x,y = np.meshgrid(np.linspace(-2,3,30), np.linspace(-2,2,30))
velocity = sys.get_total_velocity_chorin(x,y)
ax.quiver(x,y,velocity.real, velocity.imag)
ax.plot(
    [i.location.real for i in sys.linear_panels[0].panel_points],
    [i.location.imag for i in sys.linear_panels[0].panel_points], 'g'
    )
fig_s.savefig('quiver_plot_5_second')

fig_s_half = plt.figure()
ax = fig_s_half.add_axes([0.11,0.11,0.75,0.75])
ax.set_xlabel('x')
ax.set_ylabel('y')

ax.set_title('Time = 5s')
x,y = np.meshgrid(np.linspace(0,3,30), np.linspace(-2,2,30))
velocity = sys.get_total_velocity_chorin(x,y)
ax.quiver(x,y,velocity.real, velocity.imag)
fig_s_half.savefig('quiver_plot_half_5_second')


fig_cnt = plt.figure()
ax = fig_cnt.add_axes([0.11,0.11,0.75,0.75])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Time = 5s')

x,y = np.meshgrid(np.linspace(-2,3,30), np.linspace(-2,2,30))
velocity = sys.get_total_velocity_chorin(x,y)
ax.contourf(x,y,abs(velocity))
fig_cnt.savefig('contour_plot_5_second')
