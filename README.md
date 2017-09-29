# 2D Vortex Methods

As a part of a course project, I impemented some 2D Vortex method schemes in Python. The source code is available here. I’ve implemented the following:

- Continuous and Linear Vortex Panel Method for satisfying boundary conditions on any closed geometry for an arbitary number of panels.
- Random Vortex Method for simulating viscous flows using the Monte Carlo simulation approach.
- Vortex Blob Method for desingularisation of the velocity kernel. Implemented Krasny’s and Chorin’s blob.
- Euler and Runge-Kutta Second Order Integrator for solving the ODEs.
- Vortex Reflection, Vortex Annihilation and No-Slip vortex generation for accurate simulations.

### Here are some simulation results:
----
#### Motion of 3 vortices placed at the vertices of an isoceles triangle, simulated for 25 seconds. 

```python
from vortex2d import *
sys = Setup(
  vortices=[
    Vortex(complex(-1,0),1, delta=0.3), 
    Vortex(complex(0,1),1, delta=0.3), 
    Vortex(complex(1,0),1, delta=0.3)
   ]
  )
for i in range(2500):
    sys.update_RK2_chorin(0.1)
fig = plt.figure(figsize=(5,5))
ax = fig.add_axes([0.11,0.11,0.75,0.75])
for vortex in sys.vortices:
    ax.plot([i.real for i in vortex.path], [i.imag for i in vortex.path])
```
![alt text](https://github.com/deeptavker/2D_Vortex_Methods/blob/master/Images/vortices_3.png)

----
#### Flow past circular cylinder with a reynolds number of 1000. The cylinder is of unit radius and is subject to a unit velocity freestream. Linear Panels are used to ensure no penetration and RVM is used for simulating viscosity. The red and blue particles represent vortices with negative and positive circulations respectively. The vortex shedding can be seen in the picture below. 

The code can be found [here](https://github.com/deeptavker/2D_Vortex_Methods/blob/master/scripts/flow_past_circular_cylinder.py).

![alt text](https://github.com/deeptavker/2D_Vortex_Methods/blob/master/Images/cylinder_45.png)

----
#### Vortex sheet roll-up : Given an initial vorticity distribution in 2D, we compute the evolution of this vorticity distribution as discretized by a set of point vortices. We use Krasny's blob for desingularisation of the velocity kernel. 

The code is available [here](https://github.com/deeptavker/2D_Vortex_Methods/blob/master/scripts/vortex_sheet.py)

![alt text](https://github.com/deeptavker/2D_Vortex_Methods/blob/master/Images/vortex_rollup.png)

Reference : Robert Krasny, Computation of vortex sheet roll-up in the Trefftz plane, Journal of Fluid mechanics, volume 184, pp
123-155, 1987.

----
#### Flow past NACA 2412 Airfoil M=2.0% P=40.0% T=12.0%

```python
# a is taken from a coordinte generator
pts = [complex(*i) for i in zip(a[::2], a[1::2])]
pnl = Linear_panel(pts)
sys = Setup(freestreams=[Freestream(complex(1,-0.5))], linear_panels=[pnl])
pnl.solve(sys)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
x,y = np.meshgrid(np.linspace(-0.4,1.4,100), np.linspace(-0.4,0.4,100))
v = sys.get_velocity(x,y)
ax.streamplot(x,y,v.real,v.imag)
ax.plot([i.location.real for i in pnl.panel_points], [i.location.imag for i in pnl.panel_points]
```
![alt_text](https://github.com/deeptavker/2D_Vortex_Methods/blob/master/Images/NACA2412.png)

