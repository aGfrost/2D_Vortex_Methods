# 2D Vortex Methods

As a part of a course project, I impemented some 2D Vortex method schemes in Python. The source code is available here. I’ve implemented the following:

- Continuous and Linear Vortex Panel Method for satisfying boundary conditions on any closed geometry for an arbitary number of panels.
- Random Vortex Method for simulating viscous flows using the Monte Carlo simulation approach.
- Vortex Blob Method for desingularisation of the velocity kernel. Implemented Krasny’s and Chorin’s blob.
- Euler and Runge-Kutta Second Order Integrator for solving the ODEs.
- Vortex Reflection, Vortex Annihilation and No-Slip vortex generation for accurate simulations.

### Here are some simulation results:

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

 
