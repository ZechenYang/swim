"""
Active Matter Model H
A script to study active matter

Pass it a .cfg file to specify parameters for the particular solve.
To run using 4 processes, you would use:
    $ mpiexec -n 4 python3 mri.py mri_params.cfg
"""


"""
Dedalus script for 2D incompressible hydrodynamics with moving immersed boundary.

This script uses a Fourier basis in both y directions.

This script can be ran serially or in parallel, and uses the built-in analysis
framework to save data snapshots in HDF5 files.  The `merge.py` script in this
folder can be used to merge distributed analysis sets from parallel runs,
and the `plot_2d_series.py` script can be used to plot the snapshots.

To run using 4 processes (e.g), you could use:
    $ mpiexec -n 4 python3 swimming_sheet.py


"""

import os
import time
import configparser
from configparser import ConfigParser
from pathlib import Path
import numpy as np
import sys
import dedalus.public as de
from dedalus.tools  import post
from dedalus.extras import flow_tools

from mpi4py import MPI

# CW = MPI.COMM_WORLD
import logging

logger = logging.getLogger(__name__)

import body_v1 as bdy

comm = MPI.COMM_WORLD
rank = comm.rank


# Parses .cfg filename passed to script
config_file = Path(sys.argv[-1])

# Parse .cfg file to set global parameters for script
config = ConfigParser()
config.read(str(config_file))

logger.info('Running model_H.py with the following parameters:')
logger.info(config.items('parameters'))

params = config['parameters']
# Simulation
Lx = params.getfloat('Lx') # box size in x
Ly = params.getfloat('Ly') # box size in y

# FLuid
vis = params.getfloat('vis') # viscosity
γ = params.getfloat('γ') # coupling parameter for immersed boundary
hw = params.getfloat('half_width') # half-width of sheet

# Swimmer
b = params.getfloat('b') # amplitude of sheet (b in Taylor 1951 eq 1)
sigma = params.getfloat('sigma') # swimming frequency (sigma in Taylor 1951 eq 1)
delta = params.getfloat('delta') # tanh width for mask;

k = 4*np.pi/Ly # wavenumber of sheet frequency (k in Taylor 1951 eq 1)

dt = params.getfloat('dt') # timestep

# Initial body parameters
x0,U0 = 0,0
y0,V0 = 0,0

# Create bases and domain
x_basis = de.Fourier('x',384, interval=(-Lx, Lx), dealias=3/2)
y_basis = de.Fourier('y',384, interval=(-Ly, Ly), dealias=3/2)
domain  = de.Domain([x_basis, y_basis], grid_dtype=np.float64)

# Setup mask function
x,y = domain.grids(scales=domain.dealias)
K = domain.new_field()
U = domain.new_field()
V = domain.new_field()
K.set_scales(domain.dealias,keep_data=False)
U.set_scales(domain.dealias,keep_data=False)
V.set_scales(domain.dealias,keep_data=False)
K['g'], U['g'], V['g'] =  bdy.sheet(x, y, k, sigma, 0, delta, hw, b, (x0, y0))

# 2D Incompressible hydrodynamics
problem = de.IVP(domain, variables=['p','u','v','ωz'])

problem.parameters['vis']   = vis
problem.parameters['γ']   = γ
problem.parameters['K']   = K
problem.parameters['U']   = U
problem.parameters['V']   = V
problem.add_equation("dt(u) + vis*dy(ωz) + dx(p) =  ωz*v -γ*K*(u-U)")
problem.add_equation("dt(v) - vis*dx(ωz) + dy(p) = -ωz*u -γ*K*(v-V)")
problem.add_equation("ωz + dy(u) - dx(v) = 0")
problem.add_equation("dx(u) + dy(v) = 0",condition="(nx != 0) or (ny != 0)")
problem.add_equation("p = 0",condition="(nx == 0) and (ny == 0)")

# Build solver
solver = problem.build_solver(de.timesteppers.RK222)
logger.info('Solver built')

# Initial conditions
ωz = solver.state['ωz']
u = solver.state['u']

# Integration parameters
t_wave = 2*np.pi/sigma
solver.stop_sim_time = 2*t_wave #np.inf
solver.stop_wall_time = 10*60*60.
solver.stop_iteration = np.inf

# Output setting
basedir = Path('output_data')
outdir = "output_" + config_file.stem
data_dir = basedir/outdir
if domain.dist.comm.rank == 0:
    if not data_dir.exists():
        data_dir.mkdir(parents=True)

# Analysis
snapshots = solver.evaluator.add_file_handler(os.path.join(data_dir,'snapshots'), iter=20, max_writes=50)
snapshots.add_task("p")
snapshots.add_task("u")
snapshots.add_task("v")
snapshots.add_task("ωz")
snapshots.add_task("K")
snapshots.add_task("U")
snapshots.add_task("V")

# Runtime monitoring properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=20)
flow.add_property("abs(ωz)", name='q')

analysis_tasks = []
analysis_tasks.append(snapshots)

# Tasks for force computation
force = flow_tools.GlobalFlowProperty(solver, cadence=1)
force.add_property("K*(v-V)", name='Fy')
force.add_property("K*(u-U)", name='Fx')
#force.add_property("-y*K*(u-U)+x*K*(v-V)", name='T0')

# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        solver.step(dt)
        Fy_net = γ*force.volume_average('Fy')
        Fx_net = γ*force.volume_average('Fx')
        #τ0 = γ*force.volume_average('T0') + y0*F0 - x0*G0
        #F0,G0,τ0 = F0/μ, G0/μ - gravity, τ0/(μ*I)
        x0 = x0 + U0*dt
        y0 = y0 + V0*dt
        U0 = U0 + Fx_net*dt
        V0 = V0 + Fy_net*dt       
        K['g'], U['g'], V['g'] =  bdy.sheet(x, y, k, sigma, solver.sim_time, delta, hw, b)
        if (solver.iteration-1) % 20 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('Max ωz = %f' %flow.max('q'))

except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))

logger.info('beginning join operation')
for task in analysis_tasks:
    logger.info(task.base_path)
    post.merge_analysis(task.base_path)




