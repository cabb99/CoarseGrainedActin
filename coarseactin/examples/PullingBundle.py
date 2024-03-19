#!/home/cab22/miniconda3/bin/python

#SBATCH --account=commons
#SBATCH --output ./Simulations_nots/SingleABP/slurm-%A_%a.out
#SBATCH --export=All
#SBATCH --partition=commons
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
#SBATCH --array=0-15
#SBATCH --mem=16G

import sys
import pandas as pd
import coarseactin
import numpy as np

from scipy.spatial import cKDTree
import itertools


if __name__ == '__main__':
    ###################################
    # Setting Conditions for simulation#
    ###################################

    parameters = {"epsilon": [100,80,60,40],
                  "aligned": [False],
                  "actinLen": [500],
                  "bundleWidth": [1000],
                  # "repetition":range(3),
                  #"disorder": [0, 0.5],
                  "temperature": [300],
                  "system2D": [False],
                  #"frequency": [1000],
                  "speed": [0.05,0.005,0.0005],
                  "layers": [1,2],
                  # "run_time": [20],
                  # "runSteps":[10000000],
                  "abp": ['FAS', 'CAM', 'CBP', 'AAC', 'AAC2', 'CAM2'],
                  "simulation_platform": ["OpenCL"]}
    test_parameters = {"simulation_platform": "OpenCL",
                       "abp": 'CAM',
                       "layers": 2,
                       "epsilon": 100,
                       #"frequency": 1000,
                       #"speed": 0.05,
                       #'w1': 1,
                       #'w2': 0.1,
                       "speed": 0.1,
                       }

    sjob = coarseactin.SlurmJobArray("Simulations/Pulling/bundle", parameters, test_parameters)
    sjob.print_parameters()
    sjob.print_slurm_variables()
    sjob.write_csv()

    print("name :", sjob.name)

    ##############
    # Parameters #
    ##############
    aligned = sjob["aligned"]
    system2D = sjob["system2D"]
    actinLen = sjob["actinLen"]
    abp = sjob["abp"]
    Sname = sjob.name
    simulation_platform = sjob["simulation_platform"]
    if sjob['abp'] in ['CAM', 'CAM2']:
        camkii_force = 'multigaussian'
    else:
        camkii_force = 'abp'
    layers = sjob['layers']
    frequency = int(50/sjob.speed)
    ###################
    # Build the model #
    ###################
    # Calculate distance between actin filaments:
    bound_actin_template = pd.read_csv("coarseactin/data/CaMKII_bound_with_actin.csv", index_col=0)
    c = bound_actin_template[bound_actin_template['resName'] == abp][['x', 'y', 'z']]
    c.mean(axis=0)
    v = c.mean(axis=0)
    d = 2 * (v['x'] ** 2 + v['y'] ** 2) ** .5
    #layers = int(sjob["bundleWidth"]*np.sqrt(3)/2/d + 1)
    print('n_layers', layers)
    print(f'{abp} size', d)
    if abp == 'FAS':
        colliding_distance = 40#d/4 #TODO: calculate correct distance
    else:
        colliding_distance = d/4 #TODO: calculate correct distance CAM

    # Set the points in the actin network
    
    # nusayba- this is where the cactin + crosslinker being made 

    import random

    full_model = []

    if layers == 1:
        hg = coarseactin.HexGrid(2)
        coords = hg.coords()[:2]
    else:
        hg = coarseactin.HexGrid(layers)
        coords = hg.coords()

    # Make a bundle of 100nm
    if layers > 2:
        coords = coords[(d * (coords ** 2).sum(axis=1) ** .5) < sjob["bundleWidth"] / 2]

    print('Number of actin filaments:', len(coords))

    for c in coords:
        height = (random.random() - 0.5) * sjob["actinLen"] * 28.21600347 #* sjob["disorder"]
        t = np.random.random() * np.pi * 2
        rotation = np.array([[np.cos(t), -np.sin(t), 0.],
                             [np.sin(t), np.cos(t), 0.],
                             [0., 0., 1.]])
        print(c[0], c[1], height)
        full_model += [coarseactin.create_actin(length=sjob["actinLen"],
                                    translation=np.array([5000 + d * c[0], 5000 + d * c[1], 5000 + height]),
                                    rotation=rotation, abp=abp)]

    print('Concatenate chains')
    full_model = coarseactin.Scene.concatenate(full_model)

    # Remove the CaMKII that are not overlapping
    print('Removing Single ABPs')
    mean_abp = full_model[full_model['resName'] == abp].groupby('chain_index')[['x', 'y', 'z']].mean()

    pairs = cKDTree(mean_abp).query_pairs(colliding_distance)
    close_abps = list(set([mean_abp.index[a] for a, b in pairs]+[mean_abp.index[b] for a, b in pairs]))
    close_abps.sort()
    mean_abp_paired = mean_abp.loc[close_abps]

    print('Number of pairs:',len(mean_abp_paired))
    print('Removing Colliding ABPs')
    pairs_colliding = cKDTree(mean_abp_paired).query_pairs(colliding_distance)#TODO: calculate correct distance
    print('Number of collisions:', len(pairs_colliding))
    print(pairs_colliding)
    while len(pairs_colliding) > 0:
        close_abps = list(set([mean_abp_paired.index[random.choice([a, b])] for a, b in pairs_colliding]))
        close_abps.sort()
        sel = mean_abp_paired.index.difference(close_abps)
        mean_abp_paired = mean_abp_paired.loc[sel]
        print('Number of pairs:', len(mean_abp_paired))
        pairs_colliding = cKDTree(mean_abp_paired).query_pairs(colliding_distance)
        print('Number of collisions:', len(pairs_colliding))
        print(pairs_colliding)



    full_model = full_model[full_model['chain_index'].isin(mean_abp_paired.index) | full_model['resName'].isin(['ACT', 'ACD'])]

    print('Rejoining model')
    # Split and rejoin the model so that ABPs are on a different chain
    full_model = coarseactin.Scene(full_model.sort_values(['chainID', 'resSeq', 'name']))
    full_model_actin = coarseactin.Scene(full_model[full_model['resName'].isin(['ACT', 'ACD'])])
    full_model_abps = coarseactin.Scene(full_model[~full_model['resName'].isin(['ACT', 'ACD'])])
    # full_model_abps['chainID'] = 'A'
    full_model = coarseactin.Scene.concatenate([full_model_actin, full_model_abps])

    #Add 2 extra beads for pulling
    actin_chains = full_model[full_model['resName'] == 'ACD']['chain_index'].unique()
    min_residue = (actin_chains[0], full_model[full_model['chain_index']==actin_chains[0]]['res_index'].unique().min())
    max_residues = [(chain,full_model[full_model['chain_index']==chain]['res_index'].unique().max()) for chain in actin_chains[1:]]
    A_index = full_model[(full_model['chain_index']==min_residue[0]) & (full_model['res_index']==min_residue[1])].index
    B_index = np.concatenate([full_model[(full_model['chain_index']==mr[0]) & (full_model['res_index']==mr[1])].index for mr in max_residues])
    A_coord=full_model.loc[A_index][['x', 'y', 'z']].mean()
    B_coord=full_model.loc[B_index][['x', 'y', 'z']].mean()
    AB_particle=full_model.iloc[-2:].copy()
    AB_particle.index +=2
    AB_particle['x'] = [A_coord['x'], B_coord['x']]
    AB_particle['y'] = [A_coord['y'], B_coord['y']]
    AB_particle['z'] = [A_coord['z'], B_coord['z']]
    AB_particle['resSeq'] = [1, 2]
    AB_particle['chainID'] = 'A'
    AB_particle['name'] = 'B'
    AB_particle['type'] = '11'
    AB_particle['element'] = 'B'
    AB_particle['resName'] = 'Bead'
    full_model = coarseactin.Scene.concatenate([full_model, AB_particle])



    full_model.write_cif(f'{Sname}.cif', verbose=True)

    ##############
    # Simulation #
    ##############
    import sys

    sys.path.insert(0, '.')
    try:
        import openmm
        import openmm.app
        from simtk import unit as u
    except ModuleNotFoundError:
        import simtk.openmm as openmm
        import simtk.openmm.app
        import simtk.unit as u
    import time
    from sys import stdout

    time.ctime()

    # Create system
    platform = openmm.Platform.getPlatformByName(simulation_platform)
    s = coarseactin.CoarseActin.from_topology(f'{Sname}.cif', )

    # Add extra bonds for alfa actinin
    # extra_bonds=[]
    # for _, c in s.atom_list[(s.atom_list['residue_name'] == 'AAC') & (s.atom_list['atom_name'] == 'Cb')].groupby(
    #        'chain_index'):
    #    assert len(c.index)==2,'multiple Cbs in actinin'
    #    i0,i1=c.index
    #    extra_bonds+=[['AAC', i0, i1, 0, 10, 0, 350.0, 0, '1Cb-2Cb']]

    # Add extra bonds for CBP
    extra_bonds = []
    name_list = ['C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07', 'C08', 'C09', 'C10', 'C11', 'C12']
    for _, c in s.atom_list[(s.atom_list['residue_name'] == 'CBP')].groupby('chain_index'):
        cm = c[c['atom_name'].isin(name_list)]
        cb = c[c['atom_name'] == 'Cb']
        assert len(cm.index) == len(cb.index)
        assert len(cm.index) == 12
        for i0, i1 in zip(cm.index, cb.index):
            extra_bonds += [['CBP', i0, i1, 0, 10, 0, 31.77, 0, '1Cb-2Cb']]

    extra_bonds = pd.DataFrame(extra_bonds, columns=s.bonds.columns,
                               index=range(s.bonds.index.max() + 1, s.bonds.index.max() + 1 + len(extra_bonds)))
    s.bonds = pd.concat([s.bonds, extra_bonds], ignore_index=True)

    # Check that the bonds correspond with the correct molecule
    s.bonds = s.bonds[(s.bonds['molecule'] == s.atom_list['residue_name'].loc[s.bonds['i']].values) |
                      s.bonds['molecule'].isin(['Actin-ADP', 'ABP', 'CaMKII'])]

    print(s.system.getDefaultPeriodicBoxVectors())
    s.setForces(AlignmentConstraint=aligned, PlaneConstraint=system2D, CaMKII_Force=camkii_force)
    top = openmm.app.PDBxFile(f'{Sname}.cif')
    coord = openmm.app.PDBxFile(f'{Sname}.cif')

    #Add external force
    external_force = openmm.CustomExternalForce("k_spring*(z-Z_A1)^2+k_spring*(y-y_A1)^2+k_spring*(x-x_A1)^2")
    external_force.addGlobalParameter('k_spring', 10)
    A1 = coord.getPositions()[-2]
    Z_A1 = A1[2]
    external_force.addGlobalParameter('x_A1', A1[0])
    external_force.addGlobalParameter('y_A1', A1[1])
    external_force.addGlobalParameter('Z_A1', A1[2])
    external_force.addParticle(int(s.atom_list.index[-2]))

    # Add external force
    external_force2 = openmm.CustomExternalForce("k_spring*(z-Z_A2)^2+k_spring*(y-y_A2)^2+k_spring*(x-x_A2)^2")
    # s.atom_list[(s.atom_list['atom_name'] == 'A2') &
    # (s.atom_list['chain_index'] == 1) &
    # (s.atom_list['residue_index'] == 999)]
    A2 = coord.getPositions()[-1]
    Z_A2 = A2[2]
    external_force2.addGlobalParameter('k_spring', 10)
    external_force2.addGlobalParameter('x_A2', A2[0])
    external_force2.addGlobalParameter('y_A2', A2[1])
    external_force2.addGlobalParameter('Z_A2', A2[2])
    external_force2.addParticle(int(s.atom_list.index[-1]))

    external_force.setForceGroup(10)
    external_force2.setForceGroup(10)
    s.system.addForce(external_force)
    s.system.addForce(external_force2)

    #Add bond between pulling particle and the com of the pulled particles
    centroid_force = openmm.CustomCentroidBondForce(2, "0.5*k*distance(g1,g2)^2");
    centroid_force.addPerBondParameter("k");
    centroid_force.addGroup([int(a) for a in A_index])
    centroid_force.addGroup([int(a) for a in B_index])
    centroid_force.addGroup([int(s.atom_list.index[-2])])  # A
    centroid_force.addGroup([int(s.atom_list.index[-1])])  # B
    centroid_force.addBond([0, 2], [1])  # A
    centroid_force.addBond([1, 3], [1])  # B
    centroid_force.setForceGroup(11)
    s.system.addForce(centroid_force)

    # Set up simulation
    temperature = sjob["temperature"] * u.kelvin
    integrator = openmm.LangevinIntegrator(temperature, .0001 / u.picosecond, 1 * u.picoseconds)
    simulation = openmm.app.Simulation(top.topology, s.system, integrator, platform)
    simulation.context.setPositions(coord.positions)

    # Modify parameters
    simulation.context.setParameter("g_eps", sjob["epsilon"])
    #simulation.context.setParameter("w1", sjob["w1"])
    #simulation.context.setParameter("w2", sjob["w2"])

    #frequency = sjob["frequency"]
    # Add reporters
    simulation.reporters.append(openmm.app.DCDReporter(f'{Sname}.dcd', frequency), )
    simulation.reporters.append(
        openmm.app.StateDataReporter(stdout, frequency, step=True, time=True, potentialEnergy=True, totalEnergy=True,
                                     temperature=True,
                                     separator='\t', ))
    simulation.reporters.append(
        openmm.app.StateDataReporter(f'{Sname}.log', frequency, step=True, time=True, totalEnergy=True,
                                     kineticEnergy=True, potentialEnergy=True, temperature=True))

    # Change simulation parameters
    # simulation.context.setParameter("w1", 5)
    simulation.context.setParameter("g_eps", sjob['epsilon'])

    # Print initial energy
    state = simulation.context.getState(getEnergy=True)
    energy = state.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
    print(f'Initial energy: {energy} KJ/mol')
    energies = {}
    for force in s.system.getForces():
        group = force.getForceGroup()
        state = simulation.context.getState(getEnergy=True, groups=2 ** group)
        energies[force] = state.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
    print(energies)

    # Run
    simulation.minimizeEnergy()
    simulation.context.setVelocitiesToTemperature(temperature * u.kelvin)
    time0 = time.ctime()
    time_0 = time.time()
    #print('Epsilon_values: np.logspace(2,1,1000) each 1000 steps (1 frames')
    #epsilons = np.logspace(2, 1, 200)
    #print(epsilons)
    simulation.context.setParameter("w1", 5)
    simulation.context.setParameter("g_eps", sjob["epsilon"])
    for ext in range(int(1000/sjob['speed'])):
        simulation.context.setParameter("Z_A1", Z_A1 + sjob['speed'] * ext * u.nanometer)
        simulation.context.setParameter("Z_A2", Z_A2 - sjob['speed'] * ext * u.nanometer)
        simulation.step(200)

    #for eps in epsilons:
    #    simulation.context.setParameter("g_eps", eps)
    #    simulation.step(1000)
    #
    #simulation.step(200000)
    #
    #epsilons = np.logspace(1, 2, 200)
    #for eps in epsilons:
    #    simulation.context.setParameter("g_eps", eps)
    #    simulation.step(1000)
    #
    #simulation.step(1000000)

    # Turn off nematic parameter
    # simulation.context.setParameter('kp_alignment',0)
    # simulation.runForClockTime(sjob["run_time"])

    # Save checkpoint
    chk = f'{Sname}.chk'
    simulation.saveCheckpoint(chk)
