# import matplotlib.pyplot as plt
import numpy as np
import getpass, time
from math import pi
from h5py import File
from argparse import ArgumentParser
import os, os.path

from qiskit import IBMQ, QuantumCircuit, ClassicalRegister, QuantumRegister, execute, Aer, BasicAer
from qiskit.providers.aer import noise
from qiskit.tools.monitor import job_monitor, backend_monitor, backend_overview
from qiskit.quantum_info.analyzation.average import average_data
from qiskit.providers.ibmq import least_busy
from qiskit.providers.exceptions import JobError, JobTimeoutError
from qiskit.compiler import transpile, assemble



def do_job_on_simulator(real_backend , circuits:  list):

    gate_times = [
        ('u1', None, 0), ('u2', None, 100), ('u3', None, 200),
        ('cx', [1, 0], 678), ('cx', [1, 2], 547), ('cx', [2, 3], 721),
        ('cx', [4, 3], 733), ('cx', [4, 10], 721), ('cx', [5, 4], 800),
        ('cx', [5, 6], 800), ('cx', [5, 9], 895), ('cx', [6, 8], 895),
        ('cx', [7, 8], 640), ('cx', [9, 8], 895), ('cx', [9, 10], 800),
        ('cx', [11, 10], 721), ('cx', [11, 3], 634), ('cx', [12, 2], 773),
        ('cx', [13, 1], 2286), ('cx', [13, 12], 1504), ('cx', [], 800)
    ]
    properties = real_backend.properties()
    coupling_map = real_backend.configuration().coupling_map
    noise_model = noise.device.basic_device_noise_model(properties, gate_times=gate_times)
    basis_gates = noise_model.basis_gates
    simulator = Aer.get_backend('qasm_simulator')
    job = execute(circuits, simulator,
                  noise_model=noise_model,
                  coupling_map=coupling_map,
                  basis_gates=basis_gates)
    return job

def make_rotation(circuit: QuantumCircuit, registers: QuantumRegister, angles: list):

    """
    Rotates and entangle qubits in the circuit
.
    :param circuit: a QuantumCircuit object comprising two qubits
    :param registers: list of registers involved in the circuit
    :param angles: list of tuples (theta, lambda, phi) with rotation angles

    :return:
    """


    circuit.u3(*(angles[0]), registers[0])



    circuit.u3(*(angles[1]), registers[1])



    circuit.cx(registers[1], registers[0])


def get_energy(exchange, bj, result1, circuits: list) -> float:


    """
    Calculate Heisenberg energy

    :param exchange: exchange constant



    :param result: a job object storing the results of quantum calculation


    :param bj: B/J


    :param circuits: list containing XX YY and ZZ circuits



    :param codes:



    :return:



    """

    codes = {'00': 1, '01': -1, '10': -1, '11': 1}
    res = exchange/4 * (average_data(result1.get_counts(circuits[0]), codes) + average_data(result1.get_counts(circuits[1]), codes) + average_data(result1.get_counts(circuits[2]), codes)) + bj * exchange * get_magnetization(result1, [circuits[2]])

    return res


def get_fitness(result, circuits: list, codes1: dict, codes2: dict, codes: dict) -> float:
    """
    Calculate fitness

    :param exchange: exchange constant
    :param result: a job object storing the results of quantum calculation
    :param circuits: list containing XX YY and ZZ circuits
    :param codes:
    :return:
    """
    res =  abs(average_data(result.get_counts(circuits[0]), codes1))/2\
           +abs(average_data(result.get_counts(circuits[0]), codes2))/2\
           +abs(average_data(result.get_counts(circuits[1]), codes1))/2\
           +abs(average_data(result.get_counts(circuits[1]), codes2))/2\
           +abs(average_data(result.get_counts(circuits[2]), codes1))/2\
           +abs(average_data(result.get_counts(circuits[2]), codes2))/2\
           +abs(average_data(result.get_counts(circuits[0]), codes)+1)\
           +abs(average_data(result.get_counts(circuits[1]), codes)+1)\
           +abs(average_data(result.get_counts(circuits[2]), codes)+1)\
           +abs(average_data(result.get_counts(circuits[3]), codes))\
           +abs(average_data(result.get_counts(circuits[4]), codes))\
           +abs(average_data(result.get_counts(circuits[5]), codes))\
           +abs(average_data(result.get_counts(circuits[6]), codes))\
           +abs(average_data(result.get_counts(circuits[7]), codes))\
           +abs(average_data(result.get_counts(circuits[8]), codes))

    return res

def get_magnetization(result, circuits: list) -> float:

    observable_first = {'00': 1, '01': -1, '10': 1, '11': -1}

    observable_second = {'00': 1, '01': 1, '10': -1, '11': -1}

    res = (average_data(result.get_counts(circuits[0]), observable_first) + average_data(result.get_counts(circuits[0]), observable_second)) / 2

    return res

def flush_angles(theta, lada, phi, prefix=''):



    """


    Write current angles to hdf5 file.

    :param theta: theta rotation angle


    :param lada: lambda rotation angle


    :param phi: phi rotation angle


    :param prefix: prefix for output file


    :return: Nothing


    """


    with File(prefix + 'angles.h5', 'w') as f:


        f['/angles/theta'] = theta


        f['/angles/labmda'] = lada


        f['/angles/phi'] = phi


def read_angles(fn, num_genome=None):

    """

    Read angles from an hdf5 file.

    :param fn: path to hdf5 file


    :param num_genome: number of noble genomes in current caluclation fo check (optional)


    :return: tuple of noble angles


    """

    with File(fn, 'r') as f:


        theta = f['/angles/theta'][:]


        lada = f['/angles/labmda'][:]


        phi = f['/angles/phi'][:]





    if num_genome is not None:


        if theta.shape[0] != num_genome or lada.shape[0] != num_genome or phi.shape[0] != num_genome:


            raise ValueError('Wrong shape of input data! Check the number of genomes!')





    return (theta, lada, phi)


def select_genomes(energy: list, num_noble_genome: int, theta: np.array, lada: np.array, phi: np.array) -> tuple:


    """

    Select noble genomes corresponding to minimum energies.

    :param energy: list of energies from calculation


    :param num_genome: number of noble genomes in current caluclation


    :param theta: theta rotation angles


    :param lada: lambda rotation angles


    :param phi: phi rotation angles


    :return: list of noble angles


    """

    depth_loc = theta.shape[1]


    noble_theta = np.ndarray((num_noble_genome, depth_loc, 2), dtype=np.float32)


    noble_lada = np.ndarray((num_noble_genome, depth_loc, 2), dtype=np.float32)


    noble_phi = np.ndarray((num_noble_genome, depth_loc, 2), dtype=np.float32)


    min_vals = energy.copy()

    min_vals.sort()
    print(min_vals)
    # min_vals = min_vals[::-1]

    min_vals = min_vals[:num_noble_genome]
    print(min_vals)



    idx = 0

    numbers=[]
    ideal_angle_n = 0

    for n, e in enumerate(energy):
        if e in min_vals:




            noble_theta[idx,:,:] = theta[n,:,:]

            noble_lada[idx,:,:] = lada[n,:,:]

            noble_phi[idx,:,:] = phi[n,:,:]

            numbers.append(n)

            if e == min_vals[0]:
                ideal_angle_n = n

            idx += 1

            if idx == num_noble_genome:

                break


    print('igenome - fitness')
    for idx in numbers:
        print(idx, energy[idx])
    return (noble_theta, noble_lada, noble_phi, ideal_angle_n)


def evolve_genomes(noble_theta, noble_lada, noble_phi, angle_variation):

    # theta = np.random.random_sample((num_genome, depth, 2)) * pi
    # phi = np.random.random_sample((num_genome, depth, 2)) * 2 * pi
    # lada = np.random.random_sample((num_genome, depth, 2)) * 2 * pi

    for igenome in range(num_noble_genome):

        theta[igenome,:,:]=noble_theta[igenome,:,:]
        phi[igenome,:,:]=noble_phi[igenome,:,:]
        lada[igenome,:,:]=noble_lada[igenome,:,:]

    ran_theta=np.random.randint(3,size=(num_genome,depth,2))

    ran_phi=np.random.randint(3,size=(num_genome,depth,2))

    ran_lada=np.random.randint(3,size=(num_genome,depth,2))

    for igenome in range(num_noble_genome,2*num_noble_genome):

        theta[igenome,:,:]= ( 1-( 1-ran_theta[igenome-num_noble_genome,:,:] )*angle_variation ) * noble_theta[igenome-num_noble_genome,:,:]
        phi[igenome,:,:]= (1-(1-ran_phi[igenome-num_noble_genome,:,:])*angle_variation) * noble_phi[igenome-num_noble_genome,:,:]
        lada[igenome,:,:]= (1-(1-ran_lada[igenome-num_noble_genome,:,:])*angle_variation) * noble_lada[igenome-num_noble_genome,:,:]

    ran_theta=np.random.randint(2,size=(num_genome,depth,2))
    ran_phi=np.random.randint(2,size=(num_genome,depth,2))
    ran_lada=np.random.randint(2,size=(num_genome,depth,2))

    for igenome in range(2*num_noble_genome,3*num_noble_genome):

        theta[igenome,:,:]= ran_theta[igenome-2*num_noble_genome,:,:]*theta[igenome-2*num_noble_genome,:,:]+(1-ran_theta[igenome,:,:])*theta[igenome-2*num_noble_genome+1,:,:]
        phi[igenome,:,:]=ran_phi[igenome-2*num_noble_genome,:,:]*phi[igenome-2*num_noble_genome,:,:]+(1-ran_phi[igenome,:,:])*phi[igenome-2*num_noble_genome+1,:,:]
        lada[igenome,:,:]=ran_lada[igenome-2*num_noble_genome,:,:]*lada[igenome-2*num_noble_genome,:,:]+(1-ran_lada[igenome,:,:])*lada[igenome-2*num_noble_genome+1,:,:]

    ran_theta=np.random.randint(2,size=(num_genome,depth,2))

    ran_phi=np.random.randint(2,size=(num_genome,depth,2))

    ran_lada=np.random.randint(2,size=(num_genome,depth,2))

    for igenome in range(3*num_noble_genome,4*num_noble_genome):

        theta[igenome,:,:]=ran_theta[igenome-3*num_noble_genome+1,:,:]*theta[igenome-3*num_noble_genome,:,:]+(1-ran_theta[igenome,:,:])*theta[igenome-3*num_noble_genome+2,:,:]
        phi[igenome,:,:]=ran_phi[igenome-3*num_noble_genome+1,:,:]*phi[igenome-3*num_noble_genome,:,:]+(1-ran_phi[igenome,:,:])*phi[igenome-3*num_noble_genome+2,:,:]
        lada[igenome,:,:]=ran_lada[igenome-3*num_noble_genome+1,:,:]*lada[igenome-3*num_noble_genome,:,:]+(1-ran_lada[igenome,:,:])*lada[igenome-3*num_noble_genome+2,:,:]


    for igenome in range(4*num_noble_genome,5*num_noble_genome):

        theta[igenome,:,:] = np.random.random_sample(2) * pi
        phi[igenome,:,:] = np.random.random_sample(2) * 2 * pi
        lada[igenome,:,:] = np.random.random_sample(2) * 2 * pi

        # theta[igenome, :, :] = ran_theta[igenome - 3 * num_noble_genome + 1, :, :] * theta[
        #                                                                              igenome - 3 * num_noble_genome, :,
        #                                                                              :] + (
        #                                    1 - ran_theta[igenome, :, :]) * theta[igenome - 3 * num_noble_genome + 2, :,
        #                                                                    :]
        # phi[igenome, :, :] = ran_phi[igenome - 3 * num_noble_genome + 1, :, :] * phi[igenome - 3 * num_noble_genome, :,
        #                                                                          :] + (
        #                                  1 - ran_phi[igenome, :, :]) * phi[igenome - 3 * num_noble_genome + 2, :, :]
        # lada[igenome, :, :] = ran_lada[igenome - 3 * num_noble_genome + 1, :, :] * lada[igenome - 3 * num_noble_genome,
        #                                                                            :, :] + (
        #                                   1 - ran_lada[igenome, :, :]) * lada[igenome - 3 * num_noble_genome + 2, :, :]


    return [theta, lada, phi]


p = ArgumentParser()



p.add_argument('--prefix', default='', help='Prefix for all output files.')



p.add_argument('--load-angles', action='store_true', help='Save angles during calculation.')


p = p.parse_args()


if p.prefix:

 p.prefix += '_'

provider = IBMQ.enable_account('ed1f7070919a8ce0469e69c1cb5b5dc1e114879caada8d0ce25d6e28e91b40c90209146d85eec5118e2584667b02e9662802f0ffaa2494c72375a4906e129fdf')
print('Account loaded')

#name_backend='local_qasm_simulator'
# name_backend='ibmq_16_melbourne'
#name_backend='ibmqx4'

iteration = 10
num_qubits =2
Jexch=1
device_shots = 2048
depth=2
num_genome = 50
num_noble_genome = 10
num_iter = 25
angle_mutation = 0.1

theta=np.random.random_sample((num_genome,depth,num_qubits))*pi
phi=np.random.random_sample((num_genome,depth,num_qubits))*2*pi
lada=np.random.random_sample((num_genome,depth,num_qubits))*2*pi

folder_name = str(iteration) + "-num_qubits=" + str(num_qubits) + ",Jexch=" + str(Jexch) + \
              ",device_shots=" + str(device_shots) + ",depth=" + str(depth) + ",num_genome=" + str(num_genome) + \
              ",num_iter=" + str(num_iter) +  ",angle_mutation=" + str(angle_mutation)

if not os.path.exists(folder_name):
    try:
        os.mkdir(folder_name)
    except OSError:
        print("Creation of the directory failed")


folder_name += "/"
if p.load_angles:

    noble_angles = read_angles(folder_name + 'angles.h5', num_noble_genome)

    # print(noble_angles)

    theta, lada, phi = evolve_genomes(*noble_angles, angle_mutation)





result_path = folder_name + 'final_result.dat'
result_days = open(result_path, 'w')

# backend = least_busy(provider.backends(filters=lambda x: not x.configuration().simulator))
backend = provider.get_backend('ibmq_16_melbourne')
# backend = provider.get_backend('ibmq_5_yorktown')
backend_monitor(backend)

for bj_step in range(5, 21):

    pres = 0.005
    prev_energy = 10
    bj_step = bj_step / 10
    energy_path = folder_name + str(bj_step) + '-energy.dat'
    energy_days = open(energy_path, 'w')
    correlator_path = folder_name + str(bj_step) +  '-Z_correlator.dat'
    correlator_days = open(correlator_path, 'w')

    for i in range(num_iter):

        print('BJ ' + str(bj_step) + ', iteration numer ' + str(i))
        circuits = []

        energy_Heis = np.zeros((num_genome), np.float32)
        fitness = np.zeros((num_genome), np.float32)
        magnetization = np.zeros((num_genome), np.float32)
        corXX = np.zeros((num_genome), np.float32)
        corYY = np.zeros((num_genome), np.float32)
        corZZ = np.zeros((num_genome), np.float32)
        # corXY = np.zeros((num_genome), np.float32)
        # corXZ = np.zeros((num_genome), np.float32)
        # corYZ = np.zeros((num_genome), np.float32)



        for igenome in range(num_genome):

            # Creating registers
            q = QuantumRegister(num_qubits)
            c = ClassicalRegister(num_qubits)
            # quantum circuit to make an entangled bell state
            singlet = QuantumCircuit(q, c)

            for idepth in range(depth):

                lada[igenome,idepth,0] = 0
                lada[igenome, idepth, 1] = 0

                make_rotation(singlet,q, [ (theta[igenome,idepth,0],phi[igenome,idepth,0],lada[igenome,idepth,0]),
                                          (theta[igenome,idepth,1],phi[igenome,idepth,1],lada[igenome,idepth,1]) ])




            # quantum circuit to measure q in the standard basis
            measureZZ = QuantumCircuit(q, c)
            measureZZ.measure(q[0], c[0])
            measureZZ.measure(q[1], c[1])
            # singletZZ = singlet+measureZZ

            # quantum circuit to measure q in the standard basis
            measureYY = QuantumCircuit(q, c)
            measureYY.sdg(q[0])
            measureYY.sdg(q[1])
            measureYY.h(q[0])
            measureYY.h(q[1])
            measureYY.measure(q[0], c[0])
            measureYY.measure(q[1], c[1])
            # singletYY = singlet+measureYY

            # quantum circuit to measure q in the superposition basis
            measureXX = QuantumCircuit(q, c)
            measureXX.h(q[0])
            measureXX.h(q[1])
            measureXX.measure(q[0], c[0])
            measureXX.measure(q[1], c[1])
            # singletXX = singlet+measureXX


            # # quantum circuit to measure ZX
            # measureZX = QuantumCircuit(q, c)
            # measureZX.h(q[0])
            # measureZX.measure(q[0], c[0])
            # measureZX.measure(q[1], c[1])
            # singletZX = singlet+measureZX
            #
            # # quantum circuit to measure XZ
            # measureXZ = QuantumCircuit(q, c)
            # measureXZ.h(q[1])
            # measureXZ.measure(q[0], c[0])
            # measureXZ.measure(q[1], c[1])
            # singletXZ = singlet+measureXZ
            #
            # # quantum circuit to measure q in the standard basis
            # measureXY = QuantumCircuit(q, c)
            # measureXY.sdg(q[1])
            # measureXY.h(q[0])
            # measureXY.h(q[1])
            # measureXY.measure(q[0], c[0])
            # measureXY.measure(q[1], c[1])
            # singletXY = singlet+measureXY
            #
            # # quantum circuit to measure q in the standard basis
            # measureYX = QuantumCircuit(q, c)
            # measureYX.sdg(q[0])
            # measureYX.h(q[0])
            # measureYX.h(q[1])
            # measureYX.measure(q[0], c[0])
            # measureYX.measure(q[1], c[1])
            # singletYX = singlet+measureYX
            #
            # # quantum circuit to measure q in the standard basis
            # measureYZ = QuantumCircuit(q, c)
            # measureYZ.sdg(q[0])
            # measureYZ.h(q[0])
            # measureYZ.measure(q[0], c[0])
            # measureYZ.measure(q[1], c[1])
            # singletYZ = singlet+measureYZ
            #
            # # quantum circuit to measure q in the standard basis
            # measureZY = QuantumCircuit(q, c)
            # measureZY.sdg(q[1])
            # measureZY.h(q[1])
            # measureZY.measure(q[0], c[0])
            # measureZY.measure(q[1], c[1])
            # singletZY = singlet+measureZY


            exec('singletZZ_%d = singlet+measureZZ' % igenome)
            exec('singletYY_%d = singlet+measureYY' % igenome)
            exec('singletXX_%d = singlet+measureXX' % igenome)

            exec('circuits.extend([singletXX_%d,singletYY_%d,singletZZ_%d])' % (igenome, igenome, igenome))

        result=0
        no_error = True
        while no_error:
            try:
                # job = execute(circuits, BasicAer.get_backend('qasm_simulator'), shots=device_shots)
                # job = do_job_on_simulator(backend, circuits)

                job = execute(circuits, backend = backend, shots=device_shots)
                job_monitor(job)
                result = job.result()
                no_error = False
            except JobError:
                print(JobError)
                no_error = True

        print(type(result))
        observable_first = {'00': 1, '01': -1, '10': 1, '11': -1}
        observable_second = {'00': 1, '01': 1, '10': -1, '11': -1}
        observable_correlated = {'00': 1, '01': -1, '10': -1, '11': 1}

        for igenome in range(num_genome):
            exec(
                'energy_Heis[igenome] = get_energy(Jexch, bj_step, result, [singletXX_%d, singletYY_%d, singletZZ_%d])' % (
                igenome, igenome, igenome))
            fitness[igenome] = 0
            exec('magnetization[igenome] = get_magnetization(result,[singletZZ_%d])' % igenome)
            exec('corXX[igenome] = average_data(result.get_counts(singletXX_%d), observable_correlated)' % igenome)
            exec('corYY[igenome] = average_data(result.get_counts(singletYY_%d), observable_correlated)' % igenome)
            exec('corZZ[igenome] = average_data(result.get_counts(singletZZ_%d), observable_correlated)' % igenome)


        energy_for_file = energy_Heis.copy()
        energy_for_file.sort()
        energy_days.write(str(i))
        for i_energy in energy_for_file:
            energy_days.write(' ' + str(i_energy))
        energy_days.write('\n')
        energy_days.flush()

        nobles = select_genomes(energy_Heis, num_noble_genome, theta, lada, phi)
        ideal_angle_number = nobles[3]
        nobles = nobles[:3]

        correlator_days.write(str(i) + ' ' + str(energy_Heis[ideal_angle_number]) + ' ' + str(magnetization[ideal_angle_number]) + ' ' + str(fitness[ideal_angle_number]) + ' ' +
                              str(corXX[ideal_angle_number]) + ' ' + str(corYY[ideal_angle_number]) + ' ' + str(corZZ[ideal_angle_number]) + ' angles:')
        for pair in theta[ideal_angle_number, :, :]:
            correlator_days.write('{0} {1} '.format(*pair))

        for pair in phi[ideal_angle_number, :, :]:
            correlator_days.write('{0} {1} '.format(*pair))

        for pair in lada[ideal_angle_number, :, :]:
            correlator_days.write('{0} {1} '.format(*pair))
        correlator_days.write('\n')
        correlator_days.flush()


        if (abs(prev_energy - energy_Heis[ideal_angle_number]) < pres) or (i == num_iter - 1):
            result_days.write(str(bj_step) + ' ' + str(energy_Heis[ideal_angle_number]) + ' ' + str(magnetization[ideal_angle_number]) +  ' ' + str(backend) + ' angles:')
            for pair in theta[ideal_angle_number,:,:]:
                result_days.write('{0} {1} '.format(*pair))

            for pair in phi[ideal_angle_number,:,:]:
                result_days.write('{0} {1} '.format(*pair))

            for pair in lada[ideal_angle_number,:,:]:
                result_days.write('{0} {1}'.format(*pair))

            result_days.write('\n')
            result_days.flush()
            break

        else:
            prev_energy = energy_Heis[ideal_angle_number]

        # if i == num_iter - 1:
        #     result_days.write(str(bj_step) + ' ' + str(energy_Heis[ideal_angle_number]) + ' ' + str(magnetization[ideal_angle_number]) +  ' ' + str(backend) + ' angles:')
        #     for pair in theta[ideal_angle_number,:,:]:
        #         result_days.write('{0} {1} '.format(*pair))
        #
        #     for pair in phi[ideal_angle_number,:,:]:
        #         result_days.write('{0} {1} '.format(*pair))
        #
        #     for pair in lada[ideal_angle_number,:,:]:
        #         result_days.write('{0} {1}'.format(*pair))
        #
        #     result_days.write('\n')
        #     result_days.flush()

        flush_angles(*nobles, folder_name)

        [theta, lada, phi] = evolve_genomes(nobles[0], nobles[1], nobles[2], angle_mutation)


    energy_days.close()
    correlator_days.close()

result_days.close()
print('end of VQE loop')



