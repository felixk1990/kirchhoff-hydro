# @Author:  Felix Kramer
# @Date:   2021-06-03T11:02:57+02:00
# @Email:  kramer@mpi-cbg.de
# @Project: go-with-the-flow
# @Last modified by:    Felix Kramer
# @Last modified time: 2021-08-30T23:39:30+02:00
# @License: MIT

import sys
import networkx as nx
import numpy as np
import scipy.linalg as lina
import random as rd
from hailhydro.flux_init import *
from hailhydro.flow_random import *
from hailhydro.flux_overflow import *

class flux_random(overflow, flow_reroute, object):

    def __init__(self, circuit):
        super(flux_random, self).__init__(circuit)

    def calc_flow(self, *args):

        graph_matrices = self.get_broken_links_asarray(*args)
        flow_observables = list(map(self.calc_flows_mapping, graph_matrices))

        return flow_observables

    def calc_transport_observables(self, idx, conduct, flow_observables):

        # calc ensemble averages
        graph_matrices = self.get_broken_links_asarray(idx, conduct)
        R_powers = calc_random_radii(idx, conduct)
        p_sq = np.power([fo[2] for fo in flow_observables], 2)

        PHI = list(map(self.calc_noisy_absorption, R_powers[1], flow_observables))
        SHEAR = np.multiply(dV_sq, R_powers[1])

        avg_shear_sq = np.sum(np.multiply(dV_sq, R_sq), axis=0)/float(self.num_iteration)
        avg_PHI = np.mean(PHI, axis = 0)

        return shear, phi

    def calc_noisy_absorption(self, R_sq, flow_observables):

        self.update_transport_matrix(R_sq, flow_observables)

        c, B_new = self.solve_absorbing_boundary()

        return self.calc_absorption(R_sq)

    def update_transport_matrix(self, R_sq, flow_observables):

        # set peclet number and internal flow state
        self.circuit.edge['flow_rate'] = flow_observables[0]
        ref_var = self.circuit.scale['length']/self.circuit.scale['diffusion']

        V = self.calc_velocity_from_flowrate(self.circuit.edge['flow_rate'], R_sq)
        self.circuit.edge['peclet'] = self.calc_peclet(V, ref_var)
        A = self.calc_diff_flux(R_sq, 1./ref_var)

        x, z, e_up_sinh_x, e_down_sinh_x, coth_x, idx_pack = self.compute_flux_pars()

        f1 = np.multiply(z, A)
        f2 = np.multiply(A, np.multiply(x, coth_x))*0.5

        f3 = np.multiply(np.multiply(A, x), e_up_sinh_x)*0.5
        f4 = np.multiply(np.multiply(A, x), e_down_sinh_x)*0.5

        # set up concentration_matrix
        self.B_eff = np.zeros((self.N, self.N))

        for i, n in enumerate(self.circuit.list_graph_nodes):

            self.B_eff[i, i] = np.sum( np.add(np.multiply(self.B[i, :], f1), np.multiply(np.absolute(self.B[i, :]), f2)))
            self.B_eff[i, self.dict_in[n]] = -f3[self.dict_node_in[n]]
            self.B_eff[i, self.dict_out[n]] = -f4[self.dict_node_out[n]]
