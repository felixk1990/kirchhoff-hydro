# @Author:  Felix Kramer
# @Date:   2021-06-03T11:02:33+02:00
# @Email:  kramer@mpi-cbg.de
# @Project: go-with-the-flow
# @Last modified by:    Felix Kramer
# @Last modified time: 2021-09-08T20:18:02+02:00
# @License: MIT

import numpy as np
import scipy.linalg as lina
import random as rd
import networkx as nx
from hailhydro.flow_init import Flow
from dataclasses import dataclass, field
rand = {'mode': 'default', 'noise': 0.}
reroute = {'p_broken': 0, 'num_iter': 100}


@dataclass
class FlowRandom(Flow):

    default = dict(repr=False, init=True)
    flow_setting: dict = field(default_factory=dict(), **default)

    def __post_init__(self):

        self.init_flow()
        self.init_random()

    def init_random(self):

        try:
            if self.flow_setting['mode'] == 'default':
                self.set_effective_source_matrix(self.flow_setting['noise'])

            elif self.flow_setting['mode'] == 'root':
                self.mu_sq = self.flow_setting['mu_sq']
                self.var = self.flow_setting['var']
                self.set_multi_source_matrix(self.mu_sq, self.var)
        except:
            raise Exception('Warning flow landscape not set!')

    # setup_random_fluctuations
    def set_root_source_matrix(self, mean, variance):

        N = len(self.circuit.list_graph_nodes)

        self.matrix_mu = np.identity()
        for n in range(N):
            for m in range(N):
                h = 0.
                if n == 0 and m == 0:
                    h += (N-1)
                elif n == m and n != 0:
                    h += 1.
                elif n == 0 and m != 0:
                    h -= 1.
                elif m == 0 and n != 0:
                    h -= 1.

                self.matrix_mu[n, m] = h

        self.matrix_mu = self.matrix_mu*mean

        self.matrix_var = np.identity(N)
        for n in range(N):

            for m in range(N):
                h = 0.
                if n == 0 and m == 0:
                    h += (1-N)*(1-N)
                elif n != 0 and m != 0:
                    h += 1.
                elif n == 0 and m != 0:
                    h += (1-N)
                elif m == 0 and n != 0:
                    h += (1-N)

                self.matrix_var[n, m] = h

        self.matrix_var = self.matrix_var*variance

    # setup_random_fluctuations_multisink
    def set_effective_source_matrix(self, noise):

        self.noise = noise
        num_n = len(self.circuit.list_graph_nodes)
        x = np.where(self.circuit.nodes['source'] > 0)[0]
        # idx = np.where(self.circuit.nodes['source'] < 0)[0]
        # N = len(idx)
        N = num_n - len(x)
        M = len(x)

        U = np.zeros((num_n, num_n))
        V = np.zeros((num_n, num_n))

        m_sq = float(M*M)
        NM = num_n*num_n/float(m_sq)
        Nm = (N/m_sq)+2./M

        for i in range(num_n):
            for j in range(num_n)[i:]:
                delta = 0.
                sum_delta = 0.
                sum_delta_sq = 0.

                if i == j:
                    delta = 1.

                if (i in x):
                    sum_delta = 1.

                if (j in x):
                    sum_delta = 1.

                if (i in x) and (j in x):
                    sum_delta_sq = 1.
                    sum_delta = 2.

                U[i, j] = (m_sq - num_n*sum_delta + NM*sum_delta_sq)

                v1 = (Nm + delta)*sum_delta_sq
                v2 = (1.+M*delta)*sum_delta
                v3 = m_sq*delta
                V[i, j] = (v1 - v2 + v3)

                U[j, i] = U[i, j]
                V[j, i] = V[i, j]

        self.Z = np.add(U, np.multiply(self.noise, V))

    # calc_sq_flow
    def calc_sq_flow_effective(self, conduct):

        OP = np.dot(self.B, np.dot(np.diag(conduct), self.BT))
        inverse = lina.pinv(OP)
        D = np.dot(self.BT, inverse)
        DT = np.transpose(D)

        A = np.dot(np.dot(D, self.Z), DT)
        dV_sq = np.diag(A)
        F_sq = np.multiply(np.multiply(conduct, conduct), dV_sq)

        return dV_sq, F_sq

    # calc_sq_flow_random
    def calc_sq_flow_root(self, conduct):

        OP = np.dot(np.dot(self.B, conduct), self.BT)
        inverse = lina.pinv(OP)
        D = np.dot(self.BT, inverse)
        DT = np.transpose(D)

        var_matrix = np.dot(np.dot(D, self.matrix_mu), DT)
        mean_matrix = np.dot(np.dot(D, self.matrix_var), DT)

        var_flow = np.diag(var_matrix)
        mean_flow = np.diag(mean_matrix)

        dV_sq = np.add(var_flow, mean_flow)
        F_sq = np.multiply(np.multiply(conduct, conduct), dV_sq)

        return dV_sq, F_sq

@dataclass
class FlowReroute(Flow):

    default = dict(repr=False, init=True)

    flow_setting: dict = field(default_factory=dict, **default)

    def __post_init__(self):

        self.init_flow()
        self.num_iteration = self.flow_setting['num_iter']
        self.percentage_broken = self.flow_setting['p_broken']
        self.initialize_broken_link()

    def initialize_broken_link(self):

        broken_sets = []
        self.num_sets = 50000
        self.AUX = nx.Graph(self.circuit.G)
        for i in range(self.num_sets):
            cond, idx = self.generate_coherent_closure()
            if cond:
                broken_sets.append(idx)

        self.broken_sets = broken_sets

        assert(len(self.broken_sets) != 0)

    def generate_coherent_closure_deterministic(self, H, x):

        idx = rd.sample(range(len(self.circuit.list_graph_edges)), x)
        for e in idx:
            H.remove_edge(*self.circuit.list_graph_edges[e])
        cond = nx.is_connected(H)

        for e in idx:
            H.add_edge(*self.circuit.list_graph_edges[e])

        return cond, idx

    def generate_coherent_closure(self):

        prob = np.random.sample(len(self.circuit.list_graph_edges))
        idx = np.where(prob <= self.percentage_broken)[0]

        for e in idx:
            self.AUX.remove_edge(*self.circuit.list_graph_edges[e])
        cond = nx.is_connected(self.AUX)

        for e in idx:
            self.AUX.add_edge(*self.circuit.list_graph_edges[e])

        return cond, idx

    def break_links(self, idx, conduct):

        C_aux = np.array(conduct)
        C_aux[idx] = np.power(10., -20)

        return C_aux

    def get_sets(self):

        idx = rd.choices(self.broken_sets, k=self.num_iteration)

        return idx

    def calc_random_radii(self, *args):

        graph_matrices = self.get_broken_links_asarray(*args)

        R, R_sq, R_cb = [], [], []
        for gm in graph_matrices:

            kernel = gm/self.circuit.scales['conductance']

            R.append(np.power(kernel, 0.25))
            R_sq.append(np.sqrt(kernel))
            R_cb.append(np.power(kernel, 0.75))

        return [R, R_sq, R_cb]

    def calc_sq_flow(self, *args):

        # block p percent of the edges per realization
        graph_matrices = self.get_broken_links_asarray(*args)
        flow_observables = list(map(self.calc_flows_mapping, graph_matrices))

        # calc ensemble averages
        q_sq = np.power([fo[0] for fo in flow_observables], 2)
        p_sq = np.power([fo[2] for fo in flow_observables], 2)

        return p_sq, q_sq

    def calc_flows_mapping(self, graph_matrices):

        C_aux = graph_matrices
        dP, P = self.calc_pressure(C_aux, self.circuit.nodes['source'])
        Q = self.calc_flow_from_pressure(C_aux, dP)

        return [Q, P, dP]

    def calc_sq_flow_avg(self, conduct):

        idx = rd.choices(self.broken_sets, k=self.num_iteration)

        p_sq, q_sq = self.calc_sq_flow(idx, conduct)
        R, R_sq, R_cb = self.calc_random_radii(idx, conduct)

        n = float(self.num_iteration)
        avg_diss = np.sum(np.multiply(p_sq, R_cb), axis=0)/n
        avg_R = np.mean(R, axis=0)
        avg_dP_sq = np.mean(p_sq, axis=0)
        avg_F_sq = np.mean(q_sq, axis=0)

        return [avg_dP_sq, avg_F_sq, avg_R, avg_diss]

    def get_broken_links_asarray(self, idx, conduct):

        graph_matrices = [self.break_links(i, conduct) for i in idx]

        return graph_matrices
