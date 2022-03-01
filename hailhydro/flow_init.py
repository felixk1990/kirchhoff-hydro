# @Author:  Felix Kramer
# @Date:   2021-06-03T11:02:33+02:00
# @Email:  kramer@mpi-cbg.de
# @Project: go-with-the-flow
# @Last modified by:    Felix Kramer
# @Last modified time: 2021-09-04T23:31:01+02:00
# @License: MIT
import numpy as np
import networkx as nx
from kirchhoff.circuit_init import Circuit
from kirchhoff.circuit_flow import FlowCircuit
from dataclasses import dataclass, field


@dataclass
class Flow():

    constr: nx.Graph = field(repr=False, init=True)
    pars_source: dict = field(default_factory=dict, repr=False)
    pars_plexus: dict = field(default_factory=dict, repr=False)


    def __post_init__(self):

        self.init_flow()

    def init_flow(self):

        self.info: str = 'unknown'

        if isinstance(self.constr, nx.Graph):

            self.circuit = FlowCircuit(self.constr)

        elif isinstance(self.constr, FlowCircuit):

            self.circuit = self.constr

        elif isinstance(self.constr, Circuit):

            self.circuit = FlowCircuit(self.constr.G)

        else:
            raise Exception('Warning! Non-networkx type given for initialization, no internal circuit established.')

        self.set_boundaries()

    def set_boundaries(self):

        par1 = self.circuit.graph['source_mode']
        par2 = self.circuit.graph['plexus_mode']

        if par1 == '' or par2 == '':
            self.circuit.set_source_landscape(**self.pars_source)
            self.circuit.set_plexus_landscape(**self.pars_plexus)

        self.info = self.circuit.info
        self.B, self.BT = self.circuit.get_incidence_matrices()

    def find_roots(self, G):

        list_n = self.circuit.list_graph_nodes
        roots = [n for n in list_n if G.nodes[n]['source'] > 0]

        return roots

    def find_sinks(self, G):

        list_n = self.circuit.list_graph_nodes
        sinks = [n for n in list_n if G.nodes[n]['source'] < 0]

        return sinks

    def alpha_omega(self, G, j):

        labels = nx.get_edge_attributes(G, 'label')
        for e, label in labels.items():
            if label == j:
                alpha = e[1]
                omega = e[0]

        return alpha, omega

    def calc_pressure(self, conduct, source):

        OP = np.dot(self.B, np.dot(np.diag(conduct), self.BT))
        P, RES, RG, si = np.linalg.lstsq(OP, source, rcond=None)
        dP = np.dot(self.BT, P)

        return dP,  P

    def calc_flow_from_pressure(self, conduct, dP):

        Q = np.dot(np.diag(conduct), dP)

        return Q

    def calc_flow(self, conduct, source):

        dP, P = self.calc_pressure(conduct, source)
        Q = np.dot(np.diag(conduct), dP)

        return Q

    def calc_sq_flow(self, conduct, source):

        dP, P = self.calc_pressure(conduct, source)
        Q = self.calc_flow_from_pressure(conduct, dP)

        p_sq = np.multiply(dP, dP)
        q_sq = np.multiply(Q, Q)

        return p_sq,  q_sq

    def calc_cross_section_from_conductivity(self, conductivity, conductance):

        R_sq = np.sqrt(conductivity/conductance)

        return R_sq

    def calc_conductivity_from_cross_section(self, R_sq, conductance):

        conductivity = np.power(R_sq, 2)*conductance

        return conductivity

    def calc_configuration_flow(self):

        k = self.circuit.edges['conductivity']
        src = self.circuit.nodes['source']

        dP, P = self.calc_pressure(k, src)
        Q = np.dot(np.diag(k), dP)

        return Q, dP
