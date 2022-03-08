# @Author:  Felix Kramer
# @Date:   2021-06-03T11:02:57+02:00
# @Email:  kramer@mpi-cbg.de
# @Project: go-with-the-flow
# @Last modified by:    Felix Kramer
# @Last modified time: 2021-08-30T23:39:32+02:00
# @License: MIT
import numpy as np
import networkx as nx
from hailhydro.flow_init import Flow
from kirchhoff.circuit_init import Circuit
from kirchhoff.circuit_flux import FluxCircuit
from dataclasses import dataclass, field


@dataclass
class Flux(Flow):

    # incidence correlation
    defVal1 = dict(
        default_factory=dict,
        repr=False,
        init=False,
        )
    defVal2 = dict(
        default_factory=dict,
        repr=False,
        )

    pars_solute: dict = field(**defVal2)
    pars_abs: dict = field(**defVal2)
    pars_geom: dict = field(**defVal2)

    dict_in: dict = field(**defVal1)
    dict_out: dict = field(**defVal1)
    dict_edges: dict = field(**defVal1)

    # incidence indices
    dict_node_out: dict = field(**defVal1)
    dict_node_in: dict = field(**defVal1)

    def __post_init__(self):

        self.init_flux()

    def init_flux(self):

        if type(self.constr) == nx.Graph:

            self.circuit = FluxCircuit(self.constr)

        elif type(self.constr) == FluxCircuit:

            self.circuit = self.constr


        elif isinstance(self.constr, Circuit):

            self.circuit = FluxCircuit(self.constr.G)

        else:
            raise Exception('Warning! Non-networkx type given for initialization, no internal circuit established.')

        self.set_boundaries()
        self.set_solute_boundaries()
        self.init_parameters()

    def init_parameters(self):

        diff = self.circuit.scales['diffusion']
        L = self.circuit.scales['length']
        self.ref_vars = diff/L
        self.N = len(self.circuit.list_graph_nodes)
        self.M = len(self.circuit.list_graph_edges)
        self.circuit.nodes['concentration'] = np.zeros(self.N)

        sinks = self.find_sinks(self.circuit.G)
        roots = self.find_roots(self.circuit.G)

        self.sinks = sinks
        self.roots = roots
        self.nodes_sinks = [self.circuit.G.nodes[n]['label'] for n in sinks]
        self.nodes_roots = [self.circuit.G.nodes[n]['label'] for n in roots]

        self.idx_eff = [i for i in range(self.N) if i not in self.nodes_sinks]

        for i, n in enumerate(self.circuit.list_graph_nodes):
            self.dict_in[n] = []
            self.dict_out[n] = []
            self.dict_node_out[n] = np.where(self.B[i, :] > 0)[0]
            self.dict_node_in[n] = np.where(self.B[i, :] < 0)[0]

        for j, e in enumerate(self.circuit.list_graph_edges):

            alpha = e[1]
            omega = e[0]
            if self.B[alpha, j] > 0.:

                self.dict_edges[e] = [alpha, omega]
                self.dict_in[omega].append(alpha)
                self.dict_out[alpha].append(omega)

            elif self.B[alpha, j] < 0.:

                self.dict_edges[e] = [omega, alpha]
                self.dict_in[alpha].append(omega)
                self.dict_out[omega].append(alpha)

            else:
                print('and I say...whats going on? I say heyayayayayaaaaa...')

    def set_solute_boundaries(self):

        par1 = self.circuit.graph['solute_mode']
        par2 = self.circuit.graph['absorption_mode']
        par3 = self.circuit.graph['geom_mode']

        if par1 == '' or par2 == '' or par3 == '':

            self.circuit.set_solute_landscape(**self.pars_solute)
            self.circuit.set_absorption_landscape(**self.pars_abs)
            self.circuit.set_geom_landscape(**self.pars_geom)

            idx = np.where(self.circuit.nodes['solute'] > 0.)[0]
            self.circuit.scales['sum_flux'] = np.sum(self.circuit.nodes['solute'][idx])

    def calc_diff_flux(self, R_sq):

        A = np.pi*R_sq*self.ref_vars

        return A

    def calc_velocity_from_flowrate(self, Q, R_sq):

        V = np.divide(Q, R_sq*np.pi)

        return V

    def calc_peclet(self, V):

        PE = V/self.ref_vars

        return PE
