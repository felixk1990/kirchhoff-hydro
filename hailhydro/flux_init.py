# @Author: Felix Kramer <kramer>
# @Date:   08-03-2022
# @Email:  felixuwekramer@proton.me
# @Last modified by:   kramer
# @Last modified time: 08-07-2022

import numpy as np
import networkx as nx
from hailhydro.flow_init import Flow
from kirchhoff.circuit_init import Circuit
from kirchhoff.circuit_flux import FluxCircuit
from dataclasses import dataclass, field


@dataclass
class Flux(Flow):
    """
    The flux class defines variables and methods for computing Hagen-Poiseuille
    flows on kirchhoff networks. Furthermore it enables to compute simple,
    stationary advection-diffusion+absorption problems and concentration
    landscapes.

    To be used in conjunction with 'kirchhoff' and 'goflow' in order to
    simulate flow-driven network morphogenesis.

    Attributes:
        constr (networkx.Graph):\n
            A networkx graph or circuit to initilize a flow on.
        pars_source (dict):\n
            The boundary conditions (Neumann) determining the in/outlfow of
            fluid accross the network.
        pars_plexus (dict):\n
            The initial plexus, edge values of  conductivity, the flow is to
            be calculated on.

        pars_solute (dict):\n
            The initial plexus, edge values of  conductivity, the flow is to
            be calculated on.
        pars_abs (dict):\n
            The initial plexus, edge values of  conductivity, the flow is to
            be calculated on.
        pars_geom (dict):\n
            The initial plexus, edge values of  conductivity, the flow is to
            be calculated on.

        dict_in (dict):\n
            The initial plexus, edge values of  conductivity, the flow is to
            be calculated on.
        dict_out (dict):\n
            The initial plexus, edge values of  conductivity, the flow is to
            be calculated on.
        dict_edges (dict):\n
            The initial plexus, edge values of  conductivity, the flow is to
            be calculated on.

        dict_node_out (dict):\n
            The initial plexus, edge values of  conductivity, the flow is to
            be calculated on.
        dict_node_in (dict):\n
            The initial plexus, edge values of  conductivity, the flow is to
            be calculated on.

    Methods:
        init_flow():\n
            Initialize flow variables, boundaries and handle constructor
            exceptions.
        set_boundaries():\n
            Explicitly set Neumann-boudaries and initial plexus as defined via
            'pars_source/plexus' parameters. Set internal output varaibles and
            incidence information.
        find_roots(G):\n
            Given a networkx graph, return all source-nodes (needs the nodal
            'source' attribute set).
        find_sinks(G):\n
            Given a networkx graph, return all sink-nodes (needs the nodal
            'source' attribute set).
        alpha_omega(G, j):\n
            Return the start (alpha) and end(omega) node of an edge, for any
            given networkx graph with edge labeling j.
        calc_pressure(conduct, source):\n
            Compute the pressure landscape, considering the current parameter
            and plexus condition.
        calc_flow_from_pressure(conduct, dP):\n
            Compute the flow landscape, considering the current parameter
            and plexus condition.
        calc_flow(conduct, source):\n
            Compute the flow landscape, considering the current parameter
            and plexus condition.
        calc_sq_flow(sconduct, source):\n
            Compute the squared pressure/flow landscape, considering the
            current parameter and plexus condition.
        calc_cross_section_from_conductivity(conductivity, conductance):\n
            Compute the squared radii values from the current conductivity
            matrix and conductance value.
        calc_conductivity_from_cross_section(R_sq, conductance):\n
            Compute the conductivity matrix from the current squared radii
            values and conductance value.
        calc_configuration_flow():\n
            Compute the pressure/flow landscape, considering the current
            parameter and plexus condition.
        init_flux():\n
            Initialize internal flux variables, boundaries and handle
            constructor exceptions.
        init_parameters():\n
            Initialize internal variables and containers.
        set_solute_boundaries():\n
            Set flux parameters and boundaries.
        calc_diff_flux(R_sq):\n
            Compute the reweighted cross-section given an advection-diffusion
            problem.
        calc_velocity_from_flowrate(Q, R_sq):\n
            Compute the effective flow velocities.
        calc_peclet(V):\n
            Compute the Peclet numbers.

    """
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
        """
        Initialize internal flux variables, boundaries and handle constructor
        exceptions.

        Raises:
            Exception:\n
                Warning! Non-networkx type given for initialization, no
                internal circuit established.

        """

        if type(self.constr) == nx.Graph:

            self.circuit = FluxCircuit(self.constr)

        elif type(self.constr) == FluxCircuit:

            self.circuit = self.constr

        elif isinstance(self.constr, Circuit):

            self.circuit = FluxCircuit(self.constr.G)

        else:
            raise Exception(
                '''
                Warning! Non-networkx type given for initialization, no
                internal circuit established.
                '''
            )

        self.set_boundaries()
        self.set_solute_boundaries()
        self.init_parameters()

    def init_parameters(self):

        """
        Initialize internal variables and containers.

        """

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
        """
        Set flux parameters and boundaries.

        """

        par1 = self.circuit.graph['solute_mode']
        par2 = self.circuit.graph['absorption_mode']
        par3 = self.circuit.graph['geom_mode']

        if par1 == '' or par2 == '' or par3 == '':

            self.circuit.set_solute_landscape(**self.pars_solute)
            self.circuit.set_absorption_landscape(**self.pars_abs)
            self.circuit.set_geom_landscape(**self.pars_geom)

            idx = np.where(self.circuit.nodes['solute'] > 0.)[0]
            sol = self.circuit.nodes['solute'][idx]
            self.circuit.scales['sum_flux'] = np.sum(sol)

    def calc_diff_flux(self, R_sq):

        """
        Compute the reweighted cross-section given an advection-diffusion
        problem.

        Args:
            R_sq (array):\n
                The squared edge radii values.

        Returns:
            ndarray:\n
             Edge-vector of effective diffusion flux across the cross-section.

        """
        A = np.pi*R_sq*self.ref_vars

        return A

    def calc_velocity_from_flowrate(self, Q, R_sq):
        """
        Compute the effective flow velocities.

        Args:
            Q (array):\n
                Edge-vector of directed flow rates.
            R_sq (array):\n
                The squared edge radii values.

        Returns:
            ndarray: Edge-vector of cross-section averaged flow velocities.

        """
        V = np.divide(Q, R_sq*np.pi)

        return V

    def calc_peclet(self, V):
        """
        Compute the Peclet numbers.

        Args:
            V (array):\n
                Edge-vector of cross-section averaged flow velocities.

        Returns:
            ndarray: Edge-vector of peclet numbers.

        """
        PE = V/self.ref_vars

        return PE
