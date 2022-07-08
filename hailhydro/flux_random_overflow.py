# @Author: Felix Kramer <kramer>
# @Date:   02-09-2021
# @Email:  felixuwekramer@proton.me
# @Last modified by:   kramer
# @Last modified time: 08-07-2022

import numpy as np
from hailhydro.flow_random import FlowReroute
from hailhydro.flux_overflow import Overflow
from dataclasses import dataclass


@dataclass
class FluxRandom(Overflow, FlowReroute):
    """
    The flux class defines variables and methods for computing Hagen-Poiseuille
    flows on kirchhoff networks. Furthermore it enables to compute simple,
    stationary advection-diffusion+absorption problems and concentration
    landscapes.

    To be used in conjunction with 'kirchhoff' and 'goflow' in order to
    simulate flow-driven network morphogenesis. This class contains manually
    implemented handling for large Peclet numbers.

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
        solve_absorbing_boundary():\n
            Compute the concentration landscape for the absorbing boundary
            problem.
        update_transport_matrix(R):\n
            Update the effective transport matrix.
        compute_flux_PeAbs():\n
            Compute the effective exponential factors for the stationary
            concentraiton problem.
        compute_flux_idx():\n
            Identify regimes of Peclet numbers and the respective indices of
            edges.
        compute_flux_exp(x, z, idx_pack):\n
            Computes auxillary exponential factors for transport matrix
            evaluation.
        calc_absorption():\n
            Computes total absorption lanscape of the advection-diffusion
            network.
        get_concentrations_from_edges():\n
            Returns the current start and end concentraiton values of each
            individual edge.
        calc_absorption_jacobian():\n
            Compute total edge's absorption Jacobian matrix (with regard to
            radial changes).
        get_alpha_omega_from_edges():\n
            Returns the start(alpha) and end(omega) node of all network edges.
        calc_absorption_jacobian_coefficients_1(*args):\n
            Caluclation of intermediate transport matrix and Jacobain
            components.
        calc_abs_jac_coeff_11(x, pars):\n
            Auxillarycilary function for sequenced ndarray multiplication.
        calc_absorption_jacobian_coefficients_2(*args):\n
            Caluclation of intermediate transport matrix and Jacobain
            omponents.
        calc_flux_jacobian():\n
            Compute the flow components of the absorption Jacobian matrix.
        calc_cross_section_jacobian():\n
            Compute the radial Jacobian component of the absorption Jacobian
            matrix.
        calc_concentration_jacobian(J_PE, c):\n
            Compute the concentration Jacobian component of the absorption
            Jacobian matrix.
        calc_concentration_jacobian_coefficients(c):\n
            Auxillary function to compute intermediate coefficients for
            concentration Jacobian matrix evaluation.
        flux_sum_1(i, z, f2):\n
            Auxillary function for intermediate coefficient computation.
        flux_sum_2(i, z, A, f4):\n
            Auxillary function for intermediate coefficient computation.
        calc_inv_B(c):\n
            Return the reduced concentration vector and inverted transport
            matrix.
        calc_inc_jac_diag(flux_sum_1, flux_sum_2, pars):\n
            Auxillary function to compute intermediate Jacobian components for
            absoprtion Jacobian matrix.
        calc_incidence_jacobian_dev(JB_eff, dict_coeff, pars):\n
            Auxillary function to merge intermediate Jacobian components into
            effective absoprtion Jacobian matrix.
        evaluate_jacobian(j, J_C, JB_eff, inv_B, c):\n
            Update the jth row of the current concentration Jacobian matrix.
        calc_flow(*args):\n
            Compute the flow landscape, considering the current parameter
            and plexus condition.
        calc_transport_observables(idx, conduct, flow_obs):\n
            Compute the average wall-shear stress and absorption rates.
        calc_noisy_absorption(R_sq, flow_observables):\n
            Compute the absorption rate for the current flow landscpae
            realizsation.
        update_transport_matrix(R_sq, flow_obs):\n
            Update the tranport matrix for broken edge realizsations.

    """
    def __post_init__(self, circuit):

        self.init_flux()
        self.crit_pe = 50.
        self.init_random()

    def calc_flow(self, *args):
        """
        Compute the flow landscape, considering the current parameter
        and plexus condition.

        Args:
            args (iterable):\n
                Diverse set of model parameterst for
                'get_broken_links_asarray()''
        Returns:
            list: A list of edge/nodal-vectors of effective flow/pressure
            landscape realizations.

        """

        graph_matrices = self.get_broken_links_asarray(*args)
        flow_observables = list(map(self.calc_flows_mapping, graph_matrices))

        return flow_observables

    def calc_transport_observables(self, idx, conduct, flow_obs):
        """
        Compute the average wall-shear stress and absorption rates.

        Args:
            idx (list):\n
                The list of failed edge sets.
            conduct (array):\n
                The network's conductivity matrix.
            flow_obs (list):\n
                A list of edge/nodal-vectors of effective flow/pressure
                landscape realizations.

        Returns:
            ndarray: Edge-vector of average squared wall-shear stress.
            ndarray: Edge-vector of average edge absorption rate.

        """
        # calc ensemble averages
        self.get_broken_links_asarray(idx, conduct)
        R_powers = self.calc_random_radii(idx, conduct)
        dV_sq = np.power([fo[2] for fo in flow_obs], 2)

        R_sq = R_powers[1]
        PHI = list(map(self.calc_noisy_absorption, R_sq, flow_obs))
        SHEAR = np.multiply(dV_sq, R_sq)

        avg_shear_sq = np.mean(SHEAR, axis=0)
        avg_PHI = np.mean(PHI, axis=0)

        return avg_shear_sq, avg_PHI

    def calc_noisy_absorption(self, R_sq, flow_observables):
        """
        Compute the absorption rate for the current flow landscpae
        realizsation.

        Args:
            R_sq (array):\n
                Edge-vector of squared edge radii.
            flow_observables (array):\n
                A list of edge/nodal-vectors of effective flow/pressure
                landscape realizations.

        Returns:
            ndarray: Edge-vector of absorption rate values.

        """
        self.update_transport_matrix(R_sq, flow_observables)

        c, B_new = self.solve_absorbing_boundary()

        return self.calc_absorption(R_sq)

    def update_transport_matrix(self, R_sq, flow_obs):
        """
        Update the tranport matrix for broken edge realizsations.

        Args:
            R_sq (array):\n
                Edge-vector of squared edge radii.
            flow_obs (array):\n
                A list of edge/nodal-vectors of effective flow/pressure
                landscape realizations.

        """
        # set peclet number and internal flow state
        self.circuit.edge['flow_rate'] = flow_obs[0]
        ref_var = self.circuit.scale['length']/self.circuit.scale['diffusion']

        flow_rate = self.circuit.edge['flow_rate']
        V = self.calc_velocity_from_flowrate(flow_rate, R_sq)
        self.circuit.edge['peclet'] = self.calc_peclet(V, ref_var)
        A = self.calc_diff_flux(R_sq, 1./ref_var)

        x, z = self.compute_flux_PeAbs()
        idx_pack = self.compute_flux_idx()
        args = [x, z, idx_pack]
        e_up_sinh_x, e_down_sinh_x, coth_x = self.compute_flux_exp(*args)

        f1 = np.multiply(z, A)
        f2 = np.multiply(A, np.multiply(x, coth_x))*0.5

        f3 = np.multiply(np.multiply(A, x), e_up_sinh_x)*0.5
        f4 = np.multiply(np.multiply(A, x), e_down_sinh_x)*0.5

        # set up concentration_matrix
        self.B_eff = np.zeros((self.N, self.N))

        for i, n in enumerate(self.circuit.list_graph_nodes):

            b1 = np.multiply(self.B[i, :], f1)
            b2 = np.multiply(np.absolute(self.B[i, :]), f2)
            b12 = np.add(b1, b2)
            self.B_eff[i, i] = np.sum(b12)
            self.B_eff[i, self.dict_in[n]] = -f3[self.dict_node_in[n]]
            self.B_eff[i, self.dict_out[n]] = -f4[self.dict_node_out[n]]
