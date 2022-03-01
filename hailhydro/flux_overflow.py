# @Author:  Felix Kramer
# @Date:   2021-06-03T11:02:57+02:00
# @Email:  kramer@mpi-cbg.de
# @Project: go-with-the-flow
# @Last modified by:    Felix Kramer
# @Last modified time: 2021-08-30T23:38:31+02:00
# @License: MIT
import numpy as np
import scipy.linalg as lina
from hailhydro.flux_init import Flux
from dataclasses import dataclass, field


@dataclass
class Overflow(Flux):

    def __post_init__(self):

        self.init_flux()
        self.crit_pe = 50.

    # compute concentration profile
    # def calc_profile_concentration(self):
    #
    #     self.update_transport_matrix()
    #     c, B_new = self.solve_absorbing_boundary()
    #
    #     return c, B_new

    def solve_absorbing_boundary(self):

        # reduce transport matrix by cutting row, col corresponding to
        # absorbing boundary

        B_new = self.B_eff[self.idx_eff, :]
        B_new = B_new[:, self.idx_eff]
        S = self.circuit.nodes['solute'][self.idx_eff]

        concentration = np.zeros(self.N)
        # solving inverse flux problem for absorbing boundaries
        concentration_reduced = np.dot(np.linalg.inv(B_new), S)
        concentration[self.idx_eff] = concentration_reduced[:]

        # export solution
        for i, n in enumerate(self.circuit.list_graph_nodes):
            self.circuit.G.nodes[n]['concentrations'] = concentration[i]
        self.circuit.nodes['concentration'] = concentration[:]

        return concentration_reduced, B_new

    # def update_transport_matrix(self):
    #
    #     A = self.calc_diff_flux(self.circuit.edges['radius_sq'])
    #     print(self.circuit.edges['radius_sq'])
    #     k = self.circuit.edges['conductivity']
    #     s = self.circuit.nodes['source']
    #     Q = self.calc_flow(k, s)
    #     r_sq = self.circuit.edges['radius_sq']
    #     V = self.calc_velocity_from_flowrate(Q, r_sq)
    #     self.circuit.edges['peclet'] = self.calc_peclet(V)
    #     self.circuit.edges['flow_rate'] = Q
    #
    #     x, z = self.compute_flux_PeAbs()
    #     idx_pack = self.compute_flux_idx()
    #
    #     args = [x, z, idx_pack]
    #     e_up_sinh_x, e_down_sinh_x, coth_x = self.compute_flux_exp(*args)
    #
    #     f1 = np.multiply(z, A)
    #     f2 = np.multiply(A, np.multiply(x, coth_x))*0.5
    #     pars = [e_up_sinh_x, e_down_sinh_x]
    #     f3, f4 = self.calc_abs_jac_coeff_11(0.5*np.multiply(A, x), pars)
    #
    #     self.B_eff = np.zeros((self.N, self.N))
    #
    #     for i, n in enumerate(self.circuit.list_graph_nodes):
    #
    #         b1 = np.multiply(self.B[i, :], f1)
    #         b2 = np.multiply(np.absolute(self.B[i, :]), f2)
    #         b12 = np.add(b1, b2)
    #
    #         self.B_eff[i, i] = np.sum(b12)
    #         self.B_eff[i, self.dict_in[n]] = -f3[self.dict_node_in[n]]
    #         self.B_eff[i, self.dict_out[n]] = -f4[self.dict_node_out[n]]

    def update_transport_matrix(self, R):

        r_sq = np.power(R, 2)
        A = self.calc_diff_flux(r_sq)
        self.circuit.edges['radius_sq'] = r_sq

        k = self.circuit.scales['conductance']
        src = self.circuit.nodes['source']
        conductivity = self.calc_conductivity_from_cross_section(r_sq, k)
        Q = self.calc_flow(conductivity, src)

        V = self.calc_velocity_from_flowrate(Q, r_sq)
        self.circuit.edges['peclet'] = self.calc_peclet(V)
        self.circuit.edges['flow_rate'] = Q

        x, z = self.compute_flux_PeAbs()
        idx_pack = self.compute_flux_idx()

        args = [x, z, idx_pack]
        e_up_sinh_x, e_down_sinh_x, coth_x = self.compute_flux_exp(*args)

        f1 = np.multiply(z, A)
        f2 = np.multiply(A, np.multiply(x, coth_x))*0.5
        pars = [e_up_sinh_x, e_down_sinh_x]
        f3, f4 = self.calc_abs_jac_coeff_11(0.5*np.multiply(A, x), pars)

        self.B_eff = np.zeros((self.N, self.N))

        for i, n in enumerate(self.circuit.list_graph_nodes):

            b1 = np.multiply(self.B[i, :], f1)
            b2 = np.multiply(np.absolute(self.B[i, :]), f2)
            b12 = np.add(b1, b2)

            self.B_eff[i, i] = np.sum(b12)
            self.B_eff[i, self.dict_in[n]] = -f3[self.dict_node_in[n]]
            self.B_eff[i, self.dict_out[n]] = -f4[self.dict_node_out[n]]

    def compute_flux_PeAbs(self):

        pe = self.circuit.edges['peclet']
        beta = self.circuit.edges['absorption']
        x = np.sqrt(np.add(np.power(pe, 2), beta))
        z = self.circuit.edges['peclet']*0.5

        return x, z

    def compute_flux_idx(self):

        # establish the use of converging limit expressions to prevent
        # overflow error
        pe = self.circuit.edges['peclet']
        idx_lower = np.where(np.absolute(pe) < self.crit_pe)[0]
        idx_over_p = np.where((np.absolute(pe) >= self.crit_pe) & (pe > 0))[0]
        idx_over_n = np.where((np.absolute(pe) >= self.crit_pe) & (pe < 0))[0]
        idx_pack = [list(idx_lower), list(idx_over_p), list(idx_over_n)]

        return idx_pack

    def compute_flux_exp(self, x, z, idx_pack):

        e_up_sinh_x = np.zeros(self.M)
        e_down_sinh_x = np.zeros(self.M)
        coth_x = np.zeros(self.M)

        # subcritical pe
        idx_lower = idx_pack[0]
        e_up = np.exp(z[idx_lower])
        e_down = np.exp(-z[idx_lower])
        e_up_sinh_x[idx_lower] = e_up/np.sinh(x[idx_lower]*0.5)
        e_down_sinh_x[idx_lower] = e_down/np.sinh(x[idx_lower]*0.5)
        coth_x[idx_lower] = np.reciprocal(np.tanh(x[idx_lower]*0.5))

        # overcriticial,  pe positive
        idx_over_pos = idx_pack[1]
        e_up_sinh_x[idx_over_pos] = 2.
        e_down_sinh_x[idx_over_pos] = 0.
        coth_x[idx_over_pos] = 1.

        # overcriticial,  pe negative
        idx_over_neg = idx_pack[2]
        e_up_sinh_x[idx_over_neg] = 0.
        e_down_sinh_x[idx_over_neg] = 2.
        coth_x[idx_over_neg] = 1.

        return e_up_sinh_x, e_down_sinh_x, coth_x

    # calc link absorption
    def calc_absorption(self):

        # calc coefficients
        c_a, c_b = self.get_concentrations_from_edges()
        x, z = self.compute_flux_PeAbs()
        idx_pack = self.compute_flux_idx()
        args = [x, z, idx_pack]
        e_up_sinh_x, e_down_sinh_x, coth_x = self.compute_flux_exp(*args)

        pars = [e_up_sinh_x, e_down_sinh_x]
        f1_up, f1_down = self.calc_abs_jac_coeff_11(0.5*x, pars)
        x_coth_x = np.multiply(x, coth_x)*0.5
        F1 = np.add(np.subtract(x_coth_x, f1_up),  z)
        F2 = np.subtract(np.subtract(x_coth_x, f1_down),  z)

        # calc edgewise absorption
        phi = np.add(np.multiply(c_a, F1), np.multiply(c_b, F2))
        A = self.calc_diff_flux(self.circuit.edges['radius_sq'])

        PHI = np.multiply(A, phi)
        self.circuit.edges['uptake'] = PHI[:]

        return PHI

    def get_concentrations_from_edges(self):

        # set containers
        c_a, c_b = np.ones(self.M), np.ones(self.M)

        for j, e in enumerate(self.circuit.list_graph_edges):

            a, b = self.dict_edges[e]
            c_a[j] = self.circuit.G.nodes[a]['concentrations']
            c_b[j] = self.circuit.G.nodes[b]['concentrations']

        return c_a, c_b

    # calc absorption jacobian and subcomponents
    def calc_absorption_jacobian(self):

        # calc coefficients
        A = self.calc_diff_flux(self.circuit.edges['radius_sq'])
        alphas, omegas = self.get_alpha_omega_from_edges()
        c_a, c_b = self.get_concentrations_from_edges()
        c_n = self.circuit.nodes['concentration']

        x, z = self.compute_flux_PeAbs()
        idx_pack = self.compute_flux_idx()
        args = [x, z, idx_pack]
        e_up_sinh_x, e_down_sinh_x, coth_x = self.compute_flux_exp(*args)

        pars = [x, z, e_up_sinh_x, e_down_sinh_x, coth_x, idx_pack]
        F1, F2 = self.calc_absorption_jacobian_coefficients_1(*pars)
        F3, F4 = self.calc_absorption_jacobian_coefficients_2(*pars)
        qa, qb, q1, q2 = self.calc_abs_jac_coeff_11(A, [c_a, c_b, F1, F2])

        qaF3 = np.multiply(qa, F3)
        qbF4 = np.multiply(qb, F4)

        phi = np.add(np.multiply(c_a, F1), np.multiply(c_b, F2))

        # calc jacobian components
        J_PE, J_Q = self.calc_flux_jacobian()
        J_A = self.calc_cross_section_jacobian()
        J_C = self.calc_concentration_jacobian(J_PE, c_n)

        J_phi = np.zeros((self.M, self.M))
        for j, e in enumerate(self.circuit.list_graph_edges):

            J_phi[j, :] = np.add(J_phi[j, :], np.multiply(J_A[j, :], phi))

            J_phi[j, :] = np.add(J_phi[j, :], np.multiply(J_C[j, alphas], q1))
            J_phi[j, :] = np.add(J_phi[j, :], np.multiply(J_C[j, omegas], q2))

            J_phi[j, :] = np.add(J_phi[j, :], np.multiply(J_PE[j, :], qaF3))
            J_phi[j, :] = np.add(J_phi[j, :], np.multiply(J_PE[j, :], qbF4))

        return J_phi

    def get_alpha_omega_from_edges(self):

        alphas, omegas = [], []
        for j, e in enumerate(self.circuit.list_graph_edges):

            a, b = self.dict_edges[e]
            alphas.append(a)
            omegas.append(b)

        return alphas, omegas

    def calc_absorption_jacobian_coefficients_1(self, *args):

        x, z, e_up_sinh_x, e_down_sinh_x, coth_x, idx_pack = args

        pars = [e_up_sinh_x, e_down_sinh_x]
        f1_up, f1_down = self.calc_abs_jac_coeff_11(0.5*x, pars)
        F1 = np.add(np.subtract(0.5*np.multiply(x, coth_x), f1_up), z)
        F2 = np.subtract(np.subtract(0.5*np.multiply(x, coth_x), f1_down), z)

        return F1, F2

    def calc_abs_jac_coeff_11(self, x, pars):

        coeff = [np.multiply(x, y) for y in pars]

        return coeff

    def calc_absorption_jacobian_coefficients_2(self, *args):

        x, z, e_up_sinh_x, e_down_sinh_x, coth_x, idx_pack = args
        F3 = np.zeros(self.M)
        F4 = np.zeros(self.M)
        idx_lower = idx_pack[0]
        idx_over = idx_pack[1]+idx_pack[2]

        # subcritical
        sinh_x = np.sinh(x[idx_lower]*0.5)
        cosh_x = np.cosh(x[idx_lower]*0.5)
        e_up = np.exp(z[idx_lower])
        e_down = np.exp(-z[idx_lower])

        d2 = np.divide(2.*z[idx_lower], x[idx_lower])
        e2 = np.divide(z[idx_lower], sinh_x)

        f2_up = np.subtract(np.multiply(d2, np.subtract(cosh_x, e_up)), e2)
        f2_down = np.subtract(np.multiply(d2, np.subtract(cosh_x, e_down)), e2)

        m3 = np.multiply(coth_x[idx_lower], z[idx_lower])
        d3u = np.subtract(m3, 0.5*x[idx_lower])
        d3d = np.add(m3, 0.5*x[idx_lower])

        f3_up = np.add(np.multiply(e_up, d3u), sinh_x)
        f3_down = np.subtract(np.multiply(e_down, d3d), sinh_x)

        F3[idx_lower] = 0.5*np.divide(np.add(f2_up, f3_up), sinh_x)
        F4[idx_lower] = 0.5*np.divide(np.add(f2_down, f3_down), sinh_x)

        # overcritical
        d2 = np.divide(2.*z[idx_over], x[idx_over])
        m3u = np.multiply(coth_x[idx_over], z[idx_over])

        f2_up = np.multiply(d2, np.subtract(coth_x[idx_over], 2.))
        f3_up = np.add(2.*np.subtract(m3u, 0.5*x[idx_over]), 1.)
        f2_down = np.multiply(d2, coth_x[idx_over])
        # f3_down = np.zeros(len(idx_over))

        F3[idx_over] = 0.5*np.add(f2_up, f3_up)
        F4[idx_over] = 0.5*f2_down

        return F3, F4

    # calc flux jacobian and subcomponents
    def calc_flux_jacobian(self):

        # init containers
        idt = np.identity(self.M)
        J_PE, J_Q = np.zeros((self.M, self.M)), np.zeros((self.M, self.M))

        # set coefficients
        r = self.circuit.edges['radius']
        f1 = 2.*np.divide(self.circuit.edges['peclet'], r)
        f2 = 4.*np.divide(self.circuit.edges['flow_rate'], r)

        k = self.circuit.edges['conductivity']
        INV = lina.pinv(np.dot(self.B, np.dot(np.diag(k), self.BT)))
        D = np.dot(np.dot(self.BT, INV), self.B)

        # calc jacobian
        r_sq = self.circuit.edges['radius_sq']
        L = self.circuit.edges['length']
        for i, c in enumerate(k):

            sq_kernel = r_sq/r_sq[i]
            l_kernel = np.divide(L[i], L)

            s1 = 2.*c*np.multiply(D[:, i], sq_kernel)
            J_PE[i, :] = f1[i] * np.subtract(idt[i, :], s1)

            m2 = np.multiply(l_kernel, np.power(sq_kernel, 2))
            s2 = c*np.multiply(D[:, i], m2)
            J_Q[i, :] = f2[i] * np.subtract(idt[i, :], s2)

        return J_PE, J_Q

    # calc cross section jacobian and subcomponents
    def calc_cross_section_jacobian(self):

        J_A = 2.*np.pi*np.diag(self.circuit.edges['radius'])*self.ref_vars

        return J_A

    # calc concentraion jacobian and subcomponents
    def calc_concentration_jacobian(self, J_PE, c):

        # set coefficients
        dict_coeff = self.calc_concentration_jacobian_coefficients(c)
        fs1 = dict_coeff['flux_sum_1']
        fs2 = dict_coeff['flux_sum_2']
        inv_B, c = self.calc_inv_B(c)

        J_C = np.zeros((self.M, self.N))
        J_diag = np.diag(self.calc_cross_section_jacobian())
        J_A = np.zeros(self.M)

        for j, e in enumerate(self.circuit.list_graph_edges):

            J_A[j-1] = 0.
            J_A[j] = J_diag[j]
            J_PE_j = J_PE[j, :]
            pars = [[J_diag[j], J_PE_j, J_A, j] for i in range(self.N)]

            JB_eff = np.diag(list(map(self.calc_inc_jac_diag, fs1, fs2, pars)))
            self.calc_incidence_jacobian_dev(JB_eff, dict_coeff, pars[0])
            self.evaluate_jacobian(j, J_C, JB_eff, inv_B, c)

        return J_C

    def calc_concentration_jacobian_coefficients(self, c):

        dict_coeff = {}
        A = self.calc_diff_flux(self.circuit.edges['radius_sq'])
        x, z = self.compute_flux_PeAbs()
        idx_pack = self.compute_flux_idx()
        args = [x, z, idx_pack]
        e_up_sinh_x, e_down_sinh_x, coth_x = self.compute_flux_exp(*args)

        f2 = np.multiply(np.multiply(x, coth_x), A)*0.5
        pars = [e_up_sinh_x, e_down_sinh_x]
        f3, f4 = self.calc_abs_jac_coeff_11(0.5*np.multiply(A, x), pars)

        j_coth_x = np.zeros(self.M)
        idx_lower = idx_pack[0]
        d1 = np.divide(coth_x[idx_lower], np.cosh(x[idx_lower]))
        j_coth_x[idx_lower] = np.power(d1, 2)

        f2 = np.multiply(x, coth_x)*0.5
        d2 = np.divide(z, x)
        f4 = np.subtract(np.multiply(d2, coth_x), np.multiply(z, j_coth_x)*0.5)

        pars = [e_up_sinh_x, e_down_sinh_x]
        f_up, f_down = self.calc_abs_jac_coeff_11(0.5*x, pars)
        f5, f6 = self.calc_abs_jac_coeff_11(A, pars)

        m2 = np.multiply(z, coth_x)*0.5
        J_f_up = -np.multiply(f5, np.subtract(np.add(d2, x*0.25), m2))
        J_f_down = -np.multiply(f6, np.subtract(np.subtract(d2, x*0.25), m2))

        dict_coeff['f_up'] = f_up
        dict_coeff['f_down'] = f_down
        dict_coeff['J_f_up'] = J_f_up
        dict_coeff['J_f_down'] = J_f_down

        list_n = self.circuit.list_graph_nodes
        arg1 = [z, f2]
        flux_sum_1 = [self.flux_sum_1(i, *arg1) for i, n in enumerate(list_n)]
        arg2 = [z, A, f4]
        flux_sum_2 = [self.flux_sum_2(i, *arg2) for i, n in enumerate(list_n)]

        dict_coeff['flux_sum_1'] = np.array(flux_sum_1)
        dict_coeff['flux_sum_2'] = np.array(flux_sum_2)

        return dict_coeff

    def flux_sum_1(self, i, z, f2):

        f = np.multiply(np.absolute(self.B[i, :]), f2)
        fs = np.add(np.multiply(self.B[i, :], z), f)

        return fs

    def flux_sum_2(self, i, z, A, f4):

        f = np.multiply(np.absolute(self.B[i, :]), f4)
        fs = np.multiply(A, np.add(self.B[i, :]*0.5, f))

        return fs

    def calc_inv_B(self, c):

        B_new = self.B_eff[self.idx_eff, :]
        B_new = B_new[:, self.idx_eff]
        c = c[self.idx_eff]
        inv_B = np.linalg.inv(B_new)

        return inv_B, c

    def calc_inc_jac_diag(self, flux_sum_1, flux_sum_2, pars):

        J_diag, J_PE_j, J_A, j = pars
        JB_eff = J_diag*flux_sum_1[j]+np.sum(np.multiply(J_PE_j, flux_sum_2))

        return JB_eff

    def calc_incidence_jacobian_dev(self, JB_eff, dict_coeff, pars):

        J_diag, J_PE_j, J_A, j = pars

        jfd = dict_coeff['J_f_down']
        fd = dict_coeff['f_down']
        jfu = dict_coeff['J_f_up']
        fu = dict_coeff['f_up']

        for i, n in enumerate(self.circuit.list_graph_nodes):

            do = self.dict_out[n]
            dno = self.dict_node_out[n]
            fo1 = np.multiply(J_PE_j[dno], jfd[dno])
            fo2 = np.multiply(J_A[dno], fd[dno])
            JB_eff[i, do] = np.subtract(fo1, fo2)

            di = self.dict_in[n]
            dni = self.dict_node_in[n]
            fi1 = np.multiply(J_PE_j[dni], jfu[dni])
            fi2 = np.multiply(J_A[dni], fu[dni])
            JB_eff[i, di] = np.subtract(fi1, fi2)

    def evaluate_jacobian(self, j, J_C, JB_eff, inv_B, c):

        JB_new = JB_eff[self.idx_eff, :]
        JB_new = JB_new[:, self.idx_eff]
        J_C[j, self.idx_eff] = -np.dot(inv_B, np.dot(JB_new, c))
