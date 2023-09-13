import numpy as np
import matplotlib.pyplot as plt

# Constants
planck = 197.32705
alpha = 1 / 137.03599
M_p = 938.27231
M_n = 939.56563
M_r = M_p * M_n / (M_p + M_n)
M_pi0 = 134.9739
M_pi1 = 139.5675
mu_p = 2.7928474
mu_n = -1.9130427
M_e = 0.510999
b = 4.27
pi = np.arccos(-1.0)
gamma = 0.577216
beta = 0.0189
small = 1e-5


# The F_c, F_delta, F_t, and F_ls, are short-range functions that represent the finite size of the nucleon charge distributions.
def short_range_functions(r):
    r = max(r, small)
    br = b * r
    F_C_r = (1 - (1 + 11 * br / 16 + 3 * br**2 / 16 + br**3 / 48) * np.exp(-br)) / r
    F_t_r3 = (
        1
        - (1 + br + br**2 / 2 + br**3 / 6 + br**4 / 24 + br**5 / 144)
        * np.exp(-br)
    ) / r**3
    F_ls_r3 = (
        1 - (1 + br + br**2 / 2 + 7 * br**3 / 48 + br**4 / 48) * np.exp(-br)
    ) / r**3
    kr = M_e * r / planck
    F_ivp = -gamma - 5 / 6 + abs(np.log(kr)) + 6 * pi * kr / 8
    F_delta = b**3 * (1 + br + br**2 / 3) * np.exp(-br) / 16
    #  function F_np is obtained assuming the neutron electric form factor:
    #  G^m_E = beta_n * q^2 * (1 + q^2/b^2)^(-3)
    F_np_r = b**3 * (15 + 15 * br + 6 * br**2 + br**3) * np.exp(-br) / 384
    return F_C_r, F_t_r3, F_ls_r3, F_ivp, F_delta, F_np_r


# The EM interaction is the same as that used in the Nijmegen partial-wave analysis
# arguments for V_EM_NN
# r:     input separation in fm
# V_EM:  output potential in MeV (14 component array)
# ----------------------------------------------------------------------
# order of operators in V_EM(i)
# i:    1=C1    (pp)          2=DF    (pp)          3=C2      (pp)
#       4=VP    (pp)                                5=C1      (np)
#       6=s1.s2 (pp)          7=s1.s2 (nn)          8=s1.s2   (np)
#       9=S12   (pp)         10=S12   (nn)         11=S12     (np)
#      12=L.S   (pp)         13=L.S   (nn)         14=L.S     (np)
# C1 = one-photon-exchange Coulomb with form factor
# C2 = two-photon-exchange Coulomb
# DF = Darwin-Foldy
# VP = vacuum polarization (short-range approximation)
# all other terms from magnetic moment (MM) interactions
def V_EM_NN(r):
    F_C_r, F_t_r3, F_ls_r3, F_ivp, F_delta, F_np_r = short_range_functions(r)
    V_EM = np.zeros(14)
    # V_C1(pp)
    V_EM[0] = alpha * planck * F_C_r
    # V_DF(pp)
    V_EM[1] = -alpha * planck**3 * F_delta / (4 * M_p**2)
    # V_C2(pp)
    V_EM[2] = -V_EM[0] ** 2 / M_p
    # V_VP(pp)
    V_EM[3] = 2 * alpha * V_EM[0] * F_ivp / (3 * pi)
    # V_C1(np)
    V_EM[4] = alpha * planck * beta * F_np_r
    # V_MM(pp)_part1
    V_EM[5] = -alpha * planck**3 * mu_p**2 * F_delta / (6 * M_p**2)
    # V_MM(nn)_part1  ==  V_EM(nn)
    V_EM[6] = -alpha * planck**3 * mu_n**2 * F_delta / (6 * M_n**2)
    # V_MM(np)_part1
    V_EM[7] = -alpha * planck**3 * mu_p * mu_n * F_delta / (6 * M_n * M_p)
    # V_MM(pp)_part2
    V_EM[8] = -alpha * planck**3 * mu_p**2 * F_t_r3 / (4 * M_p**2)
    # V_MM(nn)_part2  ==  V_EM(nn)   -
    V_EM[9] = -alpha * planck**3 * mu_n**2 * F_t_r3 / (4 * M_n**2)
    # V_MM(np)_part2
    V_EM[10] = -alpha * planck**3 * mu_p * mu_n * F_t_r3 / (4 * M_p * M_n)
    # V_MM(pp)_part3
    V_EM[11] = -alpha * planck**3 * (4 * mu_p - 1) * F_ls_r3 / (2 * M_p**2)
    # V_MM(nn)_part3  ==  V_EM(nn)
    V_EM[12] = 0
    # V_MM(np)_part3
    V_EM[13] = -alpha * planck**3 * mu_n * F_ls_r3 / (2 * M_n * M_r)
    return V_EM


# project the strong interaction potential given above from S, T, T_z states into an operator format with 18 terms
# arguments for AV18_compoent
# r:      input separation in fm
# V_nn():  output potential in MeV (18 component array)
# ----------------------------------------------------------------------
# order of operators i in V_nn(i):
# i:    1=1                              2=t1.t2
#       3=s1.s2                          4=(s1.s2)(t1.t2)
#       5=S12 [=3(s1.r)(s2.r)-s1.s2]     6=S12(t1.t2)
#       7=L.S                            8=L.S(t1.t2)
#       9=L**2                          10=L**2(t1.t2)
#      11=L**2(s1.s2)                   12=L**2(s1.s2)(t1.t2)
#      13=(L.S)**2                      14=(L.S)**2(t1.t2)
#      15=T12 [3*t1z*t2z-t1.t2]        16=(s1.s2)T12
#      17=S12*T12                       18=t1z+t2z
def AV18_compoent(r):
    r = max(r, small)
    V_nn = np.zeros(18, dtype=np.float64)
    M_pi = (M_pi0 + 2.0 * M_pi1) / 3.0
    mu_pi0 = M_pi0 / planck
    mu_pi1 = M_pi1 / planck
    mu = M_pi / planck
    f_square = 0.075
    c = 2.1
    r0 = 0.5
    a1 = 5.0
    x = mu * r
    x0 = mu_pi0 * r
    x1 = mu_pi1 * r
    # Y_mur and T_mur
    Y_mu_r = np.exp(-x) * (1 - np.exp(-c * r**2)) / x
    T_mu_r = (1 + (3 + 3 / x) / x) * Y_mu_r * (1 - np.exp(-c * r**2))
    # v_pi(M_pi0)
    Y_mu_pi0 = (
        (M_pi0 / M_pi1) ** 2
        * (M_pi0 / 3)
        * np.exp(-x0)
        * (1 - np.exp(-c * r**2))
        / x0
    )
    T_mu_pi0 = (1 + (3 + 3 / x0) / x0) * Y_mu_pi0 * (1 - np.exp(-c * r**2))
    # v_pi(M_pi1)
    Y_mu_pi1 = (M_pi1 / 3) * np.exp(-x1) * (1 - np.exp(-c * r**2)) / x1
    T_mu_pi1 = (1 + (3 + 3 / x1) / x1) * Y_mu_pi1 * (1 - np.exp(-c * r**2))
    # v_pi_pp or nn
    Y_mu_pi0 = f_square * Y_mu_pi0
    Y_mu_pi1 = f_square * Y_mu_pi1
    T_mu_pi0 = f_square * T_mu_pi0
    T_mu_pi1 = f_square * T_mu_pi1
    T_mu_square = T_mu_r * T_mu_r
    W_r = 1 / (1 + np.exp((r - r0) * a1))
    W_r0 = 1 / (1 + np.exp(-r0 * a1))
    wsp = W_r * (1 + a1 * np.exp(-r0 * a1) * W_r0 * r)
    wsx = W_r * x
    wsx2 = wsx * x
    # calculate Q_i
    dY_mu_pi0_r0 = (M_pi0 / M_pi1) ** 2 * (M_pi0 / 3) * c / mu_pi0
    dY_mu_pi1_r0 = (M_pi1 / 3) * c / mu_pi1
    Y_mu_pi0p = Y_mu_pi0 - f_square * dY_mu_pi0_r0 * W_r * r / W_r0
    Y_mu_pi1p = Y_mu_pi1 - f_square * dY_mu_pi1_r0 * W_r * r / W_r0
    Y_mu_r = (Y_mu_pi0 + 2 * Y_mu_pi1) / 3
    T_mu_r = (T_mu_pi0 + 2 * T_mu_pi1) / 3
    # intermediate and short-range phenomenological part of the potential
    # date from : doi = "10.1103/PhysRevC.51.38"
    #   Channel  type   I_i                     P_i(i not eq t)      R_i               Q_i
    v_11_pp_c = -7.62701 * T_mu_square + 1815.4920 * wsp + 1847.8059 * wsx2 + Y_mu_pi0p
    v_11_np_c = (
        -7.62701 * T_mu_square
        + 1813.5315 * wsp
        + 1847.8059 * wsx2
        - Y_mu_pi0p
        + 2 * Y_mu_pi1p
    )
    v_11_nn_c = -7.62701 * T_mu_square + 1811.5710 * wsp + 1847.8059 * wsx2 + Y_mu_pi0p
    v_11_pp_t = 1.07985 * T_mu_square - 190.0949 * wsx - 811.2040 * wsx2 + T_mu_pi0
    v_11_np_t = (
        1.07985 * T_mu_square
        - 190.0949 * wsx
        - 811.2040 * wsx2
        - T_mu_pi0
        + 2 * T_mu_pi1
    )
    v_11_nn_t = 1.07985 * T_mu_square - 190.0949 * wsx - 811.2040 * wsx2 + T_mu_pi0
    v_11_ls = -0.62697 * T_mu_square - 570.5571 * wsp + 819.1222 * wsx2
    v_11_l2 = 0.06709 * T_mu_square + 342.0669 * wsp - 615.2339 * wsx2
    v_11_ls2 = 0.74129 * T_mu_square + 9.3418 * wsp - 376.4384 * wsx2
    v_10_c = (
        -8.62770 * T_mu_square
        + 2605.2682 * wsp
        + 441.9733 * wsx2
        - Y_mu_pi0p
        - 2 * Y_mu_pi1p
    )
    v_10_t = (
        1.485601 * T_mu_square
        - 1126.8359 * wsx
        + 370.1324 * wsx2
        - T_mu_pi0
        - 2 * T_mu_pi1
    )
    v_10_ls = 0.10180 * T_mu_square + 86.0658 * wsp - 356.5175 * wsx2
    v_10_l2 = -0.13201 * T_mu_square + 253.4350 * wsp - 1.0076 * wsx2
    v_10_ls2 = 0.07357 * T_mu_square - 217.5791 * wsp + 18.3935 * wsx2
    v_01_pp_c = -11.27028 * T_mu_square + 3346.6874 * wsp - 3 * Y_mu_pi0p
    v_01_np_c = (
        -10.66788 * T_mu_square + 3126.5542 * wsp - 3 * (-Y_mu_pi0p + 2 * Y_mu_pi1p)
    )
    v_01_nn_c = -11.27028 * T_mu_square + 3342.7664 * wsp - 3 * Y_mu_pi0p
    v_01_l2 = 0.12472 * T_mu_square + 16.7780 * wsp
    v_00_c = -2.09971 * T_mu_square + 1204.4301 * wsp - 3 * (-Y_mu_pi0p - 2 * Y_mu_pi1p)
    v_00_l2 = -0.31452 * T_mu_square + 217.4559 * wsp
    # calculate compoent
    v_11_CI = (v_11_pp_c + v_11_nn_c + v_11_np_c) / 3
    v_11_CD = (0.5 * (v_11_pp_c + v_11_nn_c) - v_11_np_c) / 6
    v_11_CA = (v_11_pp_c - v_11_nn_c) / 4
    v_11_t = (v_11_pp_t + v_11_nn_t + v_11_np_t) / 3
    v_11_CD_t = (0.5 * (v_11_pp_t + v_11_nn_t) - v_11_np_t) / 6
    v_11_CA_t = (v_11_pp_t - v_11_nn_t) / 4
    v_01_CI = (v_01_pp_c + v_01_nn_c + v_01_np_c) / 3
    v_01_CD = (0.5 * (v_01_pp_c + v_01_nn_c) - v_01_np_c) / 6
    v_01_CA = (v_01_pp_c - v_01_nn_c) / 4
    # v_1
    V_nn[0] = 0.0625 * (9 * v_11_CI + 3 * v_10_c + 3 * v_01_CI + v_00_c)
    # v_tau
    V_nn[1] = 0.0625 * (3 * v_11_CI - 3 * v_10_c + v_01_CI - v_00_c)
    # v_sigma
    V_nn[2] = 0.0625 * (3 * v_11_CI + v_10_c - 3 * v_01_CI - v_00_c)
    # v_sigma_tau
    V_nn[3] = 0.0625 * (v_11_CI - v_10_c - v_01_CI + v_00_c)
    # v_t
    V_nn[4] = 0.25 * (3 * v_11_t + v_10_t)
    # v_t_tau
    V_nn[5] = 0.25 * (v_11_t - v_10_t)
    # v_LS
    V_nn[6] = 0.25 * (3 * v_11_ls + v_10_ls)
    # v_LS_tau
    V_nn[7] = 0.25 * (v_11_ls - v_10_ls)
    # v_L^2
    V_nn[8] = 0.0625 * (9 * v_11_l2 + 3 * v_10_l2 + 3 * v_01_l2 + v_00_l2)
    # v_L^2_tau
    V_nn[9] = 0.0625 * (3 * v_11_l2 - 3 * v_10_l2 + v_01_l2 - v_00_l2)
    # v_L^2_sigma
    V_nn[10] = 0.0625 * (3 * v_11_l2 + v_10_l2 - 3 * v_01_l2 - v_00_l2)
    # v_L^2_sigma_tau
    V_nn[11] = 0.0625 * (v_11_l2 - v_10_l2 - v_01_l2 + v_00_l2)
    # v_LS^2
    V_nn[12] = 0.25 * (3 * v_11_ls2 + v_10_ls2)
    # v_LS^2_tau
    V_nn[13] = 0.25 * (v_11_ls2 - v_10_ls2)
    # v_T
    V_nn[14] = 0.25 * (3 * v_11_CD + v_01_CD)
    # v_sigma_T
    V_nn[15] = 0.25 * (v_11_CD - v_01_CD)
    # v_t_T
    V_nn[16] = v_11_CD_t
    # v_t1z+t2z
    V_nn[17] = v_01_CA
    return V_nn


# With the 18 terms of the AV18 NN-potential, we can form
# the central term: V_c
# the tensor term:  V_t
# the LS terms:     V_LS
# the L^2 term:     V_L2
# the (LS)^2 term:  V_LS2
# ---------------------------------------------
# argument for AV18_total
# l:    Orbital angular momentum quantum number.
# s:    Spin quantum number.
# j:    Total angular momentum quantum number.
# t:    Isospin quantum number.
# t1z:  Projection of isospin for particle p.
# t2z:  Projection of isospin for particle n.
# V_output() : return potential in MeV (2 x 2) matrix
# ---------------------------------------------
# order of terms in V_output(2x2):
#     V_output = [V          sqrt(8)V_t
#                 sqrt(8)V_t     V(l+2)]
# V includes all strong and EM terms
# V_t indicates spin tensor term
def AV18_total(l, s, j, t, t1z, t2z, r):
    V_output = np.zeros((2, 2), dtype=np.float64)
    # Strong interaction terms
    V_nn = AV18_compoent(r)
    S1_dot_S2 = 4 * s - 3
    tau1_dot_tau2 = 4 * t - 3
    T_12 = 3 * t1z * t2z - tau1_dot_tau2

    V_c = (
        V_nn[0]
        + tau1_dot_tau2 * V_nn[1]
        + S1_dot_S2 * V_nn[2]
        + S1_dot_S2 * tau1_dot_tau2 * V_nn[3]
        + T_12 * V_nn[14]
        + S1_dot_S2 * T_12 * V_nn[15]
        + (t1z + t2z) * V_nn[17]
    )
    V_t = V_nn[4] + tau1_dot_tau2 * V_nn[5] + T_12 * V_nn[16]
    V_LS = V_nn[6] + tau1_dot_tau2 * V_nn[7]
    V_L2 = (
        V_nn[8]
        + tau1_dot_tau2 * V_nn[9]
        + S1_dot_S2 * V_nn[10]
        + S1_dot_S2 * tau1_dot_tau2 * V_nn[11]
    )
    V_LS2 = V_nn[12] + tau1_dot_tau2 * V_nn[13]
    # Electromagnetic terms
    V_EM = V_EM_NN(r)
    # nn
    if t1z + t2z < 0:
        V_c = V_c + S1_dot_S2 * V_EM[6]
        V_t = V_t + V_EM[9]
    # np
    elif t1z + t2z == 0:
        V_c = V_c + V_EM[4] + S1_dot_S2 * V_EM[7]
        V_t = V_t + V_EM[10]
        V_LS = V_LS + V_EM[13]
    # pp
    else:
        V_c = V_c + V_EM[0] + V_EM[1] + V_EM[2] + V_EM[3] + S1_dot_S2 * V_EM[5]
        V_t = V_t + V_EM[8]
        V_LS = V_LS + V_EM[11]
    # channel:   1:single channel   2:coupled channel
    # Calculate coefficients based on quantum numbers
    # (l, s, j, t, t1z, t2z)------>(L^2, S_12, L*S, (L*S)**2)
    # V = V_c + L^2 * V_L^2 + S_12 * V_T + L*S * V_LS + (L*S)^2 * V_LS2
    channel = 1
    if s == 1 and j > l:
        channel = 2
    if channel == 1:
        S_12 = 0.0
        if s == 1 and l == j:
            S_12 = 2.0
        if l == j + 1:
            S_12 = -2.0 * (j + 2) / (2 * j + 1)
        L_dot_S = (j * (j + 1) - l * (l + 1) - s * (s + 1)) / 2
        V_output[0, 0] = (
            V_c
            + S_12 * V_t
            + L_dot_S * V_LS
            + l * (l + 1) * V_L2
            + L_dot_S**2 * V_LS2
        )
        V_output[1, 0] = 0
        V_output[0, 1] = 0
        V_output[1, 1] = 0
    elif channel == 2:
        S_12 = -2.0 * (j - 1) / (2 * j + 1)
        S_12_t = np.sqrt(36.0 * j * (j + 1)) / (2 * j + 1)
        S_12_2 = -2.0 * (j + 2) / (2 * j + 1)
        L_dot_S = (j * (j + 1) - l * (l + 1) - s * (s + 1)) / 2
        L_dot_S_2 = (j * (j + 1) - (l + 2) * (l + 3) - s * (s + 1)) / 2
        # for Deuteron
        # V(l=0):
        #       V = V_c
        V_output[0, 0] = (
            V_c
            + S_12 * V_t
            + L_dot_S * V_LS
            + l * (l + 1) * V_L2
            + L_dot_S**2 * V_LS2
        )
        # sqrt(8)V_t
        V_output[1, 0] = S_12_t * V_t
        V_output[0, 1] = S_12_t * V_t
        # V(l=2) :
        #       V(l=2) = V_c - 2*V_t - 3*V_LS + 6*V_L2 + 9*V_LS2
        V_output[1, 1] = (
            V_c
            + S_12_2 * V_t
            + L_dot_S_2 * V_LS
            + (l + 2) * (l + 3) * V_L2
            + L_dot_S_2**2 * V_LS2
        )
    return V_output


def plot():
    l_input = 0
    s_input = 1
    j_input = 1
    t_input = 0
    t1z_input = 1
    t2z_input = -1

    r_input = np.linspace(0.0, 2.0, 200)
    V_total_array = []
    V_nn_array = []

    for r in r_input:
        V_total_out = AV18_total(
            l_input, s_input, j_input, t_input, t1z_input, t2z_input, r
        )
        V_nn_out = AV18_compoent(r)
        V_total_array.append(
            (
                r,
                V_total_out[0, 0],
                V_total_out[0, 1],
                V_total_out[1, 0],
                V_total_out[1, 1],
            )
        )
        V_nn_array.append((r, *V_nn_out))

    V_total_array = np.array(V_total_array)
    V_nn_array = np.array(V_nn_array)
    #    np.savetxt('vnn_data.txt', V_nn_array, fmt='%.2f')
    #    np.savetxt('V_out_data.txt', V_total_array, fmt='%.2f')

    vnn_labels = {
        1: r"$v_c$",
        2: r"$v_{\tau}$",
        3: r"$v_{\sigma}$",
        4: r"$v_{\sigma\tau}$",
        5: r"$v_t$",
        6: r"$v_{t\tau}$",
        7: r"$v_{l^2}$",
        8: r"$v_{l^2\tau}$",
        9: r"$v_{l^2\sigma}$",
        10: r"$v_{l^2\sigma\tau}$",
        11: r"$v_{ls}$",
        12: r"$v_{ls\tau}$",
        13: r"$v_{ls^2}$",
        14: r"$v_{ls^2\tau}$",
        15: r"$v_T$",
        16: r"$v_{\sigma T}$",
        17: r"$v_{tT}$",
        18: r"$v_{\tau_{1z} + \tau_{2z}}$",
    }
    V_out_labels = {
        1: r"${V_1}$",
        2: r"$\sqrt{8}V_T$",
        3: r"$\sqrt{8}V_T$",
        4: r"${V_2}$",
    }

    fig, axs = plt.subplots(2, 1, figsize=(12, 16))
    # V1, V_T, V_2
    for i in range(1, len(V_out_labels) + 1):
        axs[0].plot(V_total_array[:, 0], V_total_array[:, i], label=V_out_labels[i])
    for ax in axs:
        ax.set_xlim(0, 2)
        ax.set_ylim(-400, 2500)
        ax.set_xlabel("r(fm)")
        ax.set_ylabel("V(MeV)")
        ax.grid(True)
    axs[0].set_title("AV18 potential in Deuteron")
    axs[0].tick_params(axis="both", direction="in", length=6, width=1)
    axs[0].legend()
    inner_ax = axs[0].inset_axes([0.35, 0.35, 0.55, 0.55])
    for i in range(1, len(V_out_labels) + 1):
        inner_ax.plot(V_total_array[:, 0], V_total_array[:, i], label=V_out_labels[i])
    inner_ax.set_xlim(0, 2)
    inner_ax.set_ylim(-400, 100)
    inner_ax.tick_params(axis="both", direction="in", length=6, width=1)
    inner_ax.grid(True)
    # AV18 Compoent(18 terms)
    for i in range(1, len(vnn_labels) + 1):
        axs[1].plot(V_nn_array[:, 0], V_nn_array[:, i], label=vnn_labels[i])
    axs[1].set_xlim(0, 2)
    axs[1].set_ylim(-200, 200)
    axs[1].set_xlabel("r(fm)")
    axs[1].set_ylabel("V(MeV)")
    axs[1].tick_params(axis="both", direction="in", length=6, width=1)
    axs[1].set_title("AV18 Compoent")
    axs[1].legend(loc="upper right", bbox_to_anchor=(1.0, 1.0), ncol=3)
    axs[1].grid(True)
    plt.savefig("Fig-AV18.png")
    plt.show()


if __name__ == "__main__":
    plot()
