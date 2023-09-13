import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.special import sph_harm
from scipy import integrate
from AV18potential import AV18_total


def g11(r, E):
    Vr_array = []
    for r_val in r:
        Vr = AV18_total(0, 1, 1, 0, 1, -1, r_val)
        Vr_array.append(Vr[0, 0])
    V1 = np.array(Vr_array)
    g11 = (V1 - E) / a
    return g11


def g12(r):
    Vr_array = []
    for r_val in r:
        Vr = AV18_total(0, 1, 1, 0, 1, -1, r_val)
        Vr_array.append(Vr[0, 1])
    VT = np.array(Vr_array)
    g12 = VT / a
    return g12


def g21(r):
    Vr_array = []
    for r_val in r:
        Vr = AV18_total(0, 1, 1, 0, 1, -1, r_val)
        Vr_array.append(Vr[1, 0])
    VT = np.array(Vr_array)
    g21 = VT / a
    return g21


def g22(r, E):
    r_strange_point = 1e-3
    Vr_array = []
    r_squared_terms = []
    for r_val in r:
        r_adjust = max(r_val, r_strange_point)
        Vr = AV18_total(0, 1, 1, 0, 1, -1, r_adjust)
        Vr_array.append(Vr[1, 1])
        r_squared_terms.append(6 * a / ((r_adjust) ** 2))
    Vr_values = np.array(Vr_array)
    r_squared_terms = np.array(r_squared_terms)
    g22 = (r_squared_terms + Vr_values - E) / a
    return g22


# This function will return 4 results(u,u',w,w')
# input parameter: r(fm), E(MeV), u1(u[1], w1(w[1]))  the last two were initial of u(first step) and w(first step)
def calculate_wave_function(r, E, u1, w1):
    u = np.zeros(n + 1)
    w = np.zeros(n + 1)
    u_d = np.zeros(n + 1)
    w_d = np.zeros(n + 1)
    G11 = g11(r, E)
    G12 = g12(r)
    G21 = g21(r)
    G22 = g22(r, E)
    k = np.zeros(n + 1)
    h = np.zeros(n + 1)
    p = np.zeros(n + 1)
    q = np.zeros(n + 1)
    u[1] = u1
    w[1] = w1

    # Calculate the solution using the loop
    for i in range(1, n):
        k[i - 1] = 1 - G11[i - 1] * d2
        h[i - 1] = 1 - G22[i - 1] * d2
        p[i - 1] = G12[i - 1] * w[i - 1]
        q[i - 1] = G21[i - 1] * u[i - 1]
        k[i] = 1 - G11[i] * d2
        h[i] = 1 - G22[i] * d2
        p[i] = G12[i] * w[i]
        q[i] = G21[i] * u[i]
        k[i + 1] = 1 - G11[i + 1] * d2
        h[i + 1] = 1 - G22[i + 1] * d2

        A1 = (12 - 10 * k[i]) * u[i] - k[i - 1] * u[i - 1] + d2 * (10 * p[i] + p[i - 1])
        A2 = G12[i + 1] * d2
        B1 = (12 - 10 * h[i]) * w[i] - h[i - 1] * w[i - 1] + d2 * (10 * q[i] + q[i - 1])
        B2 = G21[i + 1] * d2

        u[i + 1] = (A1 * h[i + 1] + A2 * B1) / (k[i + 1] * h[i + 1] - A2 * B2)
        w[i + 1] = (B1 * k[i + 1] + A1 * B2) / (k[i + 1] * h[i + 1] - A2 * B2)
        u_d[i] = (u[i + 1] - u[i - 1]) / 2 / delta
        w_d[i] = (w[i + 1] - w[i - 1]) / 2 / delta
    return u, u_d, w, w_d


# this function returns 4 functions
# Fs: r--->Infinity  s wave function u
# Fd  r--->Infinity  d wave function w
# Fs_d and Fd_d were Fs' and Fd'
def fs_fd(r, E):
    s = abs(E) / a
    r_strange_point = 1e-3
    func_s = []
    func_d = []
    func_s_d = []
    func_d_d = []
    for r_val in r:
        r_adjust = max(r_val, r_strange_point)
        fs = np.exp(-np.sqrt(s) * r_adjust)
        fd = np.exp(-np.sqrt(s) * r_adjust) * (
            1 + 3 / (r_adjust * np.sqrt(s)) + 3 / (r_adjust**2 * s)
        )
        fs_d = -np.sqrt(s) * fs
        fd_d = -np.sqrt(s) * np.exp(-np.sqrt(s) * r_adjust) * (
            1 + 3 / (r_adjust * np.sqrt(s)) + 3 / (r_adjust**2 * s)
        ) + np.exp(-np.sqrt(s) * r_adjust) * (
            -3 / (r_adjust**2 * np.sqrt(s)) - 6 / (r_adjust**3 * s)
        )
        func_s.append(fs)
        func_d.append(fd)
        func_s_d.append(fs_d)
        func_d_d.append(fd_d)
    Fs = np.array(func_s)
    Fd = np.array(func_d)
    Fs_d = np.array(func_s_d)
    Fd_d = np.array(func_d_d)
    return Fs, Fd, Fs_d, Fd_d


def cal_matrix_A(r, E):
    Fs, Fd, Fs_d, Fd_d = fs_fd(r, E)
    u1, u1_d, w1, w1_d = calculate_wave_function(r, E, 0.1, 1)
    u2, u2_d, w2, w2_d = calculate_wave_function(r, E, 1, 0.1)
    # Create a new matrix for each r_val
    A = np.zeros((4, 4), dtype=np.float64)
    A[0, 0] = u1[-2]
    A[0, 1] = u2[-2]
    A[0, 2] = Fs[-1]
    A[0, 3] = 0
    A[1, 0] = w1[-2]
    A[1, 1] = w2[-2]
    A[1, 2] = 0.0
    A[1, 3] = Fd[-1]
    A[2, 0] = u1_d[-2]
    A[2, 1] = u2_d[-2]
    A[2, 2] = Fs_d[-1]
    A[2, 3] = 0.0
    A[3, 0] = w1_d[-2]
    A[3, 1] = w2_d[-2]
    A[3, 2] = 0.0
    A[3, 3] = Fd_d[-1]
    det_A = np.linalg.det(A)
    U, S, VT = np.linalg.svd(A)
    v = VT[-1]
    return v, det_A


# det(A)==0  -------->  Eigen Energy
def find_zero_det_E(r, E_min, E_max, tolerance=1e-6):
    # Perform binary search within the given range [E_min, E_max]
    while E_max - E_min > tolerance:
        E_mid = (E_min + E_max) / 2
        v, det_A = cal_matrix_A(r, E_mid)
        if abs(det_A) < tolerance:
            return E_mid
        if det_A < 0:
            E_min = E_mid
        else:
            E_max = E_mid
    return (E_min + E_max) / 2, v, det_A


def norm_uw():
    E, v, det_A = find_zero_det_E(r, E_min, E_max, tolerance=1e-6)
    u1, _, w1, _ = calculate_wave_function(r, E, 0.1, 1)
    u2, _, w2, _ = calculate_wave_function(r, E, 1, 0.1)
    u = -v[0] * u1 - v[1] * u2
    w = -v[0] * w1 - v[1] * w2
    sum_sq = np.sum(u**2) + np.sum(w**2)
    norm_constant_uw = sum_sq * delta
    u /= np.sqrt(norm_constant_uw)
    w /= np.sqrt(norm_constant_uw)
    return E, u, w


a = 41.4710793753
E_min = -2.5
E_max = -2
r0 = 0.0
delta = 0.01
d2 = delta**2 / 12.0
n = 500
r = np.linspace(r0, r0 + delta * n, n + 1)

E, u, w = norm_uw()


def fourier_transform(func_values, k, r_values):
    integrand = func_values * np.exp(-1j * k * r_values) * r_values**2
    result = np.trapz(integrand, r_values)
    normalization_factor = 1 / np.sqrt(2 * np.pi)
    return result * normalization_factor


k_values = np.linspace(0, 10, 200)

u = [fourier_transform(u, k, r) for k in k_values]
w = [fourier_transform(w, k, r) for k in k_values]

# 计算球谐函数
theta = np.pi / 2  # np.linspace(0, np.pi, n+1)
phi = np.pi  # np.linspace(0, 2 * np.pi, n+1)
theta, phi = np.meshgrid(theta, phi)
Y_00 = sph_harm(0, 0, phi, theta)
Y_21 = sph_harm(1, 2, phi, theta)
# 计算波函数 phi = u(r) * Y_lm + w(r) * Y_lm
phi = u * Y_00 + w * Y_21

# 计算傅里叶变换
phi_k_values = []
for k in k_values:
    phi_k = np.trapz(phi * np.exp(-1j * k * r) * r**2, r)
    phi_k_values.append(phi_k)

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(k_values, np.real(phi_k_values), label="Real Phi")
plt.plot(k_values, np.imag(phi_k_values), label="Imaginary Phi")
plt.xlabel("k")
plt.ylabel("Normalization Integral")
plt.title("Wave Function Phi vs Momentum k")
plt.legend()
plt.grid(True)
plt.show()
