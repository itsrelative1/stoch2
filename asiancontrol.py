import numpy as np
import matplotlib.pyplot as plt
from asianput import payoff, path
from scipy.stats import norm
from scipy.stats.mstats import gmean

def I(s0, sigma, r, T, m) -> tuple:
    """
    Calculates the average of the stock path
    :param s0: The initial stockprice
    :param sigma: The volatility
    :param r: Interest free rate
    :param T: Time t=T in the future
    :param m: Number of points in the stock path
    :return: It and also S_0, S_0.5 and S_T to calculate HT for the control variate
    """
    St = path(s0, sigma, r, T, m)
    inner_sum = 0
    dt = T / m

    for i in range(len(St) - 1):
        inner_sum += (St[i] + St[i + 1]) / 2 * dt

    mean = gmean(St)

    return mean, St[0], St[round(len(St) / 2)], St[-1], inner_sum / T


# def analytical(K, s0, sigma, r, T):
#     d1 = (np.log(s0 / K) + T * ((r / 2 - sigma ** 2 / 4) + 5 / 36 * sigma ** 2)) / (
#         np.sqrt(T) * sigma * np.sqrt(5 / 18)
#     )

#     d2 = (np.log(s0 / K) + T * (r / 2 - sigma ** 2 / 4)) / (
#         np.sqrt(T) * sigma * np.sqrt(5 / 18)
#     )

#     return np.exp(-r * T) * (
#         K * norm.cdf(-d2) - s0 * norm.cdf(-d1) * np.exp(T * (r / 2 - sigma ** 2 / 4))
#     )


def analytical(K, s0, sigma, r, T):
    a = r / 2 - (sigma ** 2) / 4
    b = np.sqrt(5 / 18) * sigma

    d2 = (np.log(K / s0) - a * T) / (b * np.sqrt(T))
    d1 = d2 - b * np.sqrt(T)

    C = np.exp(-r * T) * (
        K * norm.cdf(d2) - s0 * np.exp((a + 0.5 * b ** 2) * T) * norm.cdf(d1)
    )

    return C


def simulate(n, K, s0, sigma, r, T, m):
    G_T_list = []
    C_T_list = []

    for i in range(n):
        mean, S0, S_half, S_T, I_T = I(s0, sigma, r, T, m)
        G_T_list.append(np.exp(-r * T) * payoff(K, I_T))
        H_T = (s0 * S_half * S_T) ** (1 / 3)
        C_T_list.append(np.exp(-r * T) * max(K - H_T, 0))

    G_0 = np.array(G_T_list).mean()
    C_0 = np.array(C_T_list).mean()
    beta = np.cov(G_T_list, C_T_list)[0][1] / np.var(C_T_list)
    expectation = analytical(K, s0, sigma, r, T)
    G_new = G_0 - beta * C_0 + beta * expectation

    G_0_std = np.sqrt(
        np.var(G_T_list)
        + beta ** 2 * np.var(C_T_list)
        - 2 * beta * np.cov(G_T_list, C_T_list)[0][1]
    ) / np.sqrt(n)

    confidence = [G_new - 1.96 * G_0_std, G_new + 1.96 * G_0_std]

    return G_new, G_0_std, confidence


g = simulate(10000, 715, 715, 0.21, -0.0027, 10, 100)
print("G0_mean=", g[0], "G0_std=", g[1])
print("Confidence interval G_0 of 95%", g[2])
