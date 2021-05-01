import numpy as np
import matplotlib.pyplot as plt
from asianput import payoff, path


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

    return St[0], St[round(len(St) / 2)], St[-1], inner_sum / T


def simulate(n, K, s0, sigma, r, T, m):
    G_T_list = []
    C_T_list = []

    for i in range(n):
        S0, S_half, S_T, I_T = I(s0, sigma, r, T, m)
        G_T_list.append(payoff(K, I_T))
        H_T = (S0 * S_half * S_T) ** (1 / 3)
        C_T_list.append(max(K - H_T, 0))

    G_0 = np.exp(-r * T) * np.array(G_T_list).mean()
    C_0 = np.exp(-r * T) * np.array(C_T_list).mean()
    beta = np.cov(G_T_list, C_T_list) / np.var(C_T_list)

    G_new = G_0 - beta * C_0
