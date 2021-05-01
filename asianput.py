import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def path(s0, sigma, r, T, m):
    path = np.zeros(m + 1)
    path[0] = s0
    dt = T / m

    for i in range(m):
        z = np.random.normal(0, 1)
        path[i + 1] = path[i] + path[i] * (dt * r + sigma * z * np.sqrt(dt))

    return path


def I(s0, sigma, r, T, m):
    St = path(s0, sigma, r, T, m)
    inner_sum = 0
    dt = T / m

    for i in range(len(St) - 1):
        inner_sum += (St[i] + St[i + 1]) / 2 * dt

    return inner_sum / T


def payoff(K, I):
    return max(0, K - I)


def simulate(n, K, s0, sigma, r, T, m):
    payoffs = []

    for i in tqdm(range(n)):
        I_T = I(s0, sigma, r, T, m)
        payoffs.append(payoff(K, I_T))

    G_0_mean = np.exp(-r * T) * np.array(payoffs).mean()
    G_0_std = np.sqrt(np.array(payoffs).var()) / np.sqrt(n)

    confidence = [G_0_mean - 1.96 * G_0_std, G_0_mean + 1.96 * G_0_std]

    return G_0_mean, G_0_std, confidence


G_0_mean, G_0_std, confidence = simulate(10000, 100, 100, 0.2, 0.06, 1, 100)

print("G0_mean=", G_0_mean, "G0_std =", G_0_std)
print("Confidence interval G_0 of 95%:", confidence)

