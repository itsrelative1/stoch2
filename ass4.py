import concurrent.futures
import numpy as np
from scipy.stats import norm

T = 1
r = -0.0027*10
# r = -0.0027
sigma = 0.21*np.sqrt(10)
# sigma = 0.21
s0 = 715
K = 715
n = 2

def StockPrice(s0, T, sigma, r, n):

    prices = [s0]

    dt = T / n

    # Calculates the stock price intervals
    for i in range(n):
        z = np.random.normal(0, 1)
        price = prices[i] + prices[i] * (dt * r + sigma * z * np.sqrt(dt))
        if price < 0:
            price = 0
        prices.append(price)

    # calculates geom abverage
    product = np.prod(prices)
    geom_avg = product**(1/3)

    return geom_avg

def PayOff(K, S):
    return max(K-S, 0)

def MC_trial(s0, K, T, sigma, r, n):
    payoffs = []
    for i in range(100000):
        S = StockPrice(s0, T, sigma, r, n)
        payoffs.append(PayOff(K, S))

    return payoffs

def Theoretical(s0, K, T, sigma, r):
    a = r/2 - (sigma**2)/4
    b = np.sqrt(5/18)*sigma

    d2 = (np.log(K/s0) - a*T)/(b*np.sqrt(T))
    d1 = d2 - b*np.sqrt(T)

    C = np.exp(-r*T)*(K*norm.cdf(d2) - s0*np.exp((a+0.5*b**2)*T)*norm.cdf(d1))

    return C

if __name__ == "__main__":
    payoffs = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        values = [executor.submit(MC_trial, s0, K, T, sigma, r, n) for _ in range(10)]
        for f in concurrent.futures.as_completed(values):
            payoffs.append(f.result())

    payoffs = np.array(payoffs)
    payoffs = np.hstack(payoffs)

    OptionPrice = np.exp(-r*T)*np.mean(payoffs)
    stderr = (np.exp(-r*T)*np.std(payoffs))/np.sqrt(len(payoffs))

    conf_interval = (OptionPrice - 1.96*stderr, OptionPrice + 1.96*stderr)

    print(OptionPrice, stderr, conf_interval)
    print(Theoretical(s0, K, T, sigma, r))
    # print(analytical(K, s0, sigma, r, T))
