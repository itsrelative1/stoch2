import concurrent.futures
import numpy as np
from scipy.stats import norm
from scipy.stats.mstats import gmean

# Parameters
T = 1
r = -0.0027*10
sigma = 0.21*np.sqrt(10)
s0 = 715
K = 715
n = 100

def StockPrice(s0, T, sigma, r, n):
    """
    Uses Euler's Scheme to create a stock price path and calculates the geometric average based on 3 points
    :param s0: The initial stock price
    :param T: Maturity time
    :param sigma: Volatility
    :param r: risk-free Interest rate
    :param n: Number of points in the stock path
    :return: Geometric average of the stock price path based on the points S_0, S_0.5T and S_T
    """
    prices = [s0]

    dt = T / n

    # Calculates the stock price intervals
    for i in range(n):
        z = np.random.normal(0, 1)
        price = prices[i] + prices[i] * (dt * r + sigma * z * np.sqrt(dt))

        # stock price cant go below 0
        if price < 0:
            price = 0
        prices.append(price)

    # Takes S_0, S_0.5T and S_t
    new_price = [prices[0], prices[49], prices[99]]

    # calculates geom abverage
    product = np.prod(new_price)
    geom_avg = product**(1/3)

    return geom_avg

def PayOff(K, S):
    """
    Put option payoff
    :param K: Strike price
    :param S: Stock price
    :return: Option pay off
    """
    return max(K-S, 0)

def MC_trial(s0, K, T, sigma, r, n):
    """
    Uses Euler's Scheme to create a stock price path and calculates the geometric average based on 3 points
    :param s0: The initial stock price
    :param K: Strike price
    :param T: Maturity time
    :param sigma: Volatility
    :param r: risk-free Interest rate
    :param n: Number of payoffs
    :return: List of 10^5 pay off realizations
    """
    payoffs = []

    # creates 10^5 payoff realizations
    for i in range(100000):
        S = StockPrice(s0, T, sigma, r, n)
        payoffs.append(PayOff(K, S))

    return payoffs

def Theoretical(s0, K, T, sigma, r):
    """
    Calculates the analytical 3-point geometric average Asian option price
    :param s0: The initial stock price
    :param K: Strike price
    :param T: Maturity time
    :param sigma: Volatility
    :param r: risk-free Interest rate
    :return: No-arbitrage option price
    """

    a = r/2 - (sigma**2)/4
    b = np.sqrt(5/18)*sigma

    d2 = (np.log(K/s0) - a*T)/(b*np.sqrt(T))
    d1 = d2 - b*np.sqrt(T)

    C = np.exp(-r*T)*(K*norm.cdf(d2) - s0*np.exp((a+0.5*b**2)*T)*norm.cdf(d1))

    return C

if __name__ == "__main__":
    payoffs = []

    # Splits 10 10^5 MC payoff realizations over 10 cores granting 10^6 realizations in total
    with concurrent.futures.ProcessPoolExecutor() as executor:
        values = [executor.submit(MC_trial, s0, K, T, sigma, r, n) for _ in range(10)]
        for f in concurrent.futures.as_completed(values):
            payoffs.append(f.result())

    # Combines results
    payoffs = np.array(payoffs)
    payoffs = np.hstack(payoffs)

    # Determines option price based on discounted mean of simulated the payoffs
    OptionPrice = np.exp(-r*T)*np.mean(payoffs)
    # Determines Standard error
    stderr = (np.exp(-r*T)*np.std(payoffs))/np.sqrt(len(payoffs))
    # Determines 95% confidence interval
    conf_interval = (OptionPrice - 1.96*stderr, OptionPrice + 1.96*stderr)

    print(OptionPrice, stderr, conf_interval)
    print(Theoretical(s0, K, T, sigma, r))
