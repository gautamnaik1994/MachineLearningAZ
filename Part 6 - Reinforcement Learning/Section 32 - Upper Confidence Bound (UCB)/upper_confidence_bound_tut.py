import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

dataset  = pd.read_csv('./Ads_CTR_Optimisation.csv')

N=10000
d=10
numbers_of_selection = [0] * d
sums_of_rewards = [0] * d
ads_seleted = []
total_reward= 0
for n in  range(0,N):
    max_upper_bound=0
    ad = 0
    for i in range(0,d):
        if(numbers_of_selection[i]>0):
            average_reward = sums_of_rewards[i]/numbers_of_selection[i]
            delta_i = math.sqrt(3/2 * math.log(n+1)/numbers_of_selection[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound>max_upper_bound:
            max_upper_bound= upper_bound
            ad = i
    ads_seleted.append(ad)
    numbers_of_selection[ad]=numbers_of_selection[ad] + 1
    reward = dataset.values[n,ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward

plt.hist(ads_seleted)
plt.title("Histogram of ads seletion")
plt.xlabel("Ads")
plt.ylabel("No of times each ad was seleted")
plt.show()

