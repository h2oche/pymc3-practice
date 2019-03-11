import pymc3 as pm
from IPython.core.pylabtools import figsize
import matplotlib.pyplot as plt
import scipy.stats as stats
figsize(12, 4)

true_p_A = 0.05
true_p_B = 0.04

#generate data
N_A = 1000
N_B = 1000
obsdata_A = stats.bernoulli.rvs(true_p_A, size=N_A)
obsdata_B = stats.bernoulli.rvs(true_p_B, size=N_B)

#print(np.mean(obs_A))
#print(np.mean(obs_B))

#set up models
with pm.Model() as model:
  #model def
  p_A = pm.Uniform("p_A", 0, 1)
  p_B = pm.Uniform("p_B", 0, 1)
  delta = pm.Deterministic("delta", p_A - p_B)

  #make observation
  obs_A = pm.Bernoulli("obs_A", p_A, observed = obsdata_A)
  obs_B = pm.Bernoulli("obs_B", p_B, observed = obsdata_B)

  #inference
  step = pm.Metropolis()
  trace = pm.sample(20000, step=step)
  burned_trace = trace[1000:]

p_A_samples = burned_trace["p_A"]
p_B_samples = burned_trace["p_B"]
delta_samples = burned_trace["delta"]

#plot data
figsize(12.5, 10)
ax = plt.subplot(311)

plt.xlim(0, .1)
plt.hist(p_A_samples, histtype='stepfilled', bins=25, alpha=0.85,
         label="posterior of $p_A$", color="#A60628", normed=True)
plt.vlines(true_p_A, 0, 80, linestyle="--", label="true $p_A$ (unknown)")
plt.legend(loc="upper right")
plt.title("Posterior distributions of $p_A$, $p_B$, and delta unknowns")

ax = plt.subplot(312)

plt.xlim(0, .1)
plt.hist(p_B_samples, histtype='stepfilled', bins=25, alpha=0.85,
         label="posterior of $p_B$", color="#467821", normed=True)
plt.vlines(true_p_B, 0, 80, linestyle="--", label="true $p_B$ (unknown)")
plt.legend(loc="upper right")

ax = plt.subplot(313)
plt.hist(delta_samples, histtype='stepfilled', bins=30, alpha=0.85,
         label="posterior of delta", color="#7A68A6", normed=True)
plt.vlines(true_p_A - true_p_B, 0, 60, linestyle="--",
           label="true delta (unknown)")
plt.vlines(0, 0, 60, color="black", alpha=0.2)
plt.legend(loc="upper right")
plt.show()