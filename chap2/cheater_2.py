import pymc3 as pm
import numpy as np
import theano.tensor as tt
from IPython.core.pylabtools import figsize
import matplotlib.pyplot as plt
import scipy.stats as stats

N = 100
X = 35
with pm.Model() as model:
  #model
  p = pm.Uniform("freq_cheating", 0, 1)
  p_skewed = pm.Deterministic("p_skewed", p*0.5 +0.25)
  #observation
  yes_responses = pm.Binomial("response", N, p_skewed, observed=X)
  #inference
  step = pm.Metropolis()
  trace = pm.sample(25000, step=step)
  burned_trace = trace[2500:]

figsize(12.5, 3)
p_trace = burned_trace["freq_cheating"]
plt.hist(p_trace, histtype="stepfilled", normed=True, alpha=0.85, bins=30, 
         label="posterior distribution", color="#348ABD")
plt.vlines([.05, .35], [0, 0], [5, 5], alpha=0.2)
plt.xlim(0, 1)
plt.legend()
plt.show()