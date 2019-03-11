import pymc3 as pm
import theano.tensor as tt
import numpy as np

with pm.Model() as model:
  param = pm.Exponential("possion_param", 1.0)
  data_gen = pm.Poisson("data_generator", param)

with model:
  data_gen_one = data_gen +1
  print(param.tag.test_value)
  print(data_gen.tag.test_value)
  print(data_gen_one.tag.test_value)

with pm.Model() as ab_testing:
  p_A = pm.Uniform("P(A)", 0, 1)
  p_B = pm.Uniform("P(B)", 0, 1)

with pm.Model() as stochastic:
  betas = pm.Uniform("betas", 0, 1, shape=10)
  print(betas.tag.test_value)

with pm.Model() as deterministc:
  param1 = pm.Exponential("param1", 1.0)
  param2 = pm.Exponential("param2", 2.0)

  def add(x, y):
    return x+y

  deter1 = param1 + param2
  deter2 = pm.Deterministic("add_p1_p2", add(param1, param2))

#numpy : direct calculation
#theano : make compute graph, lazy calculation
with pm.Model() as theano_test:
  p1 = pm.Uniform("p1", 0, 1)
  p2 = 1 - p1
  p = tt.stack([p1,p2])
  assignment = pm.Categorical("assignment", p)

with pm.Model() as data:
  data = np.array([10, 5])
  fixed_var = pm.Poisson("fixed", 1, observed=data)
  print(fixed_var.tag.test_value)

  