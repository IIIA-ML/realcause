from loading import load_gen
from data.ihdp import load_ihdp_tri
import numpy as np
import pandas as pd

model, args = load_gen()
w, t, y = model.sample()

d = load_ihdp_tri()

t_0 = np.zeros((493, 1))
t_1 = np.ones((493, 1))
t_2 = np.ones((493, 1))

y_0m = model.sample_interventional(t_0)
y_1m = model.sample_interventional(t_1)
y_2m = model.sample_interventional(t_2)

y_0r = d['y_0']
y_1r = d['y_1']
y_2r = d['y_2']

t_y = pd.DataFrame({'y': d['y'].flatten(), 't': d['t'].flatten()})

print('#################')

print('True')

print(('y_0m mean ' , y_0m.mean()))
print(('y_0r mean ' , y_0r.mean()))

print(('y_1m mean ' , y_1m.mean()))
print(('y_1r mean ' , y_1r.mean()))

print(('y_2m mean ' , y_2m.mean()))
print(('y_2r mean ' , y_2r.mean()))

print('#################')

print('Observational')

print(('y_t0m mean ' , y[t==0].mean()))
print(('y_t0r mean ' , t_y.loc[t_y['t']==0, 'y'].mean()))

print(('y_t1m mean ' , y[t==1].mean()))
print(('y_t1r mean ' , t_y.loc[t_y['t']==1, 'y'].mean()))

print(('y_t2m mean ' , y[t==2].mean()))
print(('y_t2r mean ' , t_y.loc[t_y['t']==2, 'y'].mean()))
