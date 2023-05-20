#%%
import joblib
import matplotlib.pyplot as plt
import numpy as np
A=[]
B=[]
C=[]
D=[]
for ii in range(4,11):
    [A1,B1,C1,D1]=joblib.load('msk_stlf{}.pkl'.format(ii))
    A.append(A1)
    B.append(B1)
    C.append(C1)
    D.append(D1)
cc1=np.hstack(A).transpose()
cc2=np.hstack(B).transpose()
cc3=np.hstack(C).transpose()
cc4=np.hstack(D).transpose()
plt.boxplot([cc1,cc2,cc3,cc4])

# %%
import joblib
import optuna
import plotly
study=joblib.load('study_nvida.pkl')
fig = optuna.visualization.plot_contour(study)
fig.show()

# %%
import tensorflow as tf
# %%
