#%%
import numpy as np
import joblib
allV1=joblib.load("stg_drp1.pkl")
allV2=joblib.load("stg_drp2.pkl")
#%%
v1mat=np.hstack([allV1])
v2mat=np.hstack([allV2]).transpose()
# %%
plt.boxplot(v2mat)
# %%
