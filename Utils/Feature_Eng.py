import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Investigating the strength of the association using Cramers-V Test
# Outlet_Size Outlet_Location_Type Outlet_Type

def cramers_v(var1, var2):
    cross_tab = np.array(pd.crosstab(var1, var2, rownames=None, colnames=None))
    stat = chi2_contingency(cross_tab)[0]
    obs = np.sum(cross_tab)
    dof = min(cross_tab.shape)-1
    return (stat/obs*dof)