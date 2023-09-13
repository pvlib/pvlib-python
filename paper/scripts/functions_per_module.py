
import pandas as pd
import numpy as np
import pvlib

# run in an environment with the target pvlib version installed

def recurse(module):
    objects = []
    for name in dir(module):
        if name.startswith("_"):
            continue
        obj = getattr(module, name)
        if (not hasattr(obj, '__module__') or not obj.__module__.startswith(module.__name__)) and \
           (not hasattr(obj, '__package__') or not obj.__package__.startswith(module.__name__)):
            continue
        if type(obj).__name__ == 'function':
            objects.append(obj.__module__ + "." + obj.__name__)
        if type(obj).__name__ == 'module':
            objects.extend(recurse(obj))
    return np.unique(objects)


names = recurse(pvlib)
df = pd.DataFrame({'name': names})
df['module'] = df['name'].str.split(".").str[1]
counts = df.groupby('module').count()

counts['name'].to_dict()

# %%

import matplotlib.pyplot as plt

counts_060 = {'atmosphere': 10, 'clearsky': 6, 'irradiance': 25, 'modelchain': 2, 'pvsystem': 22, 'singlediode': 6, 'solarposition': 16, 'spa': 51, 'tmy': 2, 'tools': 7, 'tracking': 1}
counts_0101 = {'atmosphere': 9, 'bifacial': 9, 'clearsky': 6, 'iam': 10, 'inverter': 6, 'iotools': 31, 'irradiance': 29, 'ivtools': 8, 'location': 1, 'modelchain': 2, 'pvsystem': 16, 'scaling': 2, 'shading': 4, 'singlediode': 5, 'snow': 4, 'soiling': 2, 'solarposition': 18, 'spa': 51, 'spectrum': 7, 'temperature': 11, 'tools': 9, 'tracking': 4}

df = pd.DataFrame({
    'v0.6.0 (2018-09-17)': pd.Series(counts_060),
    'v0.10.1 (2023-07-03)': pd.Series(counts_0101),
})

df = df.drop(['spa', 'tmy', 'tools'])

fig, ax = plt.subplots(figsize=(8, 4))
df.plot.bar(ax=ax)
ax.set_ylabel('Number of public functions')
ax.set_xlabel('pvlib module')
fig.tight_layout()
fig.savefig('functions_06_010.pdf')
