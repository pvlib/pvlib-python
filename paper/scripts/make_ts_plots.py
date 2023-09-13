# -*- coding: utf-8 -*-
"""
Created on Wed May 31 19:38:26 2023

@author: ksande
"""

import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, r'C:\Users\ksande\projects\pv-foss-engagement\sphinx\source\project')

from utils import get_github_contributor_timeseries, get_github_stars

# %%

cumulative_contributors, _ = get_github_contributor_timeseries('pvlib/pvlib-python')

# the above API only reports up to 100 contributors, I think.  So add a few more, with (very) rough dates:
fudge = pd.Series([101, 102, 103, 104], index=pd.to_datetime(['2023-07-01', '2023-08-01', '2023-09-01', '2023-09-13']))

cumulative_contributors = cumulative_contributors.append(fudge).resample('d').ffill()

# %%

fn = r'C:\Users\ksande\Downloads\pvlib_google_group_2023-09-13.csv'
df = pd.read_csv(fn, index_col=0, parse_dates=True)

df.index = pd.to_datetime(df['Join Date'])
df['n'] = 1
gg_members = df['n'].resample('d').sum().cumsum().ffill()


# %%

gh = get_github_stars('pvlib/pvlib-python')

gh = gh.set_index('star_date')
gh['n'] = 1
gh_stars = gh['n'].resample('d').sum().cumsum().ffill()


# %%

fig, axes = plt.subplots(1, 2, figsize=(10, 3), sharex=True)

gg_members.plot(ax=axes[1], label='Forum registrations')
gh_stars.plot(ax=axes[1], label='GitHub stars')

cumulative_contributors.plot(ax=axes[0])
axes[0].set_ylabel('Repository contributors')

axes[1].legend()
axes[1].set_ylabel('Count')

axes[0].set_ylim(bottom=0)
axes[1].set_ylim(bottom=0)

axes[1].set_xlabel(None)

axes[0].tick_params(axis='x', which='minor', bottom=False)
axes[1].tick_params(axis='x', which='minor', bottom=False)

fig.tight_layout()
fig.savefig('community.pdf')


