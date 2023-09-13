# -*- coding: utf-8 -*-
"""
Created on Wed May 31 21:18:07 2023

@author: ksande
"""

# based on https://matplotlib.org/stable/gallery/lines_bars_and_markers/timeline.html

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

releases = pd.Series({
    '2023-07-03': 'v0.10.1',
    '2023-06-30': 'v0.10.0',
    '2023-03-18': 'v0.9.5',
    '2022-12-20': 'v0.9.4',
    '2022-09-15': 'v0.9.3',
    '2022-08-19': 'v0.9.2',
    '2022-03-29': 'v0.9.1',
    '2021-09-01': 'v0.9.0',
    '2021-01-04': 'v0.8.1',
    '2020-09-08': 'v0.8.0',
    '2020-04-22': 'v0.7.2',
    '2020-01-17': 'v0.7.1',
    '2019-12-18': 'v0.7.0',
    '2019-05-15': 'v0.6.2',
    '2019-01-31': 'v0.6.1',
    '2018-09-17': 'v0.6.0',
    '2018-05-13': 'v0.5.2',
    '2017-10-17': 'v0.5.1',
    '2017-08-11': 'v0.5.0',
    '2017-06-05': 'v0.4.5',
    '2017-02-18': 'v0.4.4',
    '2016-12-28': 'v0.4.3',
    '2016-12-07': 'v0.4.2',
    '2016-10-05': 'v0.4.1',
    '2016-07-28': 'v0.4.0',
    '2016-06-15': 'v0.3.3',
    '2016-05-03': 'v0.3.2',
    '2016-04-19': 'v0.3.1',
    '2016-03-21': 'v0.3.0',
    '2015-11-13': 'v0.2.2',
    '2015-07-16': 'v0.2.1',
    '2015-07-06': 'v0.2.0',
    '2015-04-20': 'v0.1.0',
})
releases.index = pd.to_datetime(releases.index)
releases = releases.sort_index()

tutorials = pd.Series({
    '2016-05-10': 'PVPMC\nWorkshop',
    '2017-05-10': 'PVPMC\nWorkshop',
    '2021-06-20': 'IEEE PVSC',
    '2021-10-29': 'PyData Global',
    '2022-08-24': 'PVPMC\nWorkshop',
    '2023-05-10': 'PVPMC\nWorkshop',
    '2023-06-11': 'IEEE PVSC',
})

tutorials.index = pd.to_datetime(tutorials.index)
tutorials = tutorials.sort_index()

other_events = pd.Series({
    #'2013-08-30': 'First GitHub commit',
    '2018-09-07': 'JOSS\npaper',
    '2018-12-17': 'Reached 100\nforum members',
    '2019-04-17': 'NumFocus\nAffiliation',
    '2020-03-15': 'Reached 50\ncode contributors',
    '2021-06-01': 'GSoC 2021',  # approximate
    '2021-10-13': 'Reached 500\nforum members',
    '2023-06-28': 'Reached 100\ncode contributors'
})
other_events.index = pd.to_datetime(other_events.index)
other_events = other_events.sort_index()


def make_timeline(dates, labels, levels, dotcolor, ax):
    
    ax.vlines(dates, 0, levels, color="k", ls='-', alpha=0.3)
    ax.plot(dates, np.zeros_like(dates), "o", lw=3, color='k', markerfacecolor=dotcolor)
    for date, label, level in zip(dates, labels, levels):
        ax.annotate(label, xy=(date, level),
                    xytext=(-3, np.sign(level)*3), textcoords="offset points",
                    horizontalalignment="center",
                    verticalalignment="bottom" if level > 0 else "top")
    
    ax.yaxis.set_visible(False)
    ax.spines[["left", "top", "right", "bottom"]].set_visible(False)
    ax.set_ylim(-7, 7)
    ax.margins(y=0.1)


timeline_events = {
    'Releases': releases.str[1:],
    'Tutorials &\nUser Meetings': tutorials,
    'Milestones': other_events,
}

fig, axes = plt.subplots(len(timeline_events), 1, sharex=True, figsize=(10, 4), layout="constrained")

big = np.tile([-4, 4, -1, 1], 100)
little = np.tile([-2, 2], 100)
dotcolors = ['tab:blue', 'tab:orange', 'tab:green']

for (title, events), ax, levels, dotcolor in zip(timeline_events.items(), axes, [big, little, little], dotcolors):
    make_timeline(events.index, events.values, levels[:len(events)], dotcolor, ax)
    
    ax.annotate(title, xy=(releases.index[0], 0),
                xytext=(-40, 3), textcoords="offset points",
                horizontalalignment="center",
                verticalalignment="center",
                rotation=90, fontweight='semibold')

# once all the timeline points are in, we know how long the arrows are:
lim = axes[0].get_xlim()
x1, x2 = lim
buffer = (x2 - x1) / 50
x1 += buffer
x2 -= buffer

for ax in axes:
    ax.xaxis.set_visible(False)
    ax.arrow(x=x1, y=0, dx=x2-x1, dy=0, color='k', width=0.05, head_width=1, head_length=25)

axes[0].set_xlim(*lim)

for ax in axes[:-1]:
    ax.xaxis.set_visible(False)


axes[-1].xaxis.set_visible(True)
axes[-1].spines[["bottom"]].set_visible(True)

plt.savefig('timeline2.pdf')


