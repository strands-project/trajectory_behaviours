#!/usr/bin/env python


import cPickle as pickle
import sys
from cyclic_processes import *

vec = [1416842797, 1416832334, 1416850621, 1416850902, 1416838009, 1416843788, 1416847972,
       1416850642, 1416844969, 1416854637, 1416842869, 1416829785, 1416838225, 1416847806,
       1416850880, 1416847958, 1416852401, 1416838260, 1416838916, 1416838358, 1416838276,
       1416857778, 1416842883, 1416838285, 1416837833, 1416843769, 1416850803, 1416850851,
       1416850769, 1416829620, 1416844985, 1416842935, 1416831389, 1416844949, 1416841563,
       1416850819, 1416838243, 1416838150, 1416838510, 1416847889, 1416838010, 1416850881,
       1416842872, 1416838233, 1416850751, 1416838359, 1416834129, 1416850903, 1416847825,
       1416842007, 1416847712, 1416838272, 1416837936, 1416847886, 1416847968, 1416842866,
       1416838247, 1416838495, 1416844934, 1416847716, 1416844998, 1416842945, 1416850818,
       1416838359, 1416845006, 1416850779, 1416850937, 1416850869, 1416847879, 1416838275,
       1416847727]
       
timestamps_vec, ind = time_wrap(vec)

#timestamps_vec = [v % 86400 for v in vec]

dyn_cl = dynamic_clusters()
for t in range(len(timestamps_vec)):
    dyn_cl.add_element(t+1,timestamps_vec[t])

## NOTE: interval (bin size) is 1800 (1/2 hour) by default, but if the data are very sparse you can increase it
fitting = activity_time(timestamps_vec,interval=3600.0)

# close this with a keypress
#stop = fitting.display_indexes(['trajectories','g','b'],dyn_cl,[]) # plot_options: title, hist_colour, curve_colour

# querying the fitting of new data x:
# p = fitting.query_model(x%86400)
# example below
pc = []
pf = []
for v in timestamps_vec:
    pc.append(dyn_cl.query_clusters(v%86400))
    pf.append(fitting.query_model(v%86400))

print pf
#plot(pf)

