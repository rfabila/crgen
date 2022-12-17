# crgen
An O(n^2log n) algorithm to compute the number crossings of a geometric graph

This program depends on PyDCG. 

A geometric graph is represented by graph G and an array  pts of its points.
G is a dictionary whose keys are the indices of pts. Each entry G[i] in the dictionary
contains a list with the indices of the neighbours of vertex i.

usage:

import cr_general       
In [8]: cr_general.quadcr(G,pts)                                                
Out[8]: 148864
