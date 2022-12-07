# cython: language_level=3
#

# Author: Thomas van der Jagt


import numpy as np
from scipy.stats import iqr
from scipy.optimize import brentq
from libc.math cimport sqrt, pi, exp
cimport cython


import sys
import math


@cython.boundscheck(False)
@cython.wraparound(False)
def polygon_area(double[:,::1] points):
    # Compute polygon area given points in 3D space with shoelace formula.
    cdef Py_ssize_t n = len(points)
    cdef Py_ssize_t i
    cdef double edge1_x, edge1_y, edge1_z, edge2_x, edge2_y, edge2_z
    cdef double res_x, res_y, res_z
    res_x = 0.0
    res_y = 0.0
    res_z = 0.0

    for i in range(1, n-1):
        edge1_x = points[i][0] - points[0][0]
        edge1_y = points[i][1] - points[0][1]
        edge1_z = points[i][2] - points[0][2]
        edge2_x = points[i+1][0] - points[0][0]
        edge2_y = points[i+1][1] - points[0][1]
        edge2_z = points[i+1][2] - points[0][2]
        res_x += edge1_y*edge2_z - edge1_z*edge2_y
        res_y += edge1_z*edge2_x - edge1_x*edge2_z
        res_z += edge1_x*edge2_y - edge1_y*edge2_x
    
    return 0.5*sqrt(res_x*res_x + res_y*res_y + res_z*res_z)


@cython.boundscheck(False)
@cython.wraparound(False)
def intersection_point(double[::1] p1, double[::1] p2, double[::1] normal, double distance):
    res = np.zeros(3, dtype=np.double)
    cdef double[::1] res_view = res
    cdef double scalar

    scalar = (distance - (normal[0]*p2[0] + normal[1]*p2[1] + normal[2]*p2[2]))/(normal[0]*(p1[0]-p2[0]) + normal[1]*(p1[1]-p2[1]) + normal[2]*(p1[2]-p2[2]))
    res_view[0] = scalar * p1[0] + (1 - scalar) * p2[0]
    res_view[1] = scalar * p1[1] + (1 - scalar) * p2[1]
    res_view[2] = scalar * p1[2] + (1 - scalar) * p2[2]

    return res


def bw_sj(data, nb):
    # This function computes the bandwidth for a kernel density estimator using the Sheather-Jones (1991) method.
    # The implementation mostly a translation of the bw.SJ function in R.
    scale = min(np.std(data), iqr(data)/1.349)
    cdef Py_ssize_t n = len(data)
    cdef Py_ssize_t nb_ = nb
    cdef Py_ssize_t i, idx
    cdef double delta, term, sum_
    cdef double[::1] data_view = data

    xmin = np.min(data)
    xmax = np.max(data)
    rang = (xmax - xmin)
    cdef double dd = rang / nb
    bin_counts = np.zeros(nb, dtype=np.int64)
    cdef long long[::1] bin_view = bin_counts
    
    individual_counts, _ = np.histogram(data, bins=np.linspace(xmin, xmax, nb+1))
    cdef long long[::1] individual_counts_view = individual_counts

    # this piece of code produces bin_counts/binview which is equal to the 'cnt' array as produced by the 'bw_den' function in r
    for i in range(1, nb_):
        for idx in range(nb_-i):
            bin_view[i] += individual_counts_view[idx]*individual_counts_view[idx + i]
    bin_view[0] = int(np.sum(individual_counts*(individual_counts-1)/2))

    g1 = scale * (n**(-1./7)) * (32./(sqrt(2)*5))**(1./7)
    g2 = scale * (n**(-1./9)) * (64./(sqrt(2)*7))**(1./9)
    
    def phi_4(g):
        sum_ = 0.0
        for i in range(nb_):
            delta = i * dd / g
            delta *= delta
            if delta >= 1000:
                break
            term = exp(-delta / 2) * (delta * delta - 6 * delta + 3)
            sum_ += term * bin_view[i]
        
        sum_ = 2 * sum_ + n * 3
        return sum_/(n * (n-1)*sqrt(2*pi)*(g**(5)))
    
    def phi_6(g):
        sum_ = 0.0
        for i in range(nb_):
            delta = i * dd / g
            delta *= delta
            if delta >= 1000:
                break
            term = exp(-delta / 2) * (delta * delta * delta - 15 * delta * delta + 45 * delta - 15)
            sum_ += term * bin_view[i]
        
        sum_ = 2 * sum_ - 15 * n
        return sum_/(n * (n-1)*sqrt(2*pi)*(g**(7)))
    
    p4 = phi_4(g1)
    p6 = phi_6(g2)
    
    c1 = (1./(2*n*sqrt(pi)))**(1./5)
    c2 = (-6*sqrt(2)*p4/p6)**(1./7)
    
    def root_function(h):
        gamma_h = c2*(h**(5./7))
        pn = phi_4(gamma_h)
        return h - c1*(1./pn)**(1./5)
        
    it = 0
    hmax = 1.144 * scale * n**(-1./5)
    lower = 0.1*hmax 
    upper = hmax
    while (root_function(lower)*root_function(upper) > 0):
        if it > 99:
            break
        if it % 2:
            upper *= 1.2
        else:
            lower /= 1.2
        it += 1
    
    return brentq(root_function, lower, upper)

