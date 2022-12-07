from . import interface
import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection
from math import sin, cos, pi
from KDEpy import FFTKDE


# Author: Thomas van der Jagt


def iur_2d_hull(points, n, return_vertices=False, rng=None, normalize_length=False):
    """
    A Function for computing random line sections of a 2D convex polygon. The convex polygon should be
    represented by a set of points such that the polygon is defined as the convex hull of these points. The random
    sections are known as so-called Isotropic Uniformly Random (IUR) sections. By default only the lengths of the random
    sections are returned. Setting return_vertices to True will also return the endpoints of the lines resulting from
    the linear sections in 2d space.

    Note: convex hull computation is performed by Scipy.spatial.ConvexHull.

    Note: for computational efficiency of the sampling algorithm it is best that points are presented such that the
    center of mass of the resulting polygon is close to the origin.

    :param points: An array containing points in 2d space, a numpy.ndarray of shape (K, 2) where K is the number of
            points.
    :param n: An integer, the number of random (IUR) sections to take of the presented convex polygon.
    :param return_vertices: A boolean, indicating whether the vertices of the resulting sections should be returned.
    :param rng: A numpy.random.Generator, may be used to make the random results reproducible. If not given it defaults
            to numpy.random.default_rng().
    :param normalize_length: A boolean, indicating whether the section lengths should be scaled such that they correspond
            to sections of the same polygon scaled to have area 1.
    :return: A numpy.ndarray "lengths" of shape (n,) containing the lengths of the random sections. If return_verticees
            is set to True, then also a list "vertices" of length n is returned. At index i of "vertices" we find a
            numpy.ndarray of shape (m, 2) containing the vertices of the line resulting from a random section, the
            length of this polygon may be found at index i in "lengths". If return_vertices is set to True the tuple
            "lengths", "vertices" is returned.
    """
    if rng is None:
        rng = np.random.default_rng()
    hull = ConvexHull(points)
    h, idx = np.unique(hull.equations, axis=0, return_inverse=True)
    poly_vertices = points[hull.vertices]

    # The sphere with radius max_distance encloses the polygon
    max_distance = np.max(np.linalg.norm(poly_vertices, axis=1))
    section_vertices = []

    halfspaces = [set() for _ in range(h.shape[0])]
    for ix in range(len(idx)):
        halfspaces[idx[ix]].update(set(hull.simplices[ix]))
    halfspaces = [frozenset(half) for half in halfspaces]

    indices = np.arange(len(points))
    counter = 0
    iteration = 0
    lengths = np.zeros(n)
    uniforms = rng.uniform(size=(n, 3))
    while counter < n:
        # Generate a random line
        phi = uniforms[iteration, 0] * 2 * pi
        distance = uniforms[iteration, 1] * max_distance
        normal = np.array([sin(phi), cos(phi)])

        # Determine which vertices of the polyhedron are above and below the section line
        mask = np.dot(points, normal) - distance > 0
        indices_below = set(indices[np.logical_not(mask)])
        indices_above = set(indices[mask])

        # Check whether the plane intersects with the polygon
        if len(indices_above) > 0 and len(indices_below) > 0:
            # Determine which edges are intersected by the line
            intersected_edges = {e for e in halfspaces if not (e <= indices_below or e <= indices_above)}

            vertices = np.zeros((len(intersected_edges), 2))
            ix = 0
            # Iterate over the intersected edges
            for edge in intersected_edges:
                # Compute the intersection point, it is a convex combination of the vertices of the intersected edge
                section_indices = tuple(edge)
                p1 = points[section_indices[0]]
                p2 = points[section_indices[1]]
                scalar = (distance - np.dot(normal, p2)) / np.dot(normal, p1 - p2)
                vertices[ix] = scalar * p1 + (1 - scalar) * p2
                ix += 1

            if return_vertices:
                section_vertices.append(vertices)
            lengths[counter] = np.linalg.norm(vertices[0]-vertices[1])
            counter += 1

        iteration += 1
        if iteration == n:
            iteration = 0
            uniforms = rng.uniform(size=(n, 2))

    if normalize_length:
        lengths = lengths / (hull.volume ** (1. / 2))
    if return_vertices:
        return lengths, section_vertices
    return lengths


def iur_2d_half(halfspaces, origin, n, return_vertices=False, rng=None, normalize_length=False):
    """
    A Function for computing random line sections of a 2D convex polygon. The convex polygon should be
    represented by a halfspace representation. For more details see Scipy.Spatial.HalfspaceIntersection, which is used
    to compute the intersections of the halfspaces. The random sections are known as so-called Isotropic Uniformly
    Random (IUR) sections. By default only the lengths of the random sections are returned. Setting return_vertices
    to True will also return the endpoints of the lines resulting from the linear sections in 2d space.

    Note: convex hull computation is performed by Scipy.spatial.ConvexHull.

    Note: for computational efficiency of the sampling algorithm it is best that points are presented such that the
    center of mass of the resulting polygon is close to the origin.

    :param halfspaces: An array describing the halfspaces, a numpy.ndarray of shape (K, 3) where K is the number of
            halfspaces. See the documentation of Scipy.spatial.ConvexHull, for more details of the format.
    :param origin: A point in 2d space which has to be contained in the polygon defined by the given halfspaces.
            It is a numpy.ndarray of shape (2,).
    :param n: An integer, the number of random (IUR) sections to take of the presented convex polygon.
    :param return_vertices: A boolean, indicating whether the vertices of the resulting sections should be returned.
    :param rng: A numpy.random.Generator, may be used to make the random results reproducible. If not given it defaults
            to numpy.random.default_rng().
    :param normalize_length: A boolean, indicating whether the section lengths should be scaled such that they
            correspond to sections of the same polygon scaled to have area 1.
    :return: A numpy.ndarray "lengths" of shape (n,) containing the lengths of the random sections. If return_verticees
            is set to True, then also a list "vertices" of length n is returned. At index i of "vertices" we find a
            numpy.ndarray of shape (2, 2) containing the vertices of the line resulting from a random section, the
            length of this line may be found at index i in "lengths". If return_vertices is set to True the tuple
            "lengths", "vertices" is returned.
    """
    if rng is None:
        rng = np.random.default_rng()
    poly_vertices = HalfspaceIntersection(halfspaces, origin).intersections
    return iur_2d_hull(poly_vertices, n, return_vertices, rng, normalize_length)


def approx_length_density(data, num_points=32768, bw=None, nb=1000):
    """
    This function approximates the probabillity density function associated with the distribution of lengths of random
    sections of a polygon. This is done via kernel density estimation with a boundary correction method known as the
    reflection method. For details of the boundary correction method see:

     Eugene F. Schuster (1985) Incorporating support constraints into nonparametric estimators of densities,
     Communications in Statistics - Theory and Methods, 14:5, 1123-1136, DOI: 10.1080/03610928508828965

    As an input a (large) sample of section lengths should be provided.

    :param data: A sample of section lengths, a numpy.ndarray of shape (n,) where n is the sample size.
    :param num_points: An integer indicating how many points should be used for evaluating the kernel density estimator.
            Because the kernel density estimator is computed via a FFT algorithm this should be a power of 2.
    :param bw: A double, the bandwidth used for the kernel density estimator. If not given, the Sheather-Jones (1991)
            method is used.
    :param nb: an integer, for computing the bandwidth with the Sheather-Jones method the data is first binned. nb
            represents the number of bins.
    :return: An approximation of a probability density function evaluated on a grid of points. A tuple x_pts, y_pts is
            returned, both x_pts and y_pts are of type numpy.ndarray and of shape (num_points/2,).
    """
    new_data = np.concatenate([data, -data])
    if bw is None:
        bw = interface.bw_sj(new_data, nb)
    x_pts, y_pts = FFTKDE(bw=bw).fit(new_data).evaluate(num_points)
    y_pts = 2 * y_pts[x_pts > 0]
    x_pts = x_pts[x_pts > 0]
    return x_pts, y_pts


def vertices_2d_hull(points):
    """
    A function which arranges the vertices of a given polygon in clockwise order. The convex polygon should be
    represented by a set of points such that the polygon is defined as the convex hull of these points. This function
    may be of use for 2d plotting.

    :param points: An array containing points in 2d space, a numpy.ndarray of shape (K, 2) where K is the number of
            points.
    :return: A numpy.ndarray of shape (K, 2), containing the same points as in "points" but now in clockwise order.
    """
    return points[ConvexHull(points).vertices]


def vertices_2d_half(halfspaces, origin):
    """
    A function which arranges the vertices of a given polygon in clockwise order. The convex polygon should be
    represented by halfspace representation. For more details see Scipy.Spatial.HalfspaceIntersection, which is used
    to compute the intersections of the halfspaces. This function may be of use for 2d plotting.

    :param halfspaces: An array describing the halfspaces, a numpy.ndarray of shape (K, 3) where K is the number of
            halfspaces. See the documentation of Scipy.spatial.ConvexHull, for more details of the format.
    :param origin: A point in 2d space which has to be contained in the polygon defined by the given halfspaces.
            It is a numpy.ndarray of shape (2,).
    :return: A numpy.ndarray of shape (K, 2), containing the vertices of the given polygon in clockwise order.
    """
    return vertices_2d_hull(HalfspaceIntersection(halfspaces, origin).intersections)
