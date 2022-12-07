from . import interface
import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection
from math import sin, cos, acos, pi, sqrt
from KDEpy import FFTKDE
import itertools


# Author: Thomas van der Jagt


def iur_3d_hull(points, n, return_vertices=False, rng=None, normalize_area=False):
    """
    A Function for computing random plane sections of a 3D convex polyhedron. The convex polyhedron should be
    represented by a set of points such that the polyhedron is defined as the convex hull of these points. The random
    sections are known as so-called Isotropic Uniformly Random (IUR) sections. By default only the areas of the random
    sections are returned. Setting return_vertices to True will also return the vertices of the polygons resulting from
    the plane sections in 3d space. These vertices are then guaranteed to be ordered in clockwise or counter-clockwise
    orientation.

    Note: convex hull computation is performed by Scipy.spatial.ConvexHull.

    Note: for computational efficiency of the sampling algorithm it is best that points are presented such that the
    center of mass of the resulting polyhedron is close to the origin.

    :param points: An array containing points in 3d space, a numpy.ndarray of shape (K, 3) where K is the number of
            points.
    :param n: An integer, the number of random (IUR) sections to take of the presented convex polyhedron.
    :param return_vertices: A boolean, indicating whether the vertices of the resulting sections should be returned.
    :param rng: A numpy.random.Generator, may be used to make the random results reproducible. If not given it defaults
            to numpy.random.default_rng().
    :param normalize_area: A boolean, indicating whether the section areas should be scaled such that they correspond
            to section areas of the same polyhedron scaled to have volume 1.
    :return: A numpy.ndarray "areas" of shape (n,) containing the areas of the random sections. If return_verticees is
            set to True, then also a list "vertices" of length n is returned. At index i of "vertices" we find a
            numpy.ndarray of shape (m, 3) containing the vertices of the polygon resulting from a random section, the
            area of this polygon may be found at index i in "areas". If return_vertices is set to True the tuple
            "areas", "vertices" is returned.
    """
    if rng is None:
        rng = np.random.default_rng()
    hull = ConvexHull(points)
    h, idx = np.unique(hull.equations, axis=0, return_inverse=True)
    poly_vertices = points[hull.vertices]

    # The sphere with radius max_distance encloses the polyhedron
    max_distance = np.max(np.linalg.norm(poly_vertices, axis=1))
    edges = []
    section_vertices = []

    # halfspaces: at index i we find the indices of the vertices in halfspace/face i of the polyhedron
    halfspaces = [set() for _ in range(h.shape[0])]
    for ix in range(len(idx)):
        halfspaces[idx[ix]].update(set(hull.simplices[ix]))
    halfspaces = [frozenset(half) for half in halfspaces]

    # edges: at index i we find the edges in halfspace/face i of the polyhedron.
    # each edge is a set of length 2, containing the indices of the corresponding vertices.
    for ix in range(h.shape[0]):
        intersections = [halfspaces[ix] & halfspaces[j] for j in range(h.shape[0])]
        edges.append({edge for edge in intersections if len(edge) == 2})

    # faces per edge: a dictionary, at index 'edge' we find a set containing the faces that 'edge' is part of
    all_edges = set(itertools.chain.from_iterable(edges))
    faces_per_edge = {edge: set() for edge in all_edges}
    for i in range(len(edges)):
        for edge in edges[i]:
            faces_per_edge[edge].add(i)

    indices = np.arange(len(points))
    counter = 0
    iteration = 0
    areas = np.zeros(n)
    uniforms = rng.uniform(size=(n, 3))

    while counter < n:
        # Generate a random plane
        phi = uniforms[iteration, 0] * 2 * pi
        cos_theta = 2 * (uniforms[iteration, 1] - 0.5)
        theta = acos(cos_theta)
        distance = uniforms[iteration, 2] * max_distance
        normal = np.array([sin(theta) * cos(phi), sin(theta) * sin(phi), cos_theta])

        # Determine which vertices of the polyhedron are above and below the section plane
        mask = np.dot(points, normal) - distance > 0
        indices_below = set(indices[np.logical_not(mask)])
        indices_above = set(indices[mask])

        # Check whether the plane intersects with the polyhedron
        if len(indices_above) > 0 and len(indices_below) > 0:
            # Determine which edges are intersected by the plane
            edges_below = {e for e in all_edges if e <= indices_below}
            intersected_edges = {e for e in all_edges - edges_below if not e <= indices_above}
            edges_to_pass = {edge: faces_per_edge[edge].copy() for edge in intersected_edges}

            vertices = np.zeros((len(intersected_edges), 3))
            ix = 0
            first = True
            # Iterate over the intersected edges
            while len(intersected_edges) > 0:
                if first:
                    edge = intersected_edges.pop()
                    face = edges_to_pass[edge].pop()
                    first = False
                else:
                    edge = (edges[face] & intersected_edges).pop()
                    intersected_edges.remove(edge)
                    edges_to_pass[edge].remove(face)
                    face = edges_to_pass[edge].pop()

                # Compute the intersection point, it is a convex combination of the vertices of the intersected edge
                edge_indices = tuple(edge)
                p1 = points[edge_indices[0]]
                p2 = points[edge_indices[1]]
                vertices[ix] = interface.intersection_point(p1, p2, normal, distance)
                ix += 1

            # Compute the area of the polygon formed by the computed intersection points
            if return_vertices:
                section_vertices.append(vertices)
            areas[counter] = interface.polygon_area(vertices)
            counter += 1
        iteration += 1
        if iteration == n:
            iteration = 0
            uniforms = rng.uniform(size=(n, 3))

    if normalize_area:
        areas = areas / (hull.volume ** (2. / 3))
    if return_vertices:
        return areas, section_vertices
    return areas


def iur_3d_half(halfspaces, origin, n, return_vertices=False, rng=None, normalize_area=False):
    """
    A Function for computing random plane sections of a 3D convex polyhedron. The convex polyhedron should be
    represented by halfspace representation. For more details see Scipy.Spatial.HalfspaceIntersection, which is used
    to compute the intersections of the halfspaces. The random sections are known as so-called Isotropic Uniformly
    Random (IUR) sections. By default only the areas of the random sections are returned. Setting return_vertices to
    True will also return the vertices of the polygons resulting from the plane sections in 3d space. These vertices
    are then guaranteed to be ordered in clockwise or counter-clockwise orientation.

    Note: a convex hull computation is performed by Scipy.spatial.ConvexHull.

    Note: for computational efficiency of the sampling algorithm it is best that points are presented such that the
    center of mass of the resulting polyhedron is close to the origin.

    :param halfspaces: An array describing the halfspaces, a numpy.ndarray of shape (K, 4) where K is the number of
            halfspaces. See the documentation of Scipy.spatial.ConvexHull, for more details of the format.
    :param origin: A point in 3d space which has to be contained in the polyhedron defined by the given halfspaces.
            It is a numpy.ndarray of shape (3,).
    :param n: An integer, the number of random (IUR) sections to take of the presented convex polyhedron.
    :param return_vertices: A boolean, indicating whether the vertices of the resulting sections should be returned.
    :param rng: A numpy.random.Generator, may be used to make the random results reproducible. If not given it defaults
            to numpy.random.default_rng().
    :param normalize_area: A boolean, indicating whether the section areas should be scaled such that they correspond
            to section areas of the same polyhedron scaled to have volume 1.
    :return: A numpy.ndarray "areas" of shape (n,) containing the areas of the random sections. If return_verticees is
            set to True, then also a list "vertices" of length n is returned. At index i of "vertices" we find a
            numpy.ndarray of shape (m, 3) containing the vertices of the polygon resulting from a random section, the
            area of this polygon may be found at index i in "areas". If return_vertices is set to True the tuple
            "areas", "vertices" is returned.
    """
    if rng is None:
        rng = np.random.default_rng()
    poly_vertices = HalfspaceIntersection(halfspaces, origin).intersections
    return iur_3d_hull(poly_vertices, n, return_vertices, rng, normalize_area)


def iur_3d_shape(shape, n, return_vertices=False, rng=None):
    """
    A Function for computing random plane sections of a 3D convex polyhedron. Various shapes are implemented in this
    function, the name of the polyhedron should be given as an input. Supported shapes: cube, dodecahedron, Kelvin cell
    (AKA tetrakaidecahedron) and tetrahedron. The random sections are known as so-called Isotropic Uniformly Random
    (IUR) sections. By default only the areas of the random sections are returned. Setting return_vertices to True will
    also return the vertices of the polygons resulting from the plane sections in 3d space. These vertices are then
    guaranteed to be ordered in clockwise or counter-clockwise orientation.

    Note: a convex hull computation is performed by Scipy.spatial.ConvexHull.

    Note: for computational efficiency of the sampling algorithm it is best that points are presented such that the
    center of mass of the resulting polyhedron is close to the origin.

    :param shape: A string which can be any of: "cube", "dodecahedron", "kelvin", "tetrahedron".
    :param n: An integer, the number of random (IUR) sections to take of the presented convex polyhedron.
    :param return_vertices: A boolean, indicating whether the vertices of the resulting sections should be returned.
    :param rng: A numpy.random.Generator, may be used to make the random results reproducible. If not given it defaults
            to numpy.random.default_rng().
    :return: A numpy.ndarray "areas" of shape (n,) containing the areas of the random sections. If return_verticees is
            set to True, then also a list "vertices" of length n is returned. At index i of "vertices" we find a
            numpy.ndarray of shape (m, 3) containing the vertices of the polygon resulting from a random section, the
            area of this polygon may be found at index i in "areas". If return_vertices is set to True the tuple
            "areas", "vertices" is returned.
    """
    if rng is None:
        rng = np.random.default_rng()
    if shape == "cube":
        points = np.array([[0.5, 0.5, 0.5], [-0.5, 0.5, 0.5], [0.5, 0.5, -0.5], [0.5, -0.5, 0.5],
                           [-0.5, -0.5, 0.5], [0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [-0.5, -0.5, -0.5]])
        return iur_3d_hull(points, n, return_vertices, rng)
    elif shape == "dodecahedron":
        phi = (1 + sqrt(5)) * 0.5
        points = np.array(
            [[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, 1, -1], [1, -1, -1], [-1, -1, -1], [1, -1, 1],
             [0, phi, 1. / phi], [0, -phi, 1. / phi], [0, phi, -1. / phi], [0, -phi, -1. / phi],
             [1. / phi, 0, phi], [-1. / phi, 0, phi], [1. / phi, 0, -phi], [-1. / phi, 0, -phi],
             [phi, 1. / phi, 0], [-phi, 1. / phi, 0], [phi, -1. / phi, 0], [-phi, -1. / phi, 0]])
        return iur_3d_hull(points, n, return_vertices, rng, normalize_area=True)
    elif shape == "kelvin":
        points = np.array([[0, 1, 2], [0, -1, 2], [0, 1, -2], [0, -1, -2], [1, 0, 2], [-1, 0, 2], [1, 0, -2],
                           [-1, 0, -2], [1, 2, 0], [-1, 2, 0], [1, -2, 0], [-1, -2, 0], [0, 2, 1], [0, -2, 1],
                           [0, 2, -1], [0, -2, -1], [2, 0, 1], [-2, 0, 1], [2, 0, -1], [-2, 0, -1], [2, 1, 0],
                           [-2, 1, 0], [2, -1, 0], [-2, -1, 0]], dtype=np.double)
        return iur_3d_hull(points, n, return_vertices, rng, normalize_area=True)
    elif shape == "tetrahedron":
        points = np.array([[1, 0, -1./sqrt(2)], [-1, 0, -1./sqrt(2)], [0, 1, 1./sqrt(2)], [0, -1, 1./sqrt(2)]],
                          dtype=np.double)
        return iur_3d_hull(points, n, return_vertices, rng, normalize_area=True)
    else:
        raise ValueError('The provided shape is not supported')


def approx_area_density(data, sqrt_data=False, num_points=32768, bw=None, nb=1000):
    """
    This function approximates the probabillity density function associated with the distribution of areas of random
    sections of a polyhedron. This is done via kernel density estimation with a boundary correction method known as the
    reflection method. For details of the boundary correction method see:

     Eugene F. Schuster (1985) Incorporating support constraints into nonparametric estimators of densities,
     Communications in Statistics - Theory and Methods, 14:5, 1123-1136, DOI: 10.1080/03610928508828965

    As an input a (large) sample of section areas should be provided. One may also provide
    a sample of areas which was transformed with a square-root transformation. See the paper "On the distribution
    function of volumes of random hyperplane sections of convex bodies" for an explanation as to why to do this.

    :param data: A sample of (square-root) section areas, a numpy.ndarray of shape (n,) where n is the sample size.
    :param sqrt_data: A boolean indicating whether the provided sample contains square-root section areas.
    :param num_points: An integer indicating how many points should be used for evaluating the kernel density estimator.
            Because the kernel density estimator is computed via a FFT algorithm this should be a power of 2.
    :param bw: A double, the bandwidth used for the kernel density estimator. If not given, the Sheather-Jones (1991)
            method is used.
    :param nb: an integer, for computing the bandwidth with the Sheather-Jones method the data is first binned. nb
            represents the number of bins.
    :return: An approximation of a probability density function evaluated on a grid of points. A tuple x_pts, y_pts is
            returned, both x_pts and y_pts are of type numpy.ndarray and of shape (num_points/2,).
    """
    if sqrt_data:
        new_data = np.concatenate([data, -data])
    else:
        new_data = np.concatenate([np.sqrt(data), -np.sqrt(data)])
    if bw is None:
        bw = interface.bw_sj(new_data, nb)
    x_pts, y_pts = FFTKDE(bw=bw).fit(new_data).evaluate(num_points)
    y_pts = 2 * y_pts[x_pts > 0]
    x_pts = x_pts[x_pts > 0]
    if sqrt_data:
        return x_pts, y_pts
    else:
        return np.square(x_pts), 0.5 * y_pts / x_pts


def faces_3d_hull(points):
    """
    A function which computes the vertices and faces of a given polyhedron. The convex polyhedron should be represented
    by a set of points such that the polyhedron is defined as the convex hull of these points. This function may be of
    use for 3d plotting since the vertices within each face are ordered in a clockwise or counterclockwise fashion as
    typically required by 3d plotting functions.

    :param points: An array containing points in 3d space, a numpy.ndarray of shape (K, 3) where K is the number of
            points.
    :return: A tuple "vertices", "faces". "vertices" is an array of points in 3d space, it is a numpy.ndarray of
            shape (m, 3) where m is the number of vertices. "faces" is a list of length n, where n is the number of
            faces of the polyhedron. At index i of "faces" we find a list of integers, each integer is an index
            corresponding to a point in "vertices". All these points together form the vertices of face i.
    """
    hull = ConvexHull(points)
    h, idx = np.unique(hull.equations, axis=0, return_inverse=True)

    # halfspaces: at index i we find the indices of the vertices in halfspace/face i of the polyhedron
    halfspaces = [set() for _ in range(h.shape[0])]
    for ix in range(len(idx)):
        halfspaces[idx[ix]].update(set(hull.simplices[ix]))
    halfspaces = [frozenset(half) for half in halfspaces]

    edges = []
    # edges: at index i we find the edges in halfspace/face i of the polyhedron.
    # each edge is a set of length 2, containing the indices of the corresponding vertices.
    for ix in range(h.shape[0]):
        intersections = [halfspaces[ix] & halfspaces[j] for j in range(h.shape[0])]
        edges.append({edge for edge in intersections if len(edge) == 2})
    halfspaces = [list(half) for half in halfspaces]

    for ix in range(h.shape[0]):
        edges_in_face = edges[ix].copy()
        if len(edges_in_face) > 3:
            points_in_face = halfspaces[ix].copy()
            current = points_in_face[0]
            temp = [points_in_face[0]]
            points_in_face.pop(0)
            while len(temp) <= len(points_in_face):
                for point in points_in_face:
                    possible_edge = {current, point}
                    if possible_edge in edges_in_face:
                        current = point
                        edges_in_face.remove(possible_edge)
                        temp.append(point)
            halfspaces[ix] = temp
    return halfspaces


def vertices_faces_3d_half(halfspaces, origin):
    """
    A function which computes the vertices and faces of a given polyhedron. The convex polyhedron should be
    represented by halfspace representation. For more details see Scipy.Spatial.HalfspaceIntersection, which is used
    to compute the intersections of the halfspaces. This function may be of use for 3d plotting since the vertices
    within each face are ordered in a clockwise or counterclockwise fashion as typically required by 3d plotting
    functions.

    :param halfspaces: An array describing the halfspaces, a numpy.ndarray of shape (K, 4) where K is the number of
            halfspaces. See the documentation of Scipy.spatial.ConvexHull, for more details of the format.
    :param origin: A point in 3d space which has to be contained in the polyhedron defined by the given halfspaces.
            It is a numpy.ndarray of shape (3,).
    :return: A tuple "vertices", "faces". "vertices" is an array of points in 3d space, it is a numpy.ndarray of
            shape (m, 3) where m is the number of vertices. "faces" is a list of length n, where n is the number of
            faces of the polyhedron. At index i of "faces" we find a list of integers, each integer is an index
            corresponding to a point in "vertices". These points together form the vertices of face i.
    """
    poly_vertices = HalfspaceIntersection(halfspaces, origin).intersections
    return poly_vertices, faces_3d_hull(poly_vertices)
