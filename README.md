# pysizeunfolder
A python library with various functions related to stereological methods. In the current version there are some functions for taking random sections of convex polygons (2D) and polyhedrons (3D). These random sections are known as Isotropic Uniformly Random (IUR) sections. This library will at a later stage be updated with functions for estimating 3D grain/ particle size distributions from 2D sections. 

## Table of contents
* [Installation and dependencies](#installation-and-dependencies)
* [Code examples: Random sections of polygons](#code-examples-random-sections-of-polygons)
* [Code examples: Random sections of polyhedrons](#code-examples-random-sections-of-polyhedrons)
* [Documentation](#documentation)

## Installation and dependencies
The library may be installed by running:

```
pip install git+https://github.com/thomasvdj/vorostereology
```

If you do not have git installed and you use the popular python Anaconda distribution, then you may install git with: 

```
conda install git
```

At a later stage I may consider uploading it to Pypi. This package depends on Numpy, Scipy, Cython and KDEpy, these are installed automatically.

## Code examples: Random sections of polygons
To highlight some of the functionalities of this library we present some code snippets. All of the examples below can be found in more detail in the examples folder. The following imports are used throughout the examples:
```
import pysizeunfolder as pu
import numpy as np
```
### Example 1
In the following example we define the centered unit square via its vertices. Then, we take 1 million random linear sections of the square and we obtain the lengths. Given this sample of lengths we approximate the associated probability density function of this distribution. The random generator rng is ussed to make the results reproducible.
```
rng = np.random.default_rng(0)
points = np.array([[-0.5, -0.5], [0.5, 0.5], [-0.5, 0.5], [0.5, -0.5]])
lengths = pu.iur_2d_hull(points, n=1000000, rng)
x, y = pu.approx_length_density(lengths)
```
Plotting lengths in a histogram and the x, y points of the approximation with matplotlib:

<img src="https://github.com/thomasvdj/pysizeunfolder/blob/main/examples/square_estimate.png" width=50% height=50%>

### Example 2
In the following example we generate 10 random points in the centered unit square, and we define a polygon as the convex hull of these points. For this given polygon we take 100 random sections, and we also retrieve the vertices of the corresponding lines.
```
points = rng.uniform(low=-0.5, high=0.5, size=(10, 2))
lengths, vertices = pu.iur_2d_hull(points, n=100, return_vertices=True)
```
Visualizing the polygon and the linear sections with matplotlib:

<img src="https://github.com/thomasvdj/pysizeunfolder/blob/main/examples/random_polygon.png" width=50% height=50%>

## Code examples: Random sections of polyhedrons

### Example 1
In the following example we define the centered unit cube via its vertices. Then, we take 1 million random planar sections of the cube and we obtain the areas. Given this sample of areas we approximate the associated probability density function of this distribution. 
```
points = np.array([[0.5, 0.5, 0.5], [-0.5, 0.5, 0.5], [0.5, 0.5, -0.5], [0.5, -0.5, 0.5],
                   [-0.5, -0.5, 0.5], [0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [-0.5, -0.5, -0.5]])
areas = pu.iur_3d_hull(points, n=1000000, rng=rng)
x, y = pu.approx_area_density(areas)
```
Plotting areas in a histogram and the x, y points of the approximation with matplotlib:

<img src="https://github.com/thomasvdj/pysizeunfolder/blob/main/examples/cube_estimate.png" width=50% height=50%>

### Example 2
In the following example we generate 15 random points in the centered unit cube, and we define a polyhedron as the convex hull of these points. For this given polyhedron we take a single random plane section, and we also retrieve the vertices of the resulting polygon.
```
points = rng.uniform(low=-0.5, high=0.5, size=(15, 3))
area, section = pu.iur_3d_hull(points, 1, return_vertices=True, rng=rng)
```
Visualizing the polyhedron and the planar section with pyvista:

<img src="https://github.com/thomasvdj/pysizeunfolder/blob/main/examples/random_polyhedron.png" width=50% height=50%>

Note: for the 3D visualization pyvista was used, which is not installed by default. It may be installed via: 

```
pip install vtk pyvista
```

## Documentation
For now the documentation consists of the code examples as above (and in the examples folder). Each function is documented using docstrings, which can be found in the source code.
