import pysizeunfolder as pu
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.collections import PolyCollection


# Example 1

rng = np.random.default_rng(0)
points = np.array([[-0.5, -0.5], [0.5, 0.5], [-0.5, 0.5], [0.5, -0.5]])
lengths = pu.iur_2d_hull(points, n=1000000)
x, y = pu.approx_length_density(lengths)

plt.figure(figsize=(4, 3))
plt.hist(lengths, bins=80, ec='black', linewidth=0.2, density=True)
plt.plot(x, y)
plt.xlim(0, 1.5)
plt.ylim(0, 4)
plt.subplots_adjust(bottom=0.15)
plt.subplots_adjust(left=0.11)
plt.xlabel("Length")
plt.ylabel("Density")
plt.savefig("square estimate.png", dpi=600)
#plt.show()

# Example 2

points = rng.uniform(low=-0.5, high=0.5, size=(10, 2))
lengths, vertices = pu.iur_2d_hull(points, n=100, return_vertices=True)

poly_vertices = pu.vertices_2d_hull(points)
pc = PolyCollection([poly_vertices], facecolors='white', edgecolors='k')
lc = LineCollection(vertices, linewidths=1)

fig = plt.figure(figsize=(4, 4))
ax1 = fig.add_subplot(1, 1, 1)
ax1.add_collection(pc)
ax1.add_collection(lc)
ax1.set_axis_off()
ax1.set_xlim(-0.5, 0.5)
ax1.set_ylim(-0.5, 0.5)
plt.savefig("random polygon.png", dpi=600)
#plt.show()

# Various code examples

# This is a halfspace representation of the centered unit square
origin = np.array([0.0, 0.0])
halfspaces = np.array([[0, 1, -0.5], [0, -1, -0.5], [1, 0, -0.5], [-1, 0, -0.5], [1, 0, -0.5], [-1, 0, -0.5]],
                      dtype=np.double)
lengths = pu.iur_2d_half(halfspaces, origin, 100000)
