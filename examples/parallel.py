import os
import numpy as np
import pysizeunfolder as pu
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import pickle


# Author: Thomas van der Jagt


n = 10000000
points = np.array([[0.5, 0.5, 0.5], [-0.5, 0.5, 0.5], [0.5, 0.5, -0.5], [0.5, -0.5, 0.5],
                   [-0.5, -0.5, 0.5], [0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [-0.5, -0.5, -0.5]])

ss = np.random.SeedSequence(0)
num_cpus = os.cpu_count()
child_seeds = ss.spawn(num_cpus)
streams = [np.random.default_rng(s) for s in child_seeds]

block_size = n // num_cpus
remainder = n - num_cpus*block_size
sizes = [block_size]*num_cpus
sizes[-1] += remainder

res = Parallel(n_jobs=num_cpus)(delayed(pu.iur_3d_hull)(points, sizes[i], False, streams[i]) for i in range(num_cpus))
areas = np.concatenate(res)

f = open("cube_sample.pkl", "wb")
pickle.dump(areas, f)
f.close()

x, y = pu.approx_area_density(areas)

plt.figure()
plt.hist(areas, bins=80, ec='black', linewidth=0.2, density=True)
plt.plot(x, y)
plt.xlim(0, 1.5)
plt.ylim(0, 3)
plt.show()
