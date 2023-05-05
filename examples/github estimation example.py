import pysizeunfolder as pu
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.stats import expon, gamma


rng = np.random.default_rng(0)
reference_sample = pickle.load(open("cube_sample.pkl", "rb"))
n = 1000

# Generate a sample of observed section areas (Lemma 2), given that particles are cubes and the
# underlying size distribution is a standard exponential distribution.
sizes = rng.gamma(shape=2, scale=1, size=n)
areas = pu.iur_3d_shape("cube", n, rng=rng)
sample = np.square(sizes)*areas

# Estimate the underlying size distribution CDF using the sample of observed section areas
x_pts, y_pts = pu.estimate_size(sample, reference_sample)

# For plotting with the matplotlib step function, we need to add some additional points
x_pts = np.append(np.append(0, x_pts), 1.05*x_pts[-1])
y_pts = np.append(np.append(0, y_pts), 1)

# The true size distribution CDF
x = np.linspace(0, np.max(x_pts), 2000)
y = expon.cdf(x)

plt.figure(figsize=(4, 3))
plt.step(x_pts, y_pts, where="post", c="tab:blue", label="estimate")
plt.plot(x, y, c="red", linestyle="dashed", label="truth")
plt.xlim([0, 7])
plt.title(r"Estimate vs truth $(H(\lambda))$")
plt.xlabel(r"$\lambda$")
plt.ylabel("CDF")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("size_estimate.png", dpi=600)
#plt.show()

# Alternatively: skip the de-biasing step
# Estimate the underlying size distribution using the sample of observed section areas
x_pts, y_pts = pu.estimate_size(sample, reference_sample, debias=False)

# For plotting with the matplotlib step function, we need to add some additional points
x_pts = np.append(np.append(0, x_pts), 1.05*x_pts[-1])
y_pts = np.append(np.append(0, y_pts), 1)

# The true biased size distribution CDF
x = np.linspace(0, np.max(x_pts), 2000)
y = gamma.cdf(x, a=2, scale=1)

plt.figure(figsize=(4, 3))
plt.step(x_pts, y_pts, where="post", c="tab:blue", label="estimate")
plt.plot(x, y, c="red", linestyle="dashed", label="truth")
plt.xlim([0, 9])
plt.title(r"Estimate vs truth ($H^b(\lambda)$)")
plt.xlabel(r"$\lambda$")
plt.ylabel("CDF")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("biased_size_estimate.png", dpi=600)
#plt.show()

# Various code examples

# The estimated (biased) volume distribution function may be plotted via
x_pts = np.append(np.append(0, x_pts**3), 1.05*x_pts[-1]**3)
y_pts = np.append(np.append(0, y_pts), 1)
# Estimate both biased and debiased in a more efficient way:
x_pts_biased, y_pts_biased = pu.estimate_size(sample, reference_sample, debias=False)
y_pts = pu.de_bias(x_pts_biased, y_pts_biased, reference_sample)  # x_pts=x_pts_biased
