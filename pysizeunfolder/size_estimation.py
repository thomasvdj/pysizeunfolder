from .algorithms import hybrid_icm_em, icm, em
from .iur_3d import approx_area_density
import numpy as np
from scipy.interpolate import CubicSpline
from statsmodels.distributions import ECDF


# Author: Thomas van der Jagt


def de_bias(x_pts, estimate, reference_sample):
    """
    A separate function which performs the de-biasing step which may also be performed by the function
    pysizeunfolder.estimate_size after the optimization procedure.

    :param x_pts: A numpy.ndarray of shape (n,) representing the points at which the CDF is evaluated. Such as x_pts
        returned by the function pysizeunfolder.estimate_size.
    :param estimate: A numpy.ndarray of shape (n,) representing the values the CDF takes in x_pts. Such as y_pts
        returned by the function pysizeunfolder.estimate_size.
    :param reference_sample: A sample of section areas of the reference shape, a numpy.ndarray of shape (N,) where N is
        the sample size. Ideally N is very large.
    :return: A numpy arrays: y_pts. A numpy.ndarray of shape (n,). This arrays represent a piece-wise constant
    distribution functions. The CDF is constant on [x_pts[i],x_pts[i+1]). Here: [a,b) = {x: a <= x < b}.
        y_pts[i] is the CDF value in x_pts[i].
    """
    n = len(x_pts)
    observed_x = np.linspace(0, 1, n)
    distances = np.zeros(n)
    Hb_probabilities = np.append(estimate[0], np.diff(estimate))
    GS_cdf = ECDF(np.sqrt(reference_sample))

    FS_estimates = np.zeros((n, n))
    for i in range(n):
        kernel = GS_cdf(x_pts[i] / x_pts)
        divisor = np.cumsum(np.flip(Hb_probabilities))
        divisor[divisor == 0.0] = 1.0
        FS_estimates[:, i] = np.flip(np.cumsum(np.flip(kernel * Hb_probabilities)) / divisor)

    for i in range(n):
        term1 = np.abs(observed_x - FS_estimates[i, :])[:(n - 1)]
        term2 = np.abs(observed_x[1:] - FS_estimates[i, :(n - 1)])
        distances[i] = 0.5 * np.dot(term1 + term2, np.diff(x_pts))

    trunc_ix = np.argmin(distances)
    temp_hb = np.copy(Hb_probabilities)
    temp_hb[:trunc_ix] = 0.0
    h_est = np.cumsum(temp_hb / x_pts)
    h_est = h_est / h_est[-1]
    return h_est


def estimate_size(observed_areas, reference_sample, debias=True, algorithm="icm_em",
                  tol=0.0001, stop_iterations=10, em_max_iterations=None):
    """
    A function for estimating the size distribution CDF given a sample of observed section areas.
    The so-called length-biased size distribution CDF $H^b$ is estimated via nonparametric maximum likelihood.
    If 'debias' is set to true and additional step is performed and the estimate of $H^b$ is used to
    estimate the size distribution CDF. For details of the interpretation of these distributions and the estimation
    procedure see the paper: "Stereological determination of particle size distributions".

    NOTE: The given sample of section areas should not contain any duplicates. Within the model setting this
    is a probability zero event. In practice it may be the case that this does occur, especially when the
    measurement device used to obtain the data does not have a very high resolution. If this kind of data
    is presented to this function, a very negligible amount of noise is added to the data, to ensure unique values.
    This functionality is not extensively tested, you may mannually add a negligible amount of noise to your data for
    more control.

    :param observed_areas: A sample of observed section areas, a numpy.ndarray of shape (n,) where n is the sample size.
    :param reference_sample: A sample of section areas of the reference shape, a numpy.ndarray of shape (N,) where N is
        the sample size. Ideally N is very large.
    :param debias: A boolean, if True the size distribution is estimated, sometimes called number weighted size
        distribution. If False, the length-biased size distribution is estimated, may be interpreted as diameter
        weighted size distribution.
    :param algorithm: A string, one of "icm_em", "icm" or "em". Unless testing algorithm performance, it is best to
        leave this at the default setting.
    :param tol: A double, the tolerance used for stopping the optimization procedure (maximum likelihood). If the
        largest change in probability mass between estimates of successive iterations is below 'tol' for
        'stop_iterations' successive iterations the algorithm is terminated.
    :param stop_iterations: An integer, indicating for how many iterations the optimization procedure should run such
        that the largest change in probability mass of the estimate compared to the previous estimate is below 'tol'.
    :param em_max_iterations: An integer, only used if algorithm="em". The amount of iterations to be run by EM.
    :return: Two numpy arrays: x_pts, y_pts. Both are a numpy.ndarray of shape (n,). These arrays represent a piece-wise
        constant distribution function. The CDF is constant on [x_pts[i],x_pts[i+1]). Here: [a,b) = {x: a <= x < b}.
        y_pts[i] is the CDF value in x_pts[i].
    """
    sqrt_sample = np.sqrt(observed_areas)
    n = len(sqrt_sample)
    rng = np.random.default_rng(0)
    sigma = np.std(sqrt_sample)

    if len(np.unique(sqrt_sample)) < n:
        print("Warning: input contains duplicate values, adding a small amount of noise.")
        sqrt_sample = np.abs(sqrt_sample + rng.normal(loc=0, scale=0.000001*sigma, size=n))
    sqrt_sample = np.sort(sqrt_sample)

    # compute kernel density estimate of g_K^S
    kde_x, kde_y = approx_area_density(np.sqrt(reference_sample), sqrt_data=True)
    cs = CubicSpline(kde_x, kde_y)

    def gs_density(x):
        y = cs(x)
        y[y < 0] = 0.0
        y[x > np.max(kde_x * 1.05)] = 0.0
        y[x < 0] = 0.0
        return y

    # compute matrix alpha_{i,j}
    mat = np.broadcast_to(sqrt_sample, (n, n))
    input_mat = mat.T / mat
    data_matrix = np.reshape(gs_density(input_mat.flatten()), (n, n)) / mat

    # Compute MLE
    if algorithm == "icm_em":
        est = hybrid_icm_em(data_matrix, tol=tol, stop_iterations=stop_iterations)
    elif algorithm == "icm":
        est = icm(data_matrix, tol=tol, stop_iterations=stop_iterations)
    elif algorithm == "em":
        if em_max_iterations is None:
            em_max_iterations = 3 * n
        est = em(data_matrix, iterations=em_max_iterations)

    # Perform debiasing step if required
    if debias:
        est = de_bias(sqrt_sample, est, reference_sample)

    return sqrt_sample, est
