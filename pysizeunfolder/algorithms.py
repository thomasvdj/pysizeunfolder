from sklearn.isotonic import IsotonicRegression
import numpy as np


# Author: Thomas van der Jagt


def gcm_slope(rawslope, dx):
    np.nan_to_num(rawslope, copy=False)
    slope = IsotonicRegression().fit_transform(np.arange(len(dx)), rawslope, sample_weight=dx)
    slope[slope < 0] = 0.0

    _, idx = np.unique(slope, return_index=True)
    indices_keep = np.sort(idx)

    return slope, indices_keep


def likelihoods_vector(a_matrix, distribution_vector):
    return np.dot(a_matrix, np.append(distribution_vector[0], np.diff(distribution_vector)))


def likelihoods_vector_sparse(a_matrix, distribution_vector, indices):
    return np.dot(a_matrix[:, indices], np.diff(np.append(0, distribution_vector))[indices])


def hybrid_icm_em(a_matrix, estimate_init=None, tol=0.0001, stop_iterations=10):
    epsilon = 0.25
    n = a_matrix.shape[0]
    if estimate_init is None:
        estimate = np.linspace(1./n, 1, n)
        previous_estimate = np.linspace(1./n, 1, n)
    else:
        estimate = estimate_init
        previous_estimate = estimate_init
    diff_matrix = np.column_stack((np.diff(a_matrix), -a_matrix[:, -1])).T
    stop_counter = 0

    likelihoods = likelihoods_vector(a_matrix, estimate)

    loglikelihood = -np.mean(np.log(likelihoods)) + estimate[-1]

    divisor = np.broadcast_to(1./likelihoods, (n, n))
    ratio_matrix = diff_matrix * divisor
    first_derivatives = np.mean(ratio_matrix, axis=1)
    first_derivatives[-1] += 1
    second_derivatives = np.mean(np.square(ratio_matrix), axis=1)
    count = 0
    #indices = np.arange(n)

    while stop_counter < stop_iterations:

        rawslope = estimate - first_derivatives/second_derivatives
        proposed_estimate, proposed_indices = gcm_slope(rawslope, second_derivatives)

        # very important, we cannot simply scale estimate right before the EM step, because the EM step depends on the
        # likelihoods which correspond to the unscaled version of estimate. Removing the following line might cause
        # this algorithm to fail.
        proposed_estimate = proposed_estimate/np.max(proposed_estimate)

        proposed_likelihoods = likelihoods_vector_sparse(a_matrix, proposed_estimate, proposed_indices)
        with np.errstate(divide='ignore'):
            proposed_loglik = -np.mean(np.log(proposed_likelihoods)) + proposed_estimate[-1]

        if proposed_loglik < (loglikelihood + epsilon * np.dot(first_derivatives, proposed_estimate - estimate)):
            estimate = proposed_estimate
            likelihoods = proposed_likelihoods
            loglikelihood = proposed_loglik
            #indices = proposed_indices
        else:
            lmbda = 1.0
            s = 0.5
            z = np.copy(proposed_estimate)
            loglikelihood_z = proposed_loglik
            likelihoods_z = proposed_likelihoods
            #indices_z = np.union1d(proposed_indices, indices)

            temp_dot = np.dot(first_derivatives, z - estimate)
            cond1 = loglikelihood_z < loglikelihood + (1 - epsilon)*temp_dot
            cond2 = loglikelihood_z > loglikelihood + epsilon*temp_dot

            line_search_iteration = 0

            while (cond1 or cond2) and line_search_iteration < 100:
                if cond1:
                    lmbda += s
                if cond2:
                    lmbda -= s
                z = estimate + lmbda*(proposed_estimate - estimate)
                #likelihoods_z = likelihoods_vector3(a_matrix, z, indices_z)
                likelihoods_z = likelihoods_vector(a_matrix, z)
                loglikelihood_z = -np.mean(np.log(likelihoods_z)) + z[-1]
                temp_dot = np.dot(first_derivatives, z - estimate)
                cond1 = loglikelihood_z < loglikelihood + (1 - epsilon) * temp_dot
                cond2 = loglikelihood_z > loglikelihood + epsilon * temp_dot
                s *= 0.5
                line_search_iteration += 1

            if line_search_iteration < 100:
                estimate = z
                likelihoods = likelihoods_z
                loglikelihood = loglikelihood_z
                #indices = indices_z

        count += 1
        divisor = np.broadcast_to(1. / likelihoods, (n, n))

        # EM step
        em_probabilities = np.diff(np.append(0, estimate))
        em_probabilities = np.mean(a_matrix.T * divisor, axis=1) * em_probabilities
        estimate = np.cumsum(em_probabilities)

        #likelihoods = likelihoods_vector3(a_matrix, estimate, indices)
        likelihoods = likelihoods_vector(a_matrix, estimate)
        loglikelihood = -np.mean(np.log(likelihoods)) + estimate[-1]

        divisor = np.broadcast_to(1. / likelihoods, (n, n))
        ratio_matrix = diff_matrix * divisor
        first_derivatives = np.mean(ratio_matrix, axis=1)
        first_derivatives[-1] += 1
        second_derivatives = np.mean(np.square(ratio_matrix), axis=1)

        max_change = np.max(np.abs(estimate - previous_estimate))
        #print("Iteration:", count, "max change:", max_change)
        if max_change < tol:
            stop_counter += 1
        else:
            stop_counter = 0
        previous_estimate = estimate

    return estimate


def icm(a_matrix, estimate_init=None, tol=0.0001, stop_iterations=10):
    epsilon = 0.25
    n = a_matrix.shape[0]
    if estimate_init is None:
        estimate = np.linspace(1./n, 1, n)
        previous_estimate = np.linspace(1./n, 1, n)
    else:
        estimate = estimate_init
        previous_estimate = estimate_init
    diff_matrix = np.column_stack((np.diff(a_matrix), -a_matrix[:, -1])).T
    stop_counter = 0

    likelihoods = likelihoods_vector(a_matrix, estimate)
    loglikelihood = -np.mean(np.log(likelihoods)) + estimate[-1]

    divisor = np.broadcast_to(1./likelihoods, (n, n))
    ratio_matrix = diff_matrix * divisor
    first_derivatives = np.mean(ratio_matrix, axis=1)
    first_derivatives[-1] += 1
    second_derivatives = np.mean(np.square(ratio_matrix), axis=1)
    count = 0

    while stop_counter < stop_iterations:
        rawslope = estimate - first_derivatives/second_derivatives
        proposed_estimate, proposed_indices = gcm_slope(rawslope, second_derivatives)
        proposed_estimate = proposed_estimate/np.max(proposed_estimate)

        proposed_likelihoods = likelihoods_vector_sparse(a_matrix, proposed_estimate, proposed_indices)
        with np.errstate(divide='ignore'):
            proposed_loglik = -np.mean(np.log(proposed_likelihoods)) + proposed_estimate[-1]

        if proposed_loglik < (loglikelihood + epsilon * np.dot(first_derivatives, proposed_estimate - estimate)):
            estimate = proposed_estimate
            likelihoods = proposed_likelihoods
            loglikelihood = proposed_loglik
            #indices = proposed_indices
        else:
            lmbda = 1.0
            s = 0.5
            z = np.copy(proposed_estimate)
            loglikelihood_z = proposed_loglik
            likelihoods_z = proposed_likelihoods
            #indices_z = np.union1d(proposed_indices, indices)

            temp_dot = np.dot(first_derivatives, z - estimate)
            cond1 = loglikelihood_z < loglikelihood + (1 - epsilon)*temp_dot
            cond2 = loglikelihood_z > loglikelihood + epsilon*temp_dot

            line_search_iteration = 0

            while (cond1 or cond2) and line_search_iteration < 100:
                if cond1:
                    lmbda += s
                if cond2:
                    lmbda -= s
                z = estimate + lmbda*(proposed_estimate - estimate)
                likelihoods_z = likelihoods_vector(a_matrix, z)
                loglikelihood_z = -np.mean(np.log(likelihoods_z)) + z[-1]
                temp_dot = np.dot(first_derivatives, z - estimate)
                cond1 = loglikelihood_z < loglikelihood + (1 - epsilon) * temp_dot
                cond2 = loglikelihood_z > loglikelihood + epsilon * temp_dot
                s *= 0.5
                line_search_iteration += 1

            if line_search_iteration < 100:
                estimate = z
                likelihoods = likelihoods_z
                loglikelihood = loglikelihood_z
                #indices = indices_z

        count += 1

        divisor = np.broadcast_to(1. / likelihoods, (n, n))
        ratio_matrix = diff_matrix * divisor
        first_derivatives = np.mean(ratio_matrix, axis=1)
        first_derivatives[-1] += 1
        second_derivatives = np.mean(np.square(ratio_matrix), axis=1)

        max_change = np.max(np.abs(estimate - previous_estimate))
        if max_change < tol:
            stop_counter += 1
        else:
            stop_counter = 0
        previous_estimate = estimate

    return estimate/np.max(estimate)


def em(a_matrix, estimate_init=None, iterations=None):
    n = a_matrix.shape[0]
    if estimate_init is None:
        estimate = np.ones(n)/n
    else:
        estimate = estimate_init
    if iterations is None:
        its = 3*n
    else:
        its = iterations

    # Note: we do not have a clever stopping condition.
    for i in range(its):
        em_likelihoods = np.dot(a_matrix, estimate)
        estimate = np.mean(a_matrix.T * np.broadcast_to(1. / em_likelihoods, (n, n)), axis=1) * estimate

    return np.cumsum(estimate)
