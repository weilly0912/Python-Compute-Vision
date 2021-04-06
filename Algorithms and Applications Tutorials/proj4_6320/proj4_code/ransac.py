import numpy as np
import math
from least_squares_fundamental_matrix import solve_F
import two_view_data
import fundamental_matrix


def calculate_num_ransac_iterations(prob_success, sample_size, ind_prob_correct):
    """
    Calculate the number of RANSAC iterations needed for a given guarantee of success.

    Args:
    -   prob_success: float representing the desired guarantee of success
    -   sample_size: int the number of samples included in each RANSAC iteration
    -   ind_prob_success: float the probability that each element in a sample is correct

    Returns:
    -   num_samples: int the number of RANSAC iterations needed

    """

    #######################################################################
    # YOUR CODE HERE                                                      #
    #######################################################################
    num_samples = math.ceil(math.log10(1 - prob_success) / math.log10(1 - ind_prob_correct ** sample_size))
    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################

    return int(num_samples)


def find_inliers(x_0s, F, x_1s, threshold):
    """ Find the inliers' indices for a given model.

    There are multiple methods you could use for calculating the error
    to determine your inliers vs outliers at each pass. However, we suggest
    using the line to point distance function we wrote for the
    optimization in part 2.

    Args:
    -   x_0s: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from the left image
    -   F: The proposed fundamental matrix
    -   x_1s: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from the right image
    -   threshold: the maximum error for a point correspondence to be
                    considered an inlier
    Each row in x_1s and x_0s is a proposed correspondence (e.g. row #42 of x_0s is a point that
    corresponds to row #42 of x_1s)

    Returns:
    -    inliers: 1D array of the indices of the inliers in x_0s and x_1s

    """

    #######################################################################
    # YOUR CODE HERE                                                      #
    #######################################################################
    
    projected_2d = projection(F, x_1s)

    distances = np.linalg.norm(projected_2d - x_0s, axis=1)

    indices = np.where(distances < threshold)[0]
    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################

    return inliers


'''
weilly
'''
"""Functions to compute projection matrix using DLT."""
import numpy as np


def generate_homogenous_system(pts2d: np.ndarray,
                               pts3d: np.ndarray) -> np.ndarray:
    """Generate a matrix A s.t. Ap=0. Follow the convention in the jupyter
    notebook and process the rows in the same order as the input, i.e. the
    0th row of input should go to 0^th and 1^st row in output.
    Note: remember that input is not in homogenous coordinates. Hence you need 
    to append w=1 for all 2D inputs, and the same thing for 3D inputs.
    Args:
        pts2d: numpy array of shape nx2, with the 2D measurements.
        pts3d: numpy array of shape nx3, with the actual 3D coordinates.
    """

    n = pts2d.shape[0]

    A = np.zeros((2*n, 12))

    ############################################################################
    # TODO: YOUR CODE HERE
    ############################################################################
    for i in range(n):
        x, y, z = pts3d[i, 0], pts3d[i, 1], pts3d[i, 2]
        u, v = pts2d[i, 0], pts2d[i, 1]
        A[i * 2,:] = np.array([0, 0, 0, 0, -x, -y, -z, -1, v * x, v * y, v * z, v])
        A[i * 2 + 1, :] = np.array([x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u])
    ############################################################################
    #                             END OF YOUR CODE
    ############################################################################

    return A


def get_eigenvector_with_smallest_eigenvector(A: np.ndarray) -> np.ndarray:
    """Get the unit normalized eigenvector corresponding to the minimum 
    eigenvalue of A.
    Hints: you may want to use np.linalg.svd.
    Note: please work out carefully if you need to access a row or a column to 
    get the required eigenvector from the SVD results.
    Args:
        A: the numpy array of shape p x q, for which the eigenvector is to be computed.
    Returns:
        eigenvec: the numpy array of shape (q,), the computed eigenvector of the minimum eigenvalue, 
        (note: just a single eigenvector).
    """

    eigenvec = np.empty(A.shape[0])

    ############################################################################
    # TODO: YOUR CODE HERE
    ############################################################################
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    eigenvec = vh[-1,:]
    ############################################################################
    #                             END OF YOUR CODE
    ############################################################################

    return eigenvec


def estimate_projection_matrix_dlt(pts2d: np.ndarray,
                                   pts3d: np.ndarray) -> np.ndarray:
    """Estimate the projection matrix using DLT.
    Note: 
    1. Scale your projection matrix estimate such that the last entry is 1.
    Args:
        pts2d: numpy array of shape nx2, with the 2D measurements.
        pts3d: numpy array of shape nx3, with the actual 3D coordinates.
    Returns:
        estimated projection matrix of shape 3x4.
    """

    assert pts2d.shape[0] >= 6

    A = generate_homogenous_system(pts2d, pts3d)

    eigenvec = get_eigenvector_with_smallest_eigenvector(A)

    P = eigenvec.reshape(3, 4)

    # scaling P so that the 12th entry is 1
    P = P/P[2, 3]

    return P

def ransac_fundamental_matrix(x_0s, x_1s):
    """Find the fundamental matrix with RANSAC.

    Use RANSAC to find the best fundamental matrix by
    randomly sampling interest points. You will call your
    solve_F() from part 2 of this assignment
    and calculate_num_ransac_iterations().

    You will also need to define a new function (see above) for finding
    inliers after you have calculated F for a given sample.

    Tips:
        0. You will need to determine your P, k, and p values.
            What is an acceptable rate of success? How many points
            do you want to sample? What is your estimate of the correspondence
            accuracy in your dataset?
        1. A potentially useful function is numpy.random.choice for
            creating your random samples
        2. You will want to call your function for solving F with the random
            sample and then you will want to call your function for finding
            the inliers.
        3. You will also need to choose an error threshold to separate your
            inliers from your outliers. We suggest a threshold of 1.

    Args:
    -   x_0s: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from the left image
    -   x_1s: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from the right image
    Each row is a proposed correspondence (e.g. row #42 of x_0s is a point that
    corresponds to row #42 of x_1s)

    Returns:
    -   best_F: A numpy array of shape (3, 3) representing the best fundamental
                matrix estimation
    -   inliers_x_0: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from the left image that are inliers with
                   respect to best_F
    -   inliers_x_1: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from the right image that are inliers with
                   respect to best_F

    """

    #######################################################################
    # YOUR CODE HERE                                                      #
    #######################################################################
    from projection_matrix import projection, estimate_camera_matrix
    num_input_points = x_0s.shape[0]
    num_iterations = 100

    best_P = np.random.rand(3, 4)
    best_inlier_count = 0
    inliers_pts2d = np.array([])
    inliers_pts3d = np.array([])

    for _ in range(num_iterations):
        # randomly sample 6 correspondences
        idxes = np.random.choice(num_input_points, size=6, replace=False)

        sampled_pts2d = x_0s[idxes]
        sampled_pts3d = x_1s[idxes]

        ########################################################################
        # TODO: YOUR CODE HERE
        ########################################################################
        inliers = find_inliers(x_0s, x_1s, estimate_camera_matrix(sampled_pts2d, sampled_pts3d, estimate_projection_matrix_dlt(sampled_pts2d, sampled_pts3d)), inlier_threshold)
        P_sample = estimate_camera_matrix(sampled_pts2d, sampled_pts3d, estimate_projection_matrix_dlt(sampled_pts2d, sampled_pts3d))
        ########################################################################
        #                             END OF YOUR CODE
        ########################################################################

        if len(inliers) > best_inlier_count:
            best_inlier_count = len(inliers)
            inliers_pts2d = x_0s[inliers]
            inliers_pts3d = x_1s[inliers]
            best_P = P_sample

    print('Found projection matrix with support ', best_inlier_count)

    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################

    return best_F, inliers_x_0, inliers_x_1


def test_with_epipolar_lines():
    """Unit test you will create for your RANSAC implementation.

    It should take no arguments and it does not need to return anything,
    but it **must** display the images when run.

    Use the code in the jupyter notebook as an example for how to open the
    image files and perform the necessary operations on them in our workflow.
    Remember the steps are Harris, SIFT, match features, RANSAC fundamental matrix.

    Display the proposed correspondences, the true inlier correspondences
    found by RANSAC, and most importantly the epipolar lines in both of your images.
    It should be clear that the epipolar lines intersect where the second image
    was taken, and the true point correspondences should indeed be good matches.

    """

    #######################################################################
    # YOUR CODE HERE                                                      #
    #######################################################################

    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################
