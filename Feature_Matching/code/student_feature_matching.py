import numpy as np
from scipy.spatial.distance import cdist


def match_features(features1, features2, x1, y1, x2, y2):
    """
    This function does not need to be symmetric (e.g. it can produce
    different numbers of matches depending on the order of the arguments).

    To start with, simply implement the "ratio test", equation 4.18 in
    section 4.1.3 of Szeliski. There are a lot of repetitive features in
    these images, and all of their descriptors will look similar. The
    ratio test helps us resolve this issue (also see Figure 11 of David
    Lowe's IJCV paper).

    For extra credit you can implement various forms of spatial/geometric
    verification of matches, e.g. using the x and y locations of the features.

    Args:
    -   features1: A numpy array of shape (n,feat_dim) representing one set of
            features, where feat_dim denotes the feature dimensionality
    -   features2: A numpy array of shape (m,feat_dim) representing a second set
            features (m not necessarily equal to n)
    -   x1: A numpy array of shape (n,) containing the x-locations of features1
    -   y1: A numpy array of shape (n,) containing the y-locations of features1
    -   x2: A numpy array of shape (m,) containing the x-locations of features2
    -   y2: A numpy array of shape (m,) containing the y-locations of features2

    Returns:
    -   matches: A numpy array of shape (k,2), where k is the number of matches.
            The first column is an index in features1, and the second column is
            an index in features2
    -   confidences: A numpy array of shape (k,) with the real valued confidence for
            every match

    'matches' and 'confidences' can be empty e.g. (0x2) and (0x1)
    """
    #############################################################################
    # TODO: YOUR CODE HERE                                                        #
    #############################################################################

    #compute distance between each pair of feature
    distance = cdist(features1, features2, 'cityblock')
    sort_index = np.argsort(distance, axis = 1)

    #Set a threshold for ratio test
    threshold = 0.8
    match_confidence = []
    
    #implement the "ratio test", equation 4.18 in section 4.1.3 of Szeliski.
    for i in range(features1.shape[0]):
        dist = distance[i]
        index = sort_index[i]

        #check with threshold
        if dist[index[0]] < dist[index[1]]*threshold:
            ratio = dist[index[0]]/(dist[index[1]]+0.0001)
            match_confidence.append([i, index[0], 1-ratio])

    match_confidence = np.array((match_confidence))       
    match_confidence = match_confidence[match_confidence[:,2].argsort()]
    match_confidence = np.flipud(match_confidence)

    matches = match_confidence[:,0:2].astype(int)

    confidences = match_confidence[:,2]

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return matches, confidences

