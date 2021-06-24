# Implementation of KMeans using numpy
# Author: Diederik Sloet
# Date: 24 June 2021

# Libraries

import numpy as np
import pandas as pd

# we need data to build kmeans on.


def create_data(data_points=100, R=1, min=0, max=1, center_x=0, center_y=0):
    r = R * np.sqrt(np.random.uniform(min, max, size=data_points))
    theta = np.random.uniform(min, max, size=data_points) * 2 * np.pi

    x = center_x + np.cos(theta) * r
    y = center_y + np.sin(theta) * r

    x = np.round(x, 3)
    y = np.round(y, 3)

    df = np.column_stack([x, y])
    df = pd.DataFrame(df)
    df.columns = ["x", "y"]


# Use it to create data:
data1 = create_data(data_points=20, R=10, center_x=5, center_y=30)
data2 = create_data(data_points=20, R=10, center_x=20, center_y=10)
data3 = create_data(data_points=20, R=10, center_x=50, center_y=50)

data = data1.append(data2).append(data3).reset_index(drop=True)


class KMeans:
    """Implements the K-Means algorithm

    Uses euclidean distance* to calculate the error between centroids
    and data samples.

    Process is to first initialise K centroids. Then it calculates the distances
    of each data point to each centroid. The closest (shortest distance) centroid
    gets assigned to the data point. Next, the mean of the data points associated
    to a centroid becomes the new centroid and the previous steps are repeated
    until the sum of all the distances does not change anymore. The algorithm is
    then converged.

    *The word distance is wrongly chosen. In fact we calculate the sum of
    squared residuals but it looks a lot like the euclidean distance.

    ...

    Attributes:
    -----------
    data : pd.DataFrame
      Dataframe with the origina data.

    centroids : pd.DataFrame
      DataFrame with the fitted centroids and the mean error.

    _total_error : float
      The sum of all the errors


    Methods:
    --------

    fit():
      Iterates over the max_iters until the _total_error doesn't change anymore.
      Once the algorithm has converged it has calculated the optimal centroids.

    predict():
      Returns a pd.DataFrame with the original data plus a column with the
      associated centroid for each data point and the distance/ sum of squares
      from the data point to the centroid it is associated with.



    """

    def __init__(self, K, max_iters=5):
        """
        Constructs the initial attributes of KMeans class.

        Parameters:
        -----------
        data : pd.DataFrame, required
          The data containing the data that should be clustered.

        K : int, required
          The K number of clusters.

        max_iters : int, required
          How often (iterations) the algorithm should try to converge
          to the minimum error.
          Default = 5

        """
        self.data = None
        self.K = K  # K number of clusters
        self.max_iters = max_iters

        self.centroids = None
        self.intertia_ = None

    def fit(self, data):
        """Main method to fit the algorithm.

        Iterates over the max_iters until the _total_error doesn't change anymore.
        Once the algorithm has converged it has calculated the optimal centroids.

        Returns : None

        """
        self.data = data.copy()
        self._shape = data.shape[1]

        # initialize centroids
        self._initialize_centroids()
        count_until_concergence = 0  # counter until convergence

        # loop over max_iters
        for i in range(self.max_iters):

            new_centroids = self._assign_centroids()
            self.data["centroid"] = new_centroids["centroid"]
            self.data["error"] = new_centroids["error"]

            self._calc_new_centroids()
            new_error = new_centroids.error.sum()
            if self.intertia_ != new_error:
                self.intertia_ = new_error
                count_until_concergence += 1

            else:
                print(f"KMeans has converged in {count_until_concergence} steps.")
                print(f"Intertia = {self.intertia_}")
                break

    def predict(self, data):
        """Get dataframe with the data inclusing the centroids."""
        self.data = data.copy()
        new_centroids = self._assign_centroids()
        self.data["centroid"] = new_centroids["centroid"]
        self.data["error"] = new_centroids["error"]
        return self.data

    def _initialize_centroids(self):
        """Initialises the centroids

        Randomly selects K samples from the data and uses those as the initial
        centroids. By selecting a data point, the centroids are within the
        range of the original data.

        """

        # create K centroids within the range of the data

        data_idx = self.data.index
        sample = np.random.choice(data_idx, size=self.K, replace=False)
        self.centroids = self.data.loc[sample].reset_index(drop=True)

    def _assign_centroids(self, data=None):
        """Main helper to assign centroids to each data point."""

        # first calculate the errors between the data and the centroids

        centroid_assign = []  # The centroid the data point gets assigned
        centroid_errors = []  # The error of the data point to the assigned centroid

        # loop over each data point
        for idx in range(self.data.shape[0]):

            # error list
            errors = []
            # for each centroid, calculate the sum of squared residuals.
            for cent in range(self.centroids.shape[0]):
                error = self._calc_sum_sq(
                    self.centroids.iloc[cent, : self._shape],
                    self.data.iloc[idx, : self._shape],
                )
                errors.append(error)
            # print(idx, errors)

            # Find closest centroid for the data point and assign it to list
            closest_centroid = np.where(errors == np.amin(errors))[0].tolist()[0]
            # find the error associated to the centroid
            centroid_error = np.amin(errors)
            # print(closest_centroid, centroid_error)

            # assign values to list
            centroid_assign.append(closest_centroid)
            centroid_errors.append(centroid_error)

        return pd.DataFrame({"centroid": centroid_assign, "error": centroid_errors})

    def _calc_sum_sq(self, a, b):
        """Calculates the sum of squared residuals.

        This resembles the euclidean distance but it is in fact the sum of squares.

        """
        return np.sum(np.square(a - b))

    def _calc_new_centroids(self):
        """Calculates the mean of the datapoints associated to a centroid.

        The mean of each cluster of data point becomed the new centroid.
        """
        self.centroids = (
            self.data.groupby("centroid").agg("mean").reset_index(drop=True)
        )


# Using KMeans on the data.

kmeans = KMeans(K=3, max_iters=10)
kmeans.fit(data=data)


# We can also use the Elbow-technique to find the best number of K:

errors = []
for i in np.arange(1, 10, 1):
    kmeans = KMeans(K=i)
    kmeans.fit(data=data)
    error = kmeans.intertia_
    errors.append(error)
