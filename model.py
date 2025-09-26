import numpy as np

class PCA:
    """
    Principal Component Analysis (PCA) class.
    
    PCA is a dimensionality reduction technique that is commonly used in machine learning and data visualization.
    It uses the Singular Value Decomposition (SVD) to project data to a lower dimensional space.
    
    Attributes:
    n_components (int): The number of principal components to keep. If not set, all components are kept.
    components_ (ndarray): Principal axes in feature space, representing the directions of maximum variance in the data.
    explained_variance_ (ndarray): The amount of variance explained by each of the selected components.
    explained_variance_ratio_ (ndarray): Percentage of variance explained by each of the selected components.
    X (ndarray): Original data, centered but not scaled.
    mean_ (ndarray): Per-feature empirical mean, estimated from the training set.
    """
    def __init__(self, n_components = None):
        """
        Initialize the PCA model.
        
        Parameters:
        n_components (int, optional): The number of components to keep. If not set, all components are kept.
        """
        self.n_components = n_components
        self.components_ = None # ndarray of shape (n_components, n_features)
        self.explained_variance_ = None # ndarray of shape (n_components,)
        self.explained_variance_ratio_ = None # ndarray of shape (n_components,)
        self.X = None # ndarray of shape (n_samples, n_features)
        self.mean_ = None # ndarray of shape (n_features,)

    def centering(self, X):
        """
        Center the matrix X by subtracting the mean of its features.
        
        Parameters:
        X (ndarray): The data matrix of shape (n_samples, n_features).
        """
        self.mean_ = np.mean(X, axis=0)
        self.X = X - self.mean_

    def fit(self, X):
        """
        Fit the PCA model with the matrix X.
        
        This method computes the principal components of the matrix X and stores them in the `components_` attribute.
        It also computes the explained variance and the explained variance ratio and stores them in the `explained_variance_` and `explained_variance_ratio_` attributes, respectively.
        
        Parameters:
        X (ndarray): The data matrix of shape (n_samples, n_features).
        """
        if self.n_components is None:
            self.n_components = min(X.shape[0], X.shape[1])
            print(f"n_components is not set. Setting n_components = {self.n_components}")
        
        self.centering(X)
        print("Performing SVD...")
        U, S, Vh = np.linalg.svd(self.X, full_matrices=False)
        print("SVD done.")
        
        # if there is nan values in S, replace them with 0
        if np.isnan(S).any():
            S[np.isnan(S)] = 0
            print("There are nan values in S. Replaced them with 0. Please check your data.")

        self.components_ = Vh[:self.n_components,:]
        print("Principal Directions are stored in components_")
        self.explained_variance_ = S**2/(X.shape[0]-1)
        self.explained_variance_ratio_ = self.explained_variance_ / np.sum(self.explained_variance_) 

    def transform(self, X):
        """
        Apply the dimensionality reduction on X.
        
        This method applies the dimensionality reduction to the matrix X by projecting it on the principal components.
        
        Parameters:
        X (ndarray): The data matrix of shape (n_samples, n_features).
        
        Returns:
        X_new (ndarray): The transformed data matrix of shape (n_samples, n_components).
        """

        if self.components_ is None:
            raise ValueError("Please fit the model first.")
        print(f"Transforming X to {self.n_components} Principal Components")
        return self.X @ self.components_.T # ndarray of shape (n_samples, n_components)
    
    def fit_transform(self, X):
        """
        Fit the PCA model with the matrix X and then apply the dimensionality reduction on X.
        
        This method is a convenience method that fits the model and then applies the dimensionality reduction.
        
        Parameters:
        X (ndarray): The data matrix of shape (n_samples, n_features).
        
        Returns:
        X_new (ndarray): The transformed data matrix of shape (n_samples, n_components).
        """
        self.fit(X)
        return self.transform(X)
