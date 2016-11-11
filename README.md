# pyGMMis

We extend the common mixtures-of-Gaussians density estimation approach to account for a known sample incompleteness by repeated imputation from the current model. The method generalizes existing Expectation-Maximization techniques for truncated data to arbitrary truncation geometries and probabilistic rejection. It can also incorporate independent multivariate normal measurement errors for each of the observed samples, and recover an estimate of the error-free distribution from which both observed and unobserved samples are drawn. The code is pure python, parallelized, and capable of performing density estimation with millions of samples and thousands of model components.

More details in the forthcoming paper by Melchior & Goulding.
