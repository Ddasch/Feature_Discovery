# Feature Discovery

Feature Discovery is a machine learning library which aims to make it easier to find non-linear representations of 
your data that are beneficial for a specific classification task. Instead of manually having to look at a mountain of 
scree plots whilst trying to figure out how to engineer your feature space, Feature Discovery does the gruntwork for you 
and tries to come up with a list of promising transformations of the input features for you to look into
in more detail.   

# Background

A commonly known method for building machine learning models is the kernel method. The idea is to take a data set that
isn't linearly separable and make it separable by transforming the data into a higher dimensional space that is.

<div align="center"><img src="https://github.com/Ddasch/Feature_Discovery/blob/develop/docs/images/2-Figure1-1.png?raw=true " width="500"/></div>

By using a feature map &phi;(x) you can transform your data into a space that is easier for a machine learning model to 
tackle. Now a Support Vector Machine would train a linear model using &phi;(x) and call it a day[^1]. However, if applying
&phi; is beneficial for linear models, it must be beneficial for other models as well. 

# How does it work?
By applying a series feature maps &phi; to features / feature-pairs of your data and measuring how much a particular map
would improve the linear separability of the data. You provide the data and a list of features to look into and kernels
to try, and Feature Discovery tries them all and compiles its findings into easily digestible plots or rankings. 

## Currently Supported Kernels
### Monovariate
- Quadratic
- Sigmoid
- Logarithmic (shifted & cropped)
### Duovariate
- Difference
- Magnitude
- Polynomial (second and third order)
- Random Fourier Functions
- Gaussian (RFF approximation)
 


[^1]: Some kernels like RBF don't do this explicitly and compute the instance similarities differently.

