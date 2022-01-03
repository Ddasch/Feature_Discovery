# Feature Discovery

Feature Discovery is a machine learning library which aims to make it easier to find non-linear representations of 
your data that are beneficial for a specific classification task. Instead of manually having to look at a mountain of 
scree plots whilst trying to figure out how to engineer your feature space, Feature Discovery does the gruntwork for you 
and tries to come up with a list of promising transformations of the input features for you to look into
in more detail.   

# Why?

Feature engineering is a rather laborious task that needs a lot of domain knowledge and trial&error before you can find
quality transformations for a given ML task. Many online tutorials will tell you simply to "look at the data" in order
to realize what the perfect transformation is (or kernel if you're intending to use a SVM). 
Problem is, they do so with a dataset that only has 2 features. What if your dataset has 50 features? 
That's 50*49/2 = 1225 different scree plots to look at! Ain't nobody got time for that! Now what if we had some kind
of tool what would look through your dataset and come up with a prioritized list of promising feature-pairs to look at?
This is exactly what Feature Discovery aims to provide. With just a few lines of code, you can have all the features
of your dataset checked against common (kernel) transformations and get a quick overview as to where to engineer better
features from first. You provide the data, a list of features to look into and kernels
to try, and Feature Discovery tries them all and compiles its findings into easily digestible plots or rankings. 

# Background

A commonly known method for building machine learning models is the kernel method. The idea is to take a data set that
isn't linearly separable and make it separable by transforming the data into a higher dimensional space that is.

<div align="center"><img src="https://github.com/Ddasch/Feature_Discovery/blob/develop/docs/images/2-Figure1-1.png?raw=true " width="500"/></div>

By using a feature map &phi;(x) you can transform your data into a space that is easier for a machine learning model to 
tackle. Now a Support Vector Machine would train a linear model using &phi;(x) and call it a day[^1]. However, if applying
&phi; is beneficial for linear models, it must be beneficial for other models as well and therefore be a good indicator
of the hidden potential of a feature. 


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
 
# Installation

TODO

# How to use

```py
>>> from featurediscovery import kernel_search
>>> df =  <your pandas dataframe>
# specify the columns in df that are your feature space
>>> feature_space = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5']
# speficy which column in df is your target variable
>>> target_variable = 'y'
>>> kernel_search.evaluate_kernels(df
                     , target_variable='y'
                     , feature_space=feature_space
                     , monovariate_kernels=['quadratic', 'sigmoid', 'log', 'log_shift']
                     , duovariate_kernels=['poly3', 'rff_gauss']
                     , feature_standardizers=['raw']
                     , plot_feature_ranking=True
                     , plot_ranking_all_transformations=True
                     , eval_method='normal'
                     )
```



```py

>>> kernel_search.evaluate_kernels(df
                     , target_variable='y'
                     , feature_space=feature_space
                     , monovariate_kernels=['quadratic', 'sigmoid', 'log', 'log_shift']
                     , duovariate_kernels=['difference', 'magnitude','poly2', 'poly3', 'rff_gauss']
                     , feature_standardizers=['raw','centralized', 'standard', 'minmax']
                     , plot_feature_ranking=True/False
                     , plot_ranking_all_transformations=True/False
                     , plot_individual_kernels=True/False
                     , export_folder='<path to folder>'
                     , export_ranking=True/False
                     , export_formats=['png', 'json', 'csv']
                     , export_individual_kernel_plots=True/False
                     , eval_method='normal'
                     , use_cupy='no'
                     )
```

[^1]: Some kernels like RBF don't do this explicitly and compute the instance similarities differently. For these an
explicit mapping can only be approximated.

