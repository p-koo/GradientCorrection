# Gradient Correction 

This code repository contains code and access to data to reproduce the analysis from "Correcting gradient-based interpretations of deep neural networks for genomics" by Majdandzic, Rajesh, and Koo. Code in this repository is shared under the MIT License. 

For questions, email: koo@cshl.edu

Download model weights and data from Zenodo: https://doi.org/10.5281/zenodo.7011631


The gradient correction can be implemented easily in numpy:

```
attr_map = attribution_method_of_interest(x)  # shape (N,L,A) 
attr_map /= np.mean(attr_map, axis=2, keepdims=True)

```
