## Sparkling parameters and regularisation term
The goal of the scripts here are to study the impact of the sparkling density
used to sample our MRI image, on the regularisation parameter over

* the script `sparkling_stats.py` will take as an entry a value for lambda,
a regularisation parameter. In the script, you can define your image dataset,
the kind of sampling pattern and the param grid (decay, tau) on which you will
reconstruct you image and study the reconstruction quality.

You can declare the reconstruction function with the wrapper designed in
`wrapper_pysap.py`.

* To launch multiple lambda/tau/decay studies, you can use the bash script
 `lambda_sparkling.sh`, just put the list of lambda you want to test

 Each lambda test will generate a dic with the metric (+std, +min, +max)
 computed over the N images of the database, for each set of (tau, decay)

* `variable_separability.py` In this Python script, you can put the list of
lambda dic you want to compare. It will plot in 3D the evolution of one metric
(default: SSIM) over the sampling parameters (tau, decay)

but also 2D plot with the evolution of the metric with one
parameter (tau, decay, lambda) fixed
