# PrediXcan - Gene expression prediction scripts

Scripts to predict gene expression given PrediXcan models and genotype data.

# Create conda environment

```
$ conda env create -f environment.yml
$ conda activate predixcan_prediction  # load the environment just built
$ wget http://www.well.ox.ac.uk/~gav/resources/rbgen_<version>.tgz . # install extra dependency of R package rbgen
$ Rscript -e "install.packages( 'rbgen_<version>.tgz', repos = NULL )"
```

See extra documentation of `rbgen` and `bgen` [here](https://bitbucket.org/gavinband/bgen/wiki/browse/)

**CAUTION**: specifically, we may need to stick with pandas 0.23.4 because of [this issue](https://github.com/theislab/single-cell-tutorial/issues/7).

# Index BGEN

The script relies on `rbgen` which needs your BGEN files being indexed by `bgenix`. 
See details [here](https://bitbucket.org/gavinband/bgen/wiki/bgenix).
