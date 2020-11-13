# PrediXcan - Gene expression prediction scripts

Scripts to predict gene expression given PrediXcan models and genotype data.
The current script recognize the SNPs by rsID, so we require both the genotype and the predictdb file having SNP labelled with rsID.

# Create conda environment

```
conda env create -f environment.yml
conda activate predixcan_prediction  # load the environment just built
wget http://www.well.ox.ac.uk/~gav/resources/rbgen_<version>.tgz . # install extra dependency of R package rbgen
Rscript -e "install.packages( 'rbgen_<version>.tgz', repos = NULL )"
```

See extra documentation of `rbgen` and `bgen` [here](https://bitbucket.org/gavinband/bgen/wiki/browse/)

**CAUTION**: specifically, we may need to stick with pandas 0.23.4 because of [this issue](https://github.com/theislab/single-cell-tutorial/issues/7).

# Index BGEN

The script relies on `rbgen` which needs your BGEN files being indexed by `bgenix`. 
See details [here](https://bitbucket.org/gavinband/bgen/wiki/bgenix).

# See also

The script was mainly developed by @miltondp and see the original GitHub repository for more details [https://github.com/miltondp/predixcan_prediction](https://github.com/miltondp/predixcan_prediction).
There were some enhancements and functionalities added by @liangyy and see more details at [https://github.com/liangyy/predixcan_prediction](https://github.com/liangyy/predixcan_prediction).

These repositories above are considered to be inactive.
Please report bug and issue at the current respository.

# Example run on UKB imputed genotype (v3)

The performance of the script will depend on some parameter settings. 
In the following, we listed example run for UKB imputed genotype v3. 
Depending on the size of predictdb input, the running time may differ.
To give your a reference, it takes about 26 hours to predict the whole transcriptome of UKB samples using GTEx v8 Muscle Skeletonal models.

**Predicting the whole transcriptome**.

```
# conda activate predixcan_prediction
python [path-to-script]/predict.py \
  --bgens-dir [path-to-bgen] \
  --bgens-bgi-dir [path-to-bgen-bgi] \
  --bgens-prefix ukb_imp_chr{chr_num}_v3 \
  --bgens-sample-file [path-to-bgen-sample-file] \
  --weights-file [path-to-predictdb] \
  --output-file [path-to-output-hdf5] \
  --bgens-n-cache 250 \
  --bgens-writing-cache-size 500 \
  --autosomes \
  --max-sample-chunk-size 10000 \
  --max-gene-chunk-size 10
```

**Predicting a list of genes**: gene list should be a text file with the gene ID (consistent with predictdb input) where each row is for one gene. 

```
# conda activate predixcan_prediction
python [path-to-script]/predict.py \
  --bgens-dir [path-to-bgen] \
  --bgens-bgi-dir [path-to-bgen-bgi] \
  --bgens-prefix ukb_imp_chr{chr_num}_v3 \
  --bgens-sample-file [path-to-bgen-sample-file] \
  --weights-file [path-to-predictdb] \
  --output-file [path-to-output-hdf5] \
  --bgens-n-cache 250 \
  --bgens-writing-cache-size 500 \
  --autosomes \
  --max-sample-chunk-size 10000 \
  --max-gene-chunk-size [N] \
  --gene-list [path-gene-list]
```

If you are predicting a few genes, set `[N] = 1`. And if you are predicting a lot of genes (the number is comparable to the whole transcriptome), set `[N] = 1`.
