#!/bin/bash

# example for whole blood

python predict.py \
  --bgens-dir /mnt/ukb_v3/imp/ \
  --bgens-sample-file /mnt/ukb_v3/link_files/ukb19526_imp_chr1_v3_s487395.sample \
  --weights-file /mnt/software/gtex_v8_models/gtex_v8_Whole_Blood_itm_signif.db \
  --output-file whole_blood.h5 \
  --bgens-n-cache 1000                                     

