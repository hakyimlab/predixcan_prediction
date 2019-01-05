#!/bin/bash

mkdir -p prediction_output

parallel -j3 \
'python predict.py \
  --bgens-dir /mnt/ukb_v3/imp/ \
  --bgens-prefix ukb_imp_chr \
  --bgens-sample-file /mnt/ukb_v3/link_files/ukb19526_imp_chr1_v3_s487395.sample \
  --weights-file {} \
  --output-file prediction_output/{/.}.h5 \
  --bgens-n-cache 250 \
  --bgens-writing-cache-size 500' \
  ::: /mnt/software/gtex_v8_models/*.db

