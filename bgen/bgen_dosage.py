import sqlite3

import numpy as np
from bgen_reader import read_bgen, allele_expectation


class BGENDosage:
    def __init__(self, bgen_path, sample_path=None, cache_size=50, verbose=False):
        self.bgen_path = bgen_path
        self.bgi_path = self.bgen_path + '.bgi'
        self.sample_path = sample_path
        self.cache_size = cache_size

        self.bgen_obj = read_bgen(self.bgen_path, sample_file=self.sample_path, size=self.cache_size, verbose=verbose)

        with sqlite3.connect(self.bgi_path) as conn:
            self.variants_count = conn.execute('select count(*) from Variant').fetchone()[0]

            # FIXME: only one chromosome per BGEN file is supported
            self.chr_number = conn.execute('select distinct chromosome from Variant').fetchone()[0]

    def get_row(self, row_idx):
        row_number = (row_idx if row_idx >= 0 else self.variants_count + row_idx,)

        dosage_row = self.bgen_obj['variants'].iloc[row_number].rename({'chrom': 'chr', 'pos': 'position'})
        dosage_row['chr'] = int(dosage_row.chr)

        alleles = dosage_row.allele_ids.split(',')
        dosage_row['allele0'] = alleles[0]
        dosage_row['allele1'] = alleles[1]

        e = allele_expectation(self.bgen_obj["genotype"][row_number], nalleles=2, ploidy=2)
        dosage_row['dosages'] = e[..., -1].compute()  # count alt allele

        return dosage_row

    def _chunker(self, seq, size):
        """
        Divides a sequence in chunks according to the given size.
        :param seq:
        :param size:
        :return:
        """
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))

    def items(self, n_rows_cached=100, include_rsid=None):
        """
        Retrieve generator of variants, one by one. Although variants are returned in the order as they are stored in
        the BGEN file, when there are variants with the same positions their order is not guaranteed.
        :param n_rows_cached:
        :return:
        """

        variants_thin = self.bgen_obj['variants']
        if include_rsid is not None and len(include_rsid) > 0:
            cond = variants_thin['rsid'].isin(include_rsid)
            variants_thin = variants_thin[cond]

        row_numbers_chunks = self._chunker(variants_thin.index.tolist(), n_rows_cached)

        for chunk in row_numbers_chunks:
            chunk_variants = variants_thin.loc[chunk]
            alleles = chunk_variants['allele_ids'].str.split(',', n=1, expand=True)
            chunk_variants = chunk_variants.assign(allele0=alleles[0])
            chunk_variants = chunk_variants.assign(allele1=alleles[1])
            chunk_variants = chunk_variants.drop(columns=['allele_ids'])

            #chunk_expectations = allele_expectation(self.bgen_obj["genotype"][chunk], nalleles=2, ploidy=2)
            #check_dosages = chunk_expectations[..., -1].compute()
            check_dosages = np.dot(self.bgen_obj['genotype'][chunk].compute(), [0,1,2])

            for idx in range(len(chunk)):
                variant_info = chunk_variants.iloc[idx]
                dosage_row = variant_info.rename({'chrom': 'chr', 'pos': 'position'})
                dosage_row['chr'] = int(dosage_row.chr)

                dosage_row['dosages'] = check_dosages[idx]

                yield dosage_row
