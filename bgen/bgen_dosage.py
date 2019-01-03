import numpy as np
import pandas as pd
import sqlite3

from bgen_reader import read_bgen, allele_expectation, compute_dosage


class BGENDosage:
    def __init__(self, bgen_path, sample_path=None, cache_size=50):
        self.bgen_path = bgen_path
        self.bgi_path = self.bgen_path + '.bgi'
        self.sample_path = sample_path
        self.cache_size = cache_size

        self.bgen_obj = read_bgen(self.bgen_path, sample_file=self.sample_path, size=self.cache_size, verbose=False)

        with sqlite3.connect(self.bgi_path) as conn:
            self.variants_count = conn.execute('select count(*) from Variant').fetchone()[0]

            # FIXME: only one chromosome per BGEN file is supported
            self.chr_number = conn.execute('select distinct chromosome from Variant').fetchone()[0]

    def get_row(self, row_idx):
        row_number = (row_idx if row_idx >= 0 else self.variants_count + row_idx,)

        # with sqlite3.connect(self.bgi_path) as conn:
        #     variant_position = conn.execute('select position from Variant order by position limit 1 offset ?', row_number).fetchone()[0]
        #
        # ranges = pd.DataFrame({
        #     'chromosome': [self.chr_number],
        #     'start': [variant_position],
        #     'end': [variant_position],
        # })

        # data = self.rbgen.bgen_load(self.bgen_path, ranges)
        # variant = pandas2ri.ri2py(data[0])
        # probs = pandas2ri.ri2py(data[4])

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

    def items(self, n_rows_cached=100):
        """
        Retrieve generator of variants, one by one. Although variants are returned in the order as they are stored in
        the BGEN file, when there are variants with the same positions their order is not guaranteed.
        :param n_rows_cached:
        :return:
        """

        row_numbers_chunks = self._chunker(list(range(self.bgen_obj['variants'].shape[0])), n_rows_cached)

        for chunk in row_numbers_chunks:
            chunk_variants = self.bgen_obj['variants'].iloc[chunk]
            alleles = chunk_variants['allele_ids'].str.split(',', n=1, expand=True)
            chunk_variants = chunk_variants.assign(allele0=alleles[0])
            chunk_variants = chunk_variants.assign(allele1=alleles[1])
            chunk_variants = chunk_variants.drop(columns=['allele_ids'])

            chunk_expectations = allele_expectation(self.bgen_obj["genotype"][chunk], nalleles=2, ploidy=2)
            check_dosages = chunk_expectations[..., -1].compute()

            for idx in range(len(chunk)):
                variant_info = chunk_variants.iloc[idx]
                dosage_row = variant_info.rename({'chrom': 'chr', 'pos': 'position'})
                dosage_row['chr'] = int(dosage_row.chr)

                # alleles = dosage_row.allele_ids.split(',')
                # assert len(alleles) == 2, len(alleles)
                # dosage_row['allele0'] = alleles[0]
                # dosage_row['allele1'] = alleles[1]

                # e = allele_expectation(self.bgen_obj["genotype"][row_number], nalleles=2, ploidy=2)
                dosage_row['dosages'] = check_dosages[idx]

                yield dosage_row

        # # retrieve positions
        # stm = 'select distinct position from Variant order by file_start_position asc'
        #
        # with sqlite3.connect(self.bgi_path) as conn:
        #     cur = conn.cursor()
        #     cur.execute(stm)
        #
        #     iteration = 1
        #
        #     while True:
        #         positions = cur.fetchmany(size=n_rows_cached)
        #         if not positions:
        #             break
        #
        #         positions = [x[0] for x in positions]
        #
        #         ranges = pd.DataFrame({
        #             'chromosome': [self.chr_number],
        #             'start': [positions[0]],
        #             'end': [positions[-1]],
        #         })
        #
        #         # rbgen = importr('rbgen')
        #         cached_data = self.rbgen.bgen_load(self.bgen_path, ranges)
        #         all_variants = pandas2ri.ri2py(cached_data[0])
        #         all_probs = pandas2ri.ri2py(cached_data[4])
        #
        #         for row_idx, (rsid, row) in enumerate(all_variants.iterrows()):
        #             dosage_row = row.rename({'chromosome': 'chr'})
        #             dosage_row['chr'] = int(dosage_row.chr)
        #             dosage_row['dosages'] = np.dot(all_probs[row_idx, :, :], [0, 1, 2])
        #
        #             yield dosage_row
        #
        #         cached_data_struct = cached_data.__sexp__
        #         del(cached_data)
        #         del(cached_data_struct)
        #
        #         if iteration % 100:
        #             gc.collect()
        #
        #         iteration += 1
