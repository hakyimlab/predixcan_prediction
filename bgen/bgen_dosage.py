import sqlite3
import gc

import numpy as np
import pandas as pd

from rpy2.robjects.vectors import StrVector
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
pandas2ri.activate()


class BGENDosage:
    def __init__(self, bgen_path, bgen_bgi=None, sample_path=None):
        self.bgen_path = bgen_path
        if bgen_bgi is None:
            self.bgi_path = self.bgen_path + '.bgi'
        else:
            self.bgi_path = bgen_bgi + '.bgi'
        self.sample_path = sample_path
        
        self.rbgen = importr('rbgen')
        with sqlite3.connect(self.bgi_path) as conn:
            self.variants_count = conn.execute('select count(*) from Variant').fetchone()[0]

            # FIXME: only one chromosome per BGEN file is supported
            self.chr_number = conn.execute('select distinct chromosome from Variant').fetchone()[0]

    def get_row(self, row_idx):
        row_number = (row_idx if row_idx >= 0 else self.variants_count + row_idx, )

        with sqlite3.connect(self.bgi_path) as conn:
            variant_position = conn.execute('select position from Variant order by position limit 1 offset ?', row_number).fetchone()[0]

        ranges = pd.DataFrame({
            'chromosome': [self.chr_number],
            'start': [variant_position],
            'end': [variant_position],
        })

        data = self.rbgen.bgen_load(self.bgen_path, ranges)
        variant = pandas2ri.ri2py(data[0])
        probs = pandas2ri.ri2py(data[4])

        dosage_row = variant.iloc[0].rename({'chromosome': 'chr'})
        dosage_row['chr'] = int(dosage_row.chr)
        dosage_row['dosages'] = np.dot(probs[0, :, :], [0, 1, 2])

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
        # retrieve positions
        if include_rsid is not None:
            stm = 'select distinct rsid, position from Variant where rsid in ({}) order by file_start_position asc'.format(', '.join(["'{}'".format(x) for x in include_rsid]))
        else:
            stm = 'select distinct rsid, position from Variant order by file_start_position asc'

        with sqlite3.connect(self.bgi_path) as conn:
            cur = conn.cursor()
            cur.execute(stm)

            iteration = 0

            while True:
                if iteration > 0:
                    cached_data_struct = cached_data.__sexp__
                    del cached_data
                    del cached_data_struct
                    gc.collect()

                positions = cur.fetchmany(size=n_rows_cached)
                if not positions:
                    break

                rsids = [x[0] for x in positions]
                positions = [x[1] for x in positions]

                if include_rsid is None:
                    ranges = pd.DataFrame({
                        'chromosome': [self.chr_number],
                        'start': [positions[0]],
                        'end': [positions[-1]],
                    })

                    # rbgen = importr('rbgen')
                    cached_data = self.rbgen.bgen_load(self.bgen_path, ranges, index_filename = self.bgi_path)

                else:
                    cached_data = self.rbgen.bgen_load(self.bgen_path, rsids=StrVector(rsids), index_filename = self.bgi_path)

                all_variants = pandas2ri.ri2py(cached_data[0])
                all_probs = pandas2ri.ri2py(cached_data[4])

                iteration += 1

                for row_idx, (rsid, row) in enumerate(all_variants.iterrows()):
                    dosage_row = row.rename({'chromosome': 'chr'})
                    dosage_row['chr'] = int(dosage_row.chr)
                    dosage_row['dosages'] = np.dot(all_probs[row_idx, :, :], [0, 1, 2])

                    yield dosage_row
