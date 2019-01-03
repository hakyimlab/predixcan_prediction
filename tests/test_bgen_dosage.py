import unittest
from time import time

import numpy as np

from Software.bgen.bgen_dosage import BGENDosage
from tests.utils import get_repository_path, truncate


class BGENDosageTest(unittest.TestCase):
    def test_init(self):
        # Prepare
        # Run
        bgen_dosage = BGENDosage(get_repository_path('set00/chr1impv1.bgen'))

        # Validate
        assert bgen_dosage is not None

    def test_get_first_row(self):
        # Prepare
        bgen_dosage = BGENDosage(get_repository_path('set00/chr1impv1.bgen'))

        # Run
        dosage_row = bgen_dosage.get_row(0)

        assert dosage_row is not None

        assert hasattr(dosage_row, 'chr')
        assert dosage_row.chr == 1

        assert hasattr(dosage_row, 'rsid')
        assert dosage_row.rsid == 'rs1'

        assert hasattr(dosage_row, 'position')
        assert dosage_row.position == 100

        assert hasattr(dosage_row, 'allele0')
        assert dosage_row.allele0 == 'G'

        assert hasattr(dosage_row, 'allele1')
        assert dosage_row.allele1 == 'A'

        # assert hasattr(dosage_row, 'maf')
        # assert dosage_row.maf == 0.4894

        assert hasattr(dosage_row, 'dosages')
        assert dosage_row.dosages is not None
        assert hasattr(dosage_row.dosages, 'shape')
        assert len(dosage_row.dosages) == 300

        # NA
        assert round(dosage_row.dosages[0], 4) == round(np.dot([0.74909, 0.01333, 0.23758], [0, 1, 2]), 4) == 0.4885, dosage_row.dosages[0]
        # 1
        assert round(dosage_row.dosages[2], 4) == round(np.dot([0.05437, 0.91567, 0.02996], [0, 1, 2]), 4) == 0.9756, dosage_row.dosages[2]
        # 2
        assert round(dosage_row.dosages[3], 4) == round(np.dot([0.00650, 0.02577, 0.96773], [0, 1, 2]), 4) == 1.9612, dosage_row.dosages[3]
        # 0
        assert round(dosage_row.dosages[5], 4) == round(np.dot([0.95803, 0.03895, 0.00302], [0, 1, 2]), 4) == 0.0450, dosage_row.dosages[5]

    def test_get_second_row(self):
        # Prepare
        bgen_dosage = BGENDosage(get_repository_path('set00/chr1impv1.bgen'))

        # Run
        dosage_row = bgen_dosage.get_row(1)

        assert dosage_row is not None

        assert hasattr(dosage_row, 'chr')
        assert dosage_row.chr == 1

        assert hasattr(dosage_row, 'rsid')
        assert dosage_row.rsid == 'rs2'

        assert hasattr(dosage_row, 'position')
        assert dosage_row.position == 181

        assert hasattr(dosage_row, 'allele0')
        assert dosage_row.allele0 == 'G'

        assert hasattr(dosage_row, 'allele1')
        assert dosage_row.allele1 == 'C'

        # assert hasattr(dosage_row, 'maf')
        # assert dosage_row.maf == 0.4894

        assert hasattr(dosage_row, 'dosages')
        assert dosage_row.dosages is not None
        assert hasattr(dosage_row.dosages, 'shape')
        assert len(dosage_row.dosages) == 300

        assert round(dosage_row.dosages[0], 4) == round(np.dot([0.75232, 0.11725, 0.13043], [0, 1, 2]), 4) == 0.3781, dosage_row.dosages[0]
        assert round(dosage_row.dosages[299], 4) == round(np.dot([0.00937, 0.13421, 0.85642], [0, 1, 2]), 4) == 1.8470, dosage_row.dosages[299]

    def test_get_last_row(self):
        # Prepare
        bgen_dosage = BGENDosage(get_repository_path('set00/chr1impv1.bgen'))

        # Run
        dosage_row = bgen_dosage.get_row(-1)

        assert dosage_row is not None

        assert hasattr(dosage_row, 'chr')
        assert dosage_row.chr == 1

        assert hasattr(dosage_row, 'rsid')
        assert dosage_row.rsid == 'rs250'

        assert hasattr(dosage_row, 'position')
        assert dosage_row.position == 18389

        assert hasattr(dosage_row, 'allele0')
        assert dosage_row.allele0 == 'T'

        assert hasattr(dosage_row, 'allele1')
        assert dosage_row.allele1 == 'C'

        # assert hasattr(dosage_row, 'maf')
        # assert dosage_row.maf == 0.4722

        assert hasattr(dosage_row, 'dosages')
        assert dosage_row.dosages is not None
        assert hasattr(dosage_row.dosages, 'shape')
        assert len(dosage_row.dosages) == 300

        # 1
        assert round(dosage_row.dosages[0], 4) == round(np.dot([0.04713, 0.94817, 0.00470], [0, 1, 2]), 4) == 0.9576, dosage_row.dosages[0]
        # NA (plink)
        assert round(dosage_row.dosages[2], 4) == round(np.dot([0.07355, 0.50369, 0.42276], [0, 1, 2]), 4) == 1.3492, dosage_row.dosages[2]
        # 2
        assert round(dosage_row.dosages[8], 4) == round(np.dot([0.01488, 0.07935, 0.90576], [0, 1, 2]), 4) == 1.8909, dosage_row.dosages[8]
        # 0
        assert round(dosage_row.dosages[12], 4) == round(np.dot([0.95152, 0.02008, 0.02840], [0, 1, 2]), 4) == 0.0769, dosage_row.dosages[12]

    def test_get_last_row_other_chromosome(self):
        # Prepare
        bgen_dosage = BGENDosage(get_repository_path('set00/chr2impv1.bgen'))

        # Run
        dosage_row = bgen_dosage.get_row(-1)

        assert dosage_row is not None

        assert hasattr(dosage_row, 'chr')
        assert dosage_row.chr == 2

        assert hasattr(dosage_row, 'rsid')
        assert dosage_row.rsid == 'rs2000149'

        assert hasattr(dosage_row, 'position')
        assert dosage_row.position == 11226

        assert hasattr(dosage_row, 'allele0')
        assert dosage_row.allele0 == 'G'

        assert hasattr(dosage_row, 'allele1')
        assert dosage_row.allele1 == 'T'

        # assert hasattr(dosage_row, 'maf')
        # assert dosage_row.maf == 0.4722

        assert hasattr(dosage_row, 'dosages')
        assert dosage_row.dosages is not None
        assert hasattr(dosage_row.dosages, 'shape')
        assert len(dosage_row.dosages) == 300

        # 1
        assert round(dosage_row.dosages[0], 4) == round(np.dot([0.94620, 0.05350, 0.00030], [0, 1, 2]), 4) == 0.0541, dosage_row.dosages[0]
        assert round(dosage_row.dosages[1], 4) == round(np.dot([0.01373, 0.09532, 0.89094], [0, 1, 2]), 4) == 1.8772, dosage_row.dosages[1]
        assert round(dosage_row.dosages[299], 4) == round(np.dot([0.04675, 0.93974, 0.01351], [0, 1, 2]), 4) == 0.9668, dosage_row.dosages[299]

    def test_get_iterator(self):
        # Prepare
        bgen_dosage = BGENDosage(get_repository_path('set00/chr2impv1.bgen'))

        # Run
        all_items = list(bgen_dosage.items(n_rows_cached=10))
        assert len(all_items) == 150

        # snp 1
        assert all_items[0].chr == 2
        assert all_items[0].position == 100
        assert all_items[0].allele0 == 'A'
        assert all_items[0].allele1 == 'G'
        assert all_items[0].rsid == 'rs2000000'
        assert all_items[0].dosages.shape == (300,)
        assert truncate(all_items[0].dosages[0]) == truncate(np.dot([0.94401, 0.02976, 0.02623], [0, 1, 2])) == 0.0822
        assert truncate(all_items[0].dosages[2]) == truncate(np.dot([0.00658, 0.92760, 0.06582], [0, 1, 2])) == 1.0592

        # snp middle
        assert all_items[99].chr == 2
        assert all_items[99].position == 7516
        assert all_items[99].allele0 == 'T'
        assert all_items[99].allele1 == 'A'
        assert all_items[99].rsid == 'rs2000099'
        assert all_items[99].dosages.shape == (300,)
        assert truncate(all_items[99].dosages[0]) == truncate(np.dot([0.03148, 0.82993, 0.13854], [0, 1, 2])) == 1.1070
        assert truncate(all_items[99].dosages[5]) == truncate(np.dot([0.04327, 0.89103, 0.06570], [0, 1, 2])) == 1.0224

        # snp last
        assert all_items[149].chr == 2
        assert all_items[149].position == 11226
        assert all_items[149].allele0 == 'G'
        assert all_items[149].allele1 == 'T'
        assert all_items[149].rsid == 'rs2000149'
        assert all_items[149].dosages.shape == (300,)
        assert truncate(all_items[149].dosages[1]) == truncate(np.dot([0.01371, 0.09532, 0.89091], [0, 1, 2])) == 1.8771
        assert truncate(all_items[149].dosages[2]) == truncate(np.dot([0.07391, 0.09597, 0.83011], [0, 1, 2])) == 1.7561

    def test_get_iterator_repeated_variant_positions(self):
        # Prepare
        bgen_dosage = BGENDosage(get_repository_path('set06_repeated_positions/chr1impv1.bgen'))

        # Run
        all_items = list(bgen_dosage.items(n_rows_cached=5))
        assert len(all_items) == 11, len(all_items)

        # snp 1
        assert all_items[0].chr == 1
        assert all_items[0].position == 100
        assert all_items[0].allele0 == 'T'
        assert all_items[0].allele1 == 'A'
        assert all_items[0].rsid == 'rs1'
        assert all_items[0].dosages.shape == (20,)
        assert truncate(all_items[0].dosages[0]) == truncate(np.dot([0.06817, 0.27690, 0.65493], [0, 1, 2])) == 1.5867
        assert truncate(all_items[0].dosages[19]) == truncate(np.dot([0.00219, 0.08983, 0.90798], [0, 1, 2])) == 1.9057

        # snp 5
        assert all_items[4].chr == 1
        assert all_items[4].position == 418
        assert all_items[4].allele0 == 'T'
        assert all_items[4].allele1 == 'A'
        assert all_items[4].rsid == 'rs5'
        assert all_items[4].dosages.shape == (20,)
        assert truncate(all_items[4].dosages[0]) == truncate(np.dot([0.09158, 0.16910, 0.73933], [0, 1, 2])) == 1.6477
        assert truncate(all_items[4].dosages[1]) == truncate(np.dot([0.09820, 0.09934, 0.80246], [0, 1, 2])) == 1.7042
        assert truncate(all_items[4].dosages[19]) == truncate(np.dot([0.02833, 0.93189, 0.03978], [0, 1, 2])) == 1.0114

        # snp 6
        assert all_items[5].chr == 1
        assert all_items[5].position == 418
        assert all_items[5].allele0 == 'T'
        assert all_items[5].allele1 == 'C'
        assert all_items[5].rsid == 'rs5'
        assert all_items[5].dosages.shape == (20,)
        assert truncate(all_items[5].dosages[0]) == truncate(np.dot([0.00598, 0.02878, 0.96524], [0, 1, 2])) == 1.9592
        assert truncate(all_items[5].dosages[1]) == truncate(np.dot([0.01553, 0.14800, 0.83647], [0, 1, 2])) == 1.8209
        assert truncate(all_items[5].dosages[19]) == truncate(np.dot([0.08347, 0.02509, 0.89144], [0, 1, 2])) == 1.8079

        # snp last
        assert all_items[10].chr == 1
        assert all_items[10].position == 839
        assert all_items[10].allele0 == 'G'
        assert all_items[10].allele1 == 'A'
        assert all_items[10].rsid == 'rs10'
        assert all_items[10].dosages.shape == (20,)
        assert truncate(all_items[10].dosages[0]) == truncate(np.dot([0.03161, 0.82957, 0.13882], [0, 1, 2])) == 1.1072
        assert truncate(all_items[10].dosages[19]) == truncate(np.dot([0.96104, 0.03167, 0.00729], [0, 1, 2])) == 0.0462

    def test_get_iterator_with_n_cache_greater_than_n_variants(self):
        # Prepare
        bgen_dosage = BGENDosage(get_repository_path('set06_repeated_positions/chr2impv1.bgen'))

        # Run
        np.random.rand(0)
        all_items = list(bgen_dosage.items(n_rows_cached=15))
        assert len(all_items) == 13

        idx0 = 1
        idx1 = 0

        # snp 1
        assert all_items[idx0].chr == 2
        assert all_items[idx0].position == 100
        assert all_items[idx0].allele0 == 'C'
        assert all_items[idx0].allele1 == 'G'
        assert all_items[idx0].rsid == 'rs2000000'
        assert all_items[idx0].dosages.shape == (20,)
        assert truncate(all_items[idx0].dosages[0]) == truncate(np.dot([0.86648, 0.00133, 0.13219], [0, 1, 2])) == 0.2657
        assert truncate(all_items[idx0].dosages[19]) == truncate(np.dot([0.16810, 0.60427, 0.22762], [0, 1, 2])) == 1.0595

        # snp 2
        assert all_items[idx1].chr == 2
        assert all_items[idx1].position == 100
        assert all_items[idx1].allele0 == 'A'
        assert all_items[idx1].allele1 == 'G'
        assert all_items[idx1].rsid == 'rs2000000'
        assert all_items[idx1].dosages.shape == (20,)
        assert truncate(all_items[idx1].dosages[0]) == truncate(np.dot([0.02759, 0.17211, 0.80030], [0, 1, 2])) == 1.7727
        assert truncate(all_items[idx1].dosages[1]) == truncate(np.dot([0.63037, 0.32488, 0.04474], [0, 1, 2])) == 0.4143
        assert truncate(all_items[idx1].dosages[19]) == truncate(np.dot([0.16943, 0.05240, 0.77817], [0, 1, 2])) == 1.6087

        # snp 3
        assert all_items[2].chr == 2
        assert all_items[2].position == 168
        assert all_items[2].allele0 == 'G'
        assert all_items[2].allele1 == 'C'
        assert all_items[2].rsid == 'rs2000001'
        assert all_items[2].dosages.shape == (20,)
        assert truncate(all_items[2].dosages[0]) == truncate(np.dot([0.05452, 0.10875, 0.83674], [0, 1, 2])) == 1.7822
        assert truncate(all_items[2].dosages[1]) == truncate(np.dot([0.83410, 0.14198, 0.02392], [0, 1, 2])) == 0.1898
        assert truncate(all_items[2].dosages[19]) == truncate(np.dot([0.76940, 0.16924, 0.06136], [0, 1, 2])) == 0.2919

        # snp last
        assert all_items[12].chr == 2
        assert all_items[12].position == 934
        assert all_items[12].allele0 == 'T'
        assert all_items[12].allele1 == 'A'
        assert all_items[12].rsid == 'rs2000011'
        assert all_items[12].dosages.shape == (20,)
        assert truncate(all_items[12].dosages[0]) == truncate(np.dot([0.00457, 0.01764, 0.97778], [0, 1, 2])) == 1.9732
        assert truncate(all_items[12].dosages[19]) == truncate(np.dot([0.11567, 0.05896, 0.82537], [0, 1, 2])) == 1.7097

    def test_performance_get_iterator_with_cache(self):
        # measure time with no cache
        bgen_dosage = BGENDosage(get_repository_path('set00/chr2impv1.bgen'))

        start_time = time()
        no_cache_results = list(bgen_dosage.items(n_rows_cached=1))
        no_cache_time = time() - start_time

        # measure time with cache
        bgen_dosage = BGENDosage(get_repository_path('set00/chr2impv1.bgen'))

        start_time = time()
        cache_results = list(bgen_dosage.items(n_rows_cached=200))
        cache_time = time() - start_time

        assert len(no_cache_results) == len(cache_results)
        assert cache_time * 3.0 <= no_cache_time, (cache_time, no_cache_time)
