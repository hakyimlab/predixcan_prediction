import os
import unittest
from subprocess import call, check_output
import tempfile
import sqlite3

import h5py
import numpy as np

from tests.utils import get_full_path, truncate, get_out, get_repository_path


DOSAGES = 'dosages'


def _count_datasets(name, data):
    global entries
    if isinstance(data, h5py.Dataset):
        entries += 1
    assert data is not None


def _create_model(model_name, values):
    model_path = get_full_path('tests/data/models/{}.db'.format(model_name))
    if os.path.exists(model_path): os.remove(model_path)

    with sqlite3.connect(model_path) as conn:
        conn.execute("""
            CREATE TABLE weights
            ( "rsid" TEXT,
                "gene" TEXT,
                "weight" REAL,
                "ref_allele" TEXT,
                "eff_allele" TEXT
            );
        """)

        for val in values:
            conn.execute("""
                insert into weights (rsid, gene, weight, ref_allele, eff_allele) values
                ('{}', '{}', {}, '{}', '{}');
            """.format(val[0], val[1], val[2], val[3], val[4]))

    return model_path


class PrediXcanTests(unittest.TestCase):
    def setUp(self):
        self.predixcan_path = get_full_path(os.path.join('Software', 'PrediXcan.py'))

    def test_alleles_in_bgen_order_gene(self):
        # Prepare
        tmpdir = tempfile.mkdtemp()
        output_file = os.path.join(tmpdir, 'output.hdf5')

        model_name = 'model00'
        model_path = _create_model(model_name, [
            ['rs1', 'gene00', 0.3712, 'G', 'A']
        ])

        options = [
            'python',
            self.predixcan_path,
            '--predict',
            '--hdf5s', get_full_path('tests/data/dosages/'),
            '--hdf5s-prefix', 'dosages_chr1_',
            '--weights', model_path,
            '--pred_exp', output_file,
        ]

        return_code = call(options)
        assert return_code == 0

        assert os.path.isfile(output_file)
        with h5py.File(output_file, 'r') as hdf5_file:
            assert len(hdf5_file.keys()) == 3
            assert 'pred_expr' in hdf5_file.keys()

            preds = hdf5_file['pred_expr']
            assert preds.shape == (1, 300)
            assert preds.dtype == np.dtype('float32')
            assert preds.scaleoffset == 4
            assert preds.chunks == (1, 300)

            assert truncate(preds[0, 0]) == truncate(0.3712 * (np.dot([0.74909, 0.01333, 0.23758], [0, 1, 2]))), preds[0, 0]

    def test_alleles_not_in_bgen_order_gene(self):
        # Prepare
        tmpdir = tempfile.mkdtemp()
        output_file = os.path.join(tmpdir, 'output.hdf5')

        model_name = 'model00'
        model_path = _create_model(model_name, [
            ['rs1', 'gene00', 0.3712, 'A', 'G']
        ])

        options = [
            'python',
            self.predixcan_path,
            '--predict',
            '--hdf5s', get_full_path('tests/data/dosages/'),
            '--hdf5s-prefix', 'dosages_chr1_',
            '--weights', model_path,
            '--pred_exp', output_file,
        ]

        return_code = call(options)
        assert return_code == 0

        assert os.path.isfile(output_file)
        with h5py.File(output_file, 'r') as hdf5_file:
            assert len(hdf5_file.keys()) == 3
            assert 'pred_expr' in hdf5_file.keys()

            preds = hdf5_file['pred_expr']
            assert preds.shape == (1, 300)
            assert preds.chunks == (1, 300)

            assert truncate(preds[0, 0]) == truncate(0.3712 * (2 - np.dot([0.74909, 0.01333, 0.23758], [0, 1, 2]))), preds[0, 0]

    def test_two_rsids_for_one_gene(self):
        # Prepare
        tmpdir = tempfile.mkdtemp()
        output_file = os.path.join(tmpdir, 'output.hdf5')

        model_name = 'model00'
        model_path = _create_model(model_name, [
            ['rs1', 'gene00', 0.3712, 'A', 'G'], # ambiguous
            ['rs2', 'gene00', 0.0807, 'G', 'C'], # non-ambiguous
        ])

        options = [
            'python',
            self.predixcan_path,
            '--predict',
            '--hdf5s', get_full_path('tests/data/dosages/'),
            '--hdf5s-prefix', 'dosages_chr1_',
            '--weights', model_path,
            '--pred_exp', output_file,
        ]

        return_code = call(options)
        assert return_code == 0

        assert os.path.isfile(output_file)
        with h5py.File(output_file, 'r') as hdf5_file:
            assert len(hdf5_file.keys()) == 3

            assert 'genes' in hdf5_file.keys()
            genes = hdf5_file['genes']
            assert genes.shape == (1,)
            assert genes[0] == 'gene00'

            assert 'pred_expr' in hdf5_file.keys()
            preds = hdf5_file['pred_expr']
            assert preds.shape == (1, 300)
            assert preds.chunks == (1, 300)

            assert truncate(preds[0, 0]) == truncate(
                0.3712 * (2 - np.dot([0.74909, 0.01333, 0.23758], [0, 1, 2])) +
                0.0807 * (np.dot([0.75232, 0.11729, 0.13050], [0, 1, 2]))
            ), preds[0, 0]

            assert truncate(preds[0, 299]) == truncate(
                0.3712 * (2 - np.dot([0.05763, 0.77328, 0.16910], [0, 1, 2])) +
                0.0807 * (np.dot([0.00937, 0.13421, 0.85658], [0, 1, 2]))
            ), preds[0, 299]

    def test_two_rsids_for_one_gene_one_rsid_for_another_gene(self):
        # Prepare
        tmpdir = tempfile.mkdtemp()
        output_file = os.path.join(tmpdir, 'output.hdf5')

        model_name = 'model00'
        model_path = _create_model(model_name, [
            ['rs1', 'gene00', 0.3712, 'A', 'G'], # ambiguous
            ['rs2', 'gene00', 0.0807, 'G', 'C'], # non-ambiguous
            ['rs10', 'gene01', 0.6188, 'A', 'T'], # non-ambiguous
        ])

        options = [
            'python',
            self.predixcan_path,
            '--predict',
            '--hdf5s', get_full_path('tests/data/dosages/'),
            '--hdf5s-prefix', 'dosages_chr1_',
            '--weights', model_path,
            '--pred_exp', output_file,
        ]

        return_code = call(options)
        assert return_code == 0

        assert os.path.isfile(output_file)
        with h5py.File(output_file, 'r') as hdf5_file:
            assert len(hdf5_file.keys()) == 3

            assert 'samples' in hdf5_file.keys()
            samples = hdf5_file['samples']
            assert samples.shape == (300,)
            assert all(samples[:].astype(str) == np.array([str(x) for x in range(1, 300 + 1)]))

            assert 'genes' in hdf5_file.keys()
            genes = hdf5_file['genes']
            assert genes.shape == (2,)
            assert genes[0] == 'gene00'
            assert genes[1] == 'gene01'

            assert 'pred_expr' in hdf5_file.keys()
            preds = hdf5_file['pred_expr']
            assert preds.shape == (2, 300)
            assert preds.chunks == (2, 300)

            # gene00
            assert truncate(preds[0, 0]) == truncate(
                0.3712 * (2 - np.dot([0.74909, 0.01333, 0.23758], [0, 1, 2])) +
                0.0807 * (np.dot([0.75232, 0.11729, 0.13040], [0, 1, 2]))
            ) == 0.5915, preds[0, 0]

            assert truncate(preds[0, 299]) == truncate(
                0.3712 * (2 - np.dot([0.05763, 0.77328, 0.16910], [0, 1, 2])) +
                0.0807 * (np.dot([0.00937, 0.13421, 0.85650], [0, 1, 2]))
            ) == 0.4788, preds[0, 299]

            # gene01
            assert truncate(preds[1, 0]) == truncate(
                0.6188 * (np.dot([0.11764, 0.86431, 0.01805], [0, 1, 2]))
            ) == 0.5571, preds[1, 0]

            assert truncate(preds[1, 298]) == truncate(
                0.6188 * (np.dot([0.03509, 0.82783, 0.13705], [0, 1, 2]))
            ) == 0.6818, preds[1, 298]

    def test_chunks(self):
        # Prepare
        tmpdir = tempfile.mkdtemp()
        output_file = os.path.join(tmpdir, 'output.hdf5')

        model_name = 'model00'
        alleles = {
            0: ('A', 'G'),
            1: ('C', 'A'),
            9: ('C', 'T'),
            10: ('G', 'T'),
            11: ('C', 'G'),
            21: ('G', 'A'),
        }

        weights = {
            0: 0.1158,
            1: 0.5455,
            9: 0.9876,
            10: 0.1755,
            11: 0.2754,
            21: 0.6855,
        }

        model_path = _create_model(model_name,
            [['rs{}'.format(i+1), 'gene{:0>2d}'.format(j), weights.get(j, 0.5), alleles.get(j, ('G',))[0], alleles.get(j, ('','C'))[1]] for j in range(21 + 1) for i in range(j * 10, j * 10 + 2)],
        )

        options = [
            'python',
            self.predixcan_path,
            '--predict',
            '--hdf5s', get_full_path('tests/data/dosages/'),
            '--hdf5s-prefix', 'dosages_chr1_',
            '--weights', model_path,
            '--pred_exp', output_file,
        ]

        return_code = call(options)
        assert return_code == 0

        assert os.path.isfile(output_file)
        with h5py.File(output_file, 'r') as hdf5_file:
            assert len(hdf5_file.keys()) == 3

            assert 'samples' in hdf5_file.keys()
            samples = hdf5_file['samples']
            assert samples.shape == (300,)
            assert all(samples[:].astype(str) == np.array([str(x) for x in range(1, 300 + 1)]))

            assert 'genes' in hdf5_file.keys()
            genes = hdf5_file['genes']
            assert genes.shape == (22,)
            assert genes[0] == 'gene00'
            assert genes[1] == 'gene01'
            assert genes[20] == 'gene20'
            assert genes[21] == 'gene21'

            assert 'pred_expr' in hdf5_file.keys()
            preds = hdf5_file['pred_expr']
            assert preds.shape == (22, 300)
            assert preds.chunks == (10, 300)

            # gene00
            assert truncate(preds[0, 0]) == truncate(
                0.1158 * (2 - np.dot([0.74909, 0.01333, 0.23740], [0, 1, 2])) +
                0.1158 * (np.dot([0.75232, 0.11725, 0.13030], [0, 1, 2]))
            ) == 0.2188, preds[0, 0]

            assert truncate(preds[0, 299]) == truncate(
                0.1158 * (2 - np.dot([0.05763, 0.77328, 0.16910], [0, 1, 2])) +
                0.1158 * (np.dot([0.00937, 0.13421, 0.85640], [0, 1, 2]))
            ) == 0.3167, preds[0, 299]

            # gene01
            assert truncate(preds[1, 0]) == truncate(
                0.5455 * (2 - np.dot([0.96807, 0.01962, 0.01231], [0, 1, 2])) +
                0.5455 * (np.dot([0.00190, 0.00429, 0.99381], [0, 1, 2]))
            ) == 2.1534, preds[1, 0]

            assert truncate(preds[1, 1]) == truncate(
                0.5455 * (2 - np.dot([0.91510, 0.06826, 0.01669], [0, 1, 2])) +
                0.5455 * (np.dot([0.70937, 0.04886, 0.24177], [0, 1, 2]))
            ) == 1.3259, preds[1, 1]

            # gene09
            assert truncate(preds[9, 0]) == truncate(
                0.9876 * (2 - np.dot([0.74754, 0.13307, 0.11935], [0, 1, 2])) +
                0.9876 * (2 - np.dot([0.03755, 0.78400, 0.17849], [0, 1, 2]))
            ) == 2.4564, preds[9, 0]

            assert truncate(preds[9, 298]) == truncate(
                0.9876 * (2 - np.dot([0.71102, 0.00968, 0.27929], [0, 1, 2])) +
                0.9876 * (2 - np.dot([0.08631, 0.77285, 0.14089], [0, 1, 2]))
            ) == 2.3476, preds[9, 298]

            # gene10
            assert truncate(preds[10, 0]) == truncate(
                0.1755 * (2 - np.dot([0.05931, 0.08242, 0.85827], [0, 1, 2])) +
                0.1755 * (np.dot([0.83525, 0.01184, 0.15291], [0, 1, 2]))
            ) == 0.091, preds[10, 0]

            assert truncate(preds[10, 2]) == truncate(
                0.1755 * (2 - np.dot([0.61247, 0.22145, 0.16605], [0, 1, 2])) +
                0.1755 * (np.dot([0.09727, 0.77103, 0.13170], [0, 1, 2]))
            ) == 0.4353, preds[10, 2]

            # gene11
            assert truncate(preds[11, 0]) == truncate(
                0.2754 * (np.dot([0.83211, 0.12816, 0.03970], [0, 1, 2])) +
                0.2754 * (np.dot([0.96441, 0.02567, 0.00990], [0, 1, 2]))
            ) == 0.0696, preds[11, 0]

            assert truncate(preds[11, 299]) == truncate(
                0.2754 * (np.dot([0.04018, 0.84357, 0.11625], [0, 1, 2])) +
                0.2754 * (np.dot([0.07541, 0.11284, 0.81175], [0, 1, 2]))
            ) == 0.7745, preds[11, 299]

            # gene21
            assert truncate(preds[21, 0]) == truncate(
                0.6855 * (np.dot([0.73030, 0.13711, 0.13259], [0, 1, 2])) +
                0.6855 * (2 - np.dot([0.11456, 0.04225, 0.84315], [0, 1, 2]))
            ) == 0.4618, preds[21, 0]

            assert truncate(preds[21, 299]) == truncate(
                0.6855 * (np.dot([0.13023, 0.18599, 0.68379], [0, 1, 2])) +
                0.6855 * (2 - np.dot([0.85909, 0.07115, 0.06976], [0, 1, 2]))
            ) == 2.2915, preds[21, 299]

    def test_many_dosages_files(self):
        # Prepare
        tmpdir = tempfile.mkdtemp()
        output_file = os.path.join(tmpdir, 'output.hdf5')

        model_name = 'model00'
        alleles = {
            0: ('A', 'G'),
            1: ('C', 'A'),
            9: ('C', 'T'),
            10: ('G', 'T'),
            11: ('C', 'G'),
            21: ('G', 'A'),
        }

        weights = {
            0: 0.1158,
            1: 0.5455,
            9: 0.9876,
            10: 0.1755,
            11: 0.2754,
            21: 0.6855,
        }

        model_path = _create_model(model_name,
            [['rs{}'.format(i+1), 'gene{:0>3d}'.format(j), weights.get(j, 0.5), alleles.get(j, ('G',))[0], alleles.get(j, ('','C'))[1]] for j in range(21 + 1) for i in range(j * 10, j * 10 + 2)] +
            [['rs{}'.format(i+1), 'gene2{:0>2d}'.format(j), weights.get(j, 0.5), alleles.get(j, ('G',))[0], alleles.get(j, ('','C'))[1]] for j in range(11 + 1) for i in range(2000000 + j * 10, 2000000 + j * 10 + 2)],
        )

        options = [
            'python',
            self.predixcan_path,
            '--predict',
            '--hdf5s', get_full_path('tests/data/dosages/'),
            '--hdf5s-prefix', 'dosages_chr',
            '--weights', model_path,
            '--pred_exp', output_file,
        ]

        return_code = call(options)
        assert return_code == 0

        assert os.path.isfile(output_file)
        with h5py.File(output_file, 'r') as hdf5_file:
            assert len(hdf5_file.keys()) == 3

            assert 'samples' in hdf5_file.keys()
            samples = hdf5_file['samples']
            assert samples.shape == (300,)
            assert all(samples[:].astype(str) == np.array([str(x) for x in range(1, 300 + 1)]))

            assert 'genes' in hdf5_file.keys()
            genes = hdf5_file['genes']
            assert genes.shape == (22 + 12,)
            assert genes[0] == 'gene000'
            assert genes[1] == 'gene001'
            assert genes[20] == 'gene020'
            assert genes[21] == 'gene021'
            assert genes[22] == 'gene200'
            assert genes[23] == 'gene201'
            assert genes[32] == 'gene210'
            assert genes[-1] == 'gene211'

            assert 'pred_expr' in hdf5_file.keys()
            preds = hdf5_file['pred_expr']
            assert preds.shape == (22 + 12, 300)
            assert preds.chunks == (10, 300)

            # genes from chr1
            # gene00
            assert truncate(preds[0, 0]) == truncate(
                0.1158 * (2 - np.dot([0.74909, 0.01333, 0.23740], [0, 1, 2])) +
                0.1158 * (np.dot([0.75232, 0.11725, 0.13030], [0, 1, 2]))
            ) == 0.2188, preds[0, 0]

            assert truncate(preds[0, 299]) == truncate(
                0.1158 * (2 - np.dot([0.05763, 0.77328, 0.16910], [0, 1, 2])) +
                0.1158 * (np.dot([0.00937, 0.13421, 0.85640], [0, 1, 2]))
            ) == 0.3167, preds[0, 299]

            # gene01
            assert truncate(preds[1, 0]) == truncate(
                0.5455 * (2 - np.dot([0.96807, 0.01962, 0.01231], [0, 1, 2])) +
                0.5455 * (np.dot([0.00190, 0.00429, 0.99381], [0, 1, 2]))
            ) == 2.1534, preds[1, 0]

            assert truncate(preds[1, 1]) == truncate(
                0.5455 * (2 - np.dot([0.91510, 0.06826, 0.01669], [0, 1, 2])) +
                0.5455 * (np.dot([0.70937, 0.04886, 0.24177], [0, 1, 2]))
            ) == 1.3259, preds[1, 1]

            # gene09
            assert truncate(preds[9, 0]) == truncate(
                0.9876 * (2 - np.dot([0.74754, 0.13307, 0.11935], [0, 1, 2])) +
                0.9876 * (2 - np.dot([0.03755, 0.78400, 0.17849], [0, 1, 2]))
            ) == 2.4564, preds[9, 0]

            assert truncate(preds[9, 298]) == truncate(
                0.9876 * (2 - np.dot([0.71102, 0.00968, 0.27929], [0, 1, 2])) +
                0.9876 * (2 - np.dot([0.08631, 0.77285, 0.14089], [0, 1, 2]))
            ) == 2.3476, preds[9, 298]

            # gene10
            assert truncate(preds[10, 0]) == truncate(
                0.1755 * (2 - np.dot([0.05931, 0.08242, 0.85827], [0, 1, 2])) +
                0.1755 * (np.dot([0.83525, 0.01184, 0.15291], [0, 1, 2]))
            ) == 0.091, preds[10, 0]

            assert truncate(preds[10, 2]) == truncate(
                0.1755 * (2 - np.dot([0.61247, 0.22145, 0.16605], [0, 1, 2])) +
                0.1755 * (np.dot([0.09727, 0.77103, 0.13170], [0, 1, 2]))
            ) == 0.4353, preds[10, 2]

            # gene11
            assert truncate(preds[11, 0]) == truncate(
                0.2754 * (np.dot([0.83211, 0.12816, 0.03970], [0, 1, 2])) +
                0.2754 * (np.dot([0.96441, 0.02567, 0.00990], [0, 1, 2]))
            ) == 0.0696, preds[11, 0]

            assert truncate(preds[11, 299]) == truncate(
                0.2754 * (np.dot([0.04018, 0.84357, 0.11625], [0, 1, 2])) +
                0.2754 * (np.dot([0.07541, 0.11284, 0.81175], [0, 1, 2]))
            ) == 0.7745, preds[11, 299]

            # gene21
            assert truncate(preds[21, 0]) == truncate(
                0.6855 * (np.dot([0.73030, 0.13711, 0.13255], [0, 1, 2])) +
                0.6855 * (2 - np.dot([0.11456, 0.04225, 0.84315], [0, 1, 2]))
            ) == 0.4617, preds[21, 0]

            assert truncate(preds[21, 299]) == truncate(
                0.6855 * (np.dot([0.13023, 0.18599, 0.68379], [0, 1, 2])) +
                0.6855 * (2 - np.dot([0.85909, 0.07115, 0.06976], [0, 1, 2]))
            ) == 2.2915, preds[21, 299]

            # genes from chr 2
            # gene200
            assert truncate(preds[22, 0]) == truncate(
                0.1158 * (np.dot([0.96459, 0.02124, 0.01418], [0, 1, 2])) +
                0.1158 * (2 - np.dot([0.91804, 0.01235, 0.06966], [0, 1, 2]))
            ) == 0.2197, preds[22, 0]

            assert truncate(preds[22, 299]) == truncate(
                0.1158 * (np.dot([0.15472, 0.80145, 0.04384], [0, 1, 2])) +
                0.1158 * (2 - np.dot([0.95387, 0.00694, 0.03919], [0, 1, 2]))
            ) == 0.3246, preds[22, 299]

            # gene211
            assert truncate(preds[33, 0]) == truncate(
                0.2754 * (np.dot([0.07602, 0.84035, 0.08360], [0, 1, 2])) +
                0.2754 * (np.dot([0.79911, 0.18546, 0.01550], [0, 1, 2]))
            ) == 0.337, preds[33, 0]

            assert truncate(preds[33, 299]) == truncate(
                0.2754 * (np.dot([0.18570, 0.81029, 0.00390], [0, 1, 2])) +
                0.2754 * (np.dot([0.21121, 0.01581, 0.77310], [0, 1, 2]))
            ) == 0.6554, preds[33, 299]
