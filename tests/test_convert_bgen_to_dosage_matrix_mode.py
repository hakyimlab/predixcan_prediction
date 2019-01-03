import os
import unittest
from subprocess import call, check_output
import tempfile

import numpy as np
import h5py

from tests.utils import get_full_path, truncate, get_out, get_repository_path


DOSAGES = 'dosages'


def _count_datasets(name, data):
    global entries
    if isinstance(data, h5py.Dataset):
        entries += 1
    assert data is not None


class ConvertBGENToDosageMatrixMode(unittest.TestCase):
    def setUp(self):
        self.convert_path = get_full_path(os.path.join('Software', 'convert_bgen_to_dosage.py'))

    def test_mandatory_command_arguments(self):
        tmpdir = tempfile.mkdtemp()
        output_file = os.path.join(tmpdir, 'output.hdf5')

        options = [
            'python',
            self.convert_path,
            get_full_path('tests/data/set00'),
            output_file,
        ]

        return_code = call(options)
        assert return_code == 0

    def test_shapes_dtype_and_chunks_of_mandatory_datasets(self):
        tmpdir = tempfile.mkdtemp()
        output_file = os.path.join(tmpdir, 'output.hdf5')

        options = [
            'python',
            self.convert_path,
            get_full_path('tests/data/set00'),
            output_file,
        ]

        return_code = call(options)
        assert return_code == 0

        assert os.path.isfile(output_file)
        with h5py.File(output_file, 'r') as hdf5_file:
            # check dosages and its shape
            assert DOSAGES in hdf5_file
            dosages = hdf5_file[DOSAGES]
            assert dosages.shape == (250 + 150, 300)
            assert dosages.dtype == np.dtype('float32')
            assert dosages.chunks is not None

            # check amount of 'variants/chr' entries
            global entries
            entries = 0

            chrs = hdf5_file['variants/chr']
            rsids = hdf5_file['variants/rsids']
            positions = hdf5_file['variants/position']
            allele0s = hdf5_file['variants/allele0']
            allele1s = hdf5_file['variants/allele1']

            assert chrs.shape == (250 + 150,)
            assert chrs.dtype == np.dtype('uint8')
            assert chrs.chunks is None

            assert rsids.shape == (250 + 150,)
            assert rsids.dtype == np.dtype('S20')
            assert rsids.chunks is None

            assert positions.shape == (250 + 150,)
            assert positions.dtype == np.dtype('uint64')
            assert positions.chunks is None

            assert allele0s.shape == (250 + 150,)
            assert allele0s.dtype == np.dtype('S20')
            assert allele0s.chunks is None

            assert allele1s.shape == (250 + 150,)
            assert allele1s.dtype == np.dtype('S20')
            assert allele1s.chunks is None

    def test_set00_dosages_against_metadata(self):
        tmpdir = tempfile.mkdtemp()
        output_file = os.path.join(tmpdir, 'output.hdf5')

        options = [
            'python',
            self.convert_path,
            get_full_path('tests/data/set00'),
            output_file,
        ]

        return_code = call(options)
        assert return_code == 0

        assert os.path.isfile(output_file)
        with h5py.File(output_file, 'r') as hdf5_file:
            # check dosages and its shape
            assert DOSAGES in hdf5_file
            dosages = hdf5_file[DOSAGES]
            assert dosages.shape == (250 + 150, 300)

            chrs = hdf5_file['variants/chr']
            rsids = hdf5_file['variants/rsids']
            positions = hdf5_file['variants/position']
            allele0s = hdf5_file['variants/allele0']
            allele1s = hdf5_file['variants/allele1']

            # chr 1: check first snp
            assert chrs[0] == 1
            assert rsids[0] == 'rs1'
            assert positions[0] == 100
            assert allele0s[0] == 'G'
            assert allele1s[0] == 'A'
            assert truncate(dosages[0, 0], 4) == truncate(np.dot([0.74909, 0.01333, 0.23758], [0, 1, 2]), 4) == 0.4884
            assert truncate(dosages[0, 2], 4) == truncate(np.dot([0.05437, 0.91567, 0.02996], [0, 1, 2]), 4) == 0.9755
            assert truncate(dosages[0, 3], 4) == truncate(np.dot([0.00650, 0.02577, 0.96773], [0, 1, 2]), 4) == 1.9612
            assert truncate(dosages[0, 5], 4) == truncate(np.dot([0.95803, 0.03895, 0.00302], [0, 1, 2]), 4) == 0.0449

            # chr 1: check second snp
            assert chrs[1] == 1
            assert rsids[1] == 'rs2'
            assert positions[1] == 181
            assert allele0s[1] == 'G'
            assert allele1s[1] == 'C'
            assert truncate(dosages[1, 0], 4) == truncate(np.dot([0.75232, 0.1172, 0.13043], [0, 1, 2]), 4) == 0.3780
            assert truncate(dosages[1, 299], 4) == truncate(np.dot([0.00937, 0.13421, 0.85642], [0, 1, 2]), 4) == 1.8470

            # chr 1: check last snp
            assert chrs[249] == 1
            assert rsids[249] == 'rs250'
            assert positions[249] == 18389
            assert allele0s[249] == 'T'
            assert allele1s[249] == 'C'
            assert truncate(dosages[249, 0], 4) == truncate(np.dot([0.04713, 0.94817, 0.00470], [0, 1, 2]), 4) == 0.9575
            assert truncate(dosages[249, 2], 4) == truncate(np.dot([0.07355, 0.50369, 0.42276], [0, 1, 2]), 4) == 1.3492
            assert truncate(dosages[249, 8], 4) == truncate(np.dot([0.01488, 0.07935, 0.90576], [0, 1, 2]), 4) == 1.8908
            assert truncate(dosages[249, 12], 4) == truncate(np.dot([0.95152, 0.02008, 0.02840], [0, 1, 2]), 4) == 0.0768

            # chr 2: check first snp
            assert chrs[250 + 0] == 2
            assert rsids[250 + 0] == 'rs2000000'
            assert positions[250 + 0] == 100
            assert allele0s[250 + 0] == 'A'
            assert allele1s[250 + 0] == 'G'
            assert truncate(dosages[250 + 0, 0], 4) == truncate(np.dot([0.9440, 0.02977, 0.02623], [0, 1, 2]), 4) == 0.0822
            assert truncate(dosages[250 + 0, 1], 4) == truncate(np.dot([0.06851, 0.85242, 0.07907], [0, 1, 2]), 4) == 1.0105
            assert truncate(dosages[250 + 0, 299], 4) == truncate(np.dot([0.08272, 0.89635, 0.02093], [0, 1, 2]), 4) == 0.9382

            # chr 2: check last snp
            assert chrs[250 + 149] == 2
            assert rsids[250 + 149] == 'rs2000149'
            assert positions[250 + 149] == 11226
            assert allele0s[250 + 149] == 'G'
            assert allele1s[250 + 149] == 'T'
            assert truncate(dosages[250 + 149, 0], 4) == truncate(np.dot([0.9462, 0.0535, 0.0003], [0, 1, 2]), 4) == 0.0541
            assert truncate(dosages[250 + 149, 1], 4) == truncate(np.dot([0.01373, 0.09532, 0.8909], [0, 1, 2]), 4) == 1.8771
            assert truncate(dosages[250 + 149, 299], 4) == truncate(np.dot([0.04675, 0.93974, 0.01351], [0, 1, 2]), 4) == 0.9667

    def test_set01_dosages_against_metadata(self):
        tmpdir = tempfile.mkdtemp()
        output_file = os.path.join(tmpdir, 'output.hdf5')

        options = [
            'python',
            self.convert_path,
            get_full_path('tests/data/set01'),
            output_file,
        ]

        return_code = call(options)
        assert return_code == 0

        assert os.path.isfile(output_file)
        with h5py.File(output_file, 'r') as hdf5_file:
            # check dosages and its shape
            assert DOSAGES in hdf5_file
            dosages = hdf5_file[DOSAGES]
            assert dosages.shape == (7 + 12 + 10, 20)

            chrs = hdf5_file['variants/chr']
            rsids = hdf5_file['variants/rsids']
            positions = hdf5_file['variants/position']
            allele0s = hdf5_file['variants/allele0']
            allele1s = hdf5_file['variants/allele1']

            # chr1: check first snp
            assert chrs[0] == 1
            assert rsids[0] == 'rs1'
            assert positions[0] == 100
            assert allele0s[0] == 'T'
            assert allele1s[0] == 'A'
            assert truncate(dosages[0, 3], 4) == truncate(np.dot([0.84437, 0.11125, 0.04438], [0, 1, 2]), 4) == 0.2
            assert truncate(dosages[0, 2], 4) == truncate(np.dot([0.02175, 0.90098, 0.07728], [0, 1, 2]), 4) == 1.0555
            assert truncate(dosages[0, 19], 4) == truncate(np.dot([0.00219, 0.08987, 0.90798], [0, 1, 2]), 4) == 1.9058

            # chr1: check second snp
            assert chrs[1] == 1
            assert rsids[1] == 'rs2'
            assert positions[1] == 187
            assert allele0s[1] == 'C'
            assert allele1s[1] == 'T'
            assert truncate(dosages[1, 2], 4) == truncate(np.dot([0.95842, 0.00602, 0.03556], [0, 1, 2]), 4) == 0.0771
            assert truncate(dosages[1, 5], 4) == truncate(np.dot([0.0181, 0.8632, 0.11860], [0, 1, 2]), 4) == 1.1004
            assert truncate(dosages[1, 19], 4) == truncate(np.dot([0.06156, 0.00129, 0.93716], [0, 1, 2]), 4) == 1.8756

            # chr1: check last snp
            assert chrs[9] == 1
            assert rsids[9] == 'rs10'
            assert positions[9] == 839
            assert allele0s[9] == 'G'
            assert allele1s[9] == 'A'
            assert truncate(dosages[9, 0], 4) == truncate(np.dot([0.03161, 0.82957, 0.13882], [0, 1, 2]), 4) == 1.1072
            assert truncate(dosages[9, 4], 4) == truncate(np.dot([0.01553, 0.14800, 0.83647], [0, 1, 2]), 4) == 1.8209
            assert truncate(dosages[9, 19], 4) == truncate(np.dot([0.96104, 0.03167, 0.00729], [0, 1, 2]), 4) == 0.0462

            # chr2: check first snp
            assert chrs[10 + 0] == 2
            assert rsids[10 + 0] == 'rs2000000'
            assert positions[10 + 0] == 100
            assert allele0s[10 + 0] == 'C'
            assert allele1s[10 + 0] == 'G'
            assert truncate(dosages[10 + 0, 0], 4) == truncate(np.dot([0.86648, 0.00133, 0.13219], [0, 1, 2]), 4) == 0.2657
            assert truncate(dosages[10 + 0, 1], 4) == truncate(np.dot([0.08567, 0.04659, 0.86774], [0, 1, 2]), 4) == 1.7820
            assert truncate(dosages[10 + 0, 19], 4) == truncate(np.dot([0.16810, 0.60427, 0.22762], [0, 1, 2]), 4) == 1.0595

            # chr3: check middle snp
            assert chrs[10 + 12 + 3] == 3
            assert rsids[10 + 12 + 3] == 'rs3000003'
            assert positions[10 + 12 + 3] == 332
            assert allele0s[10 + 12 + 3] == 'G'
            assert allele1s[10 + 12 + 3] == 'A'
            assert truncate(dosages[10 + 12 + 3, 0], 4) == truncate(np.dot([0.03547, 0.04198, 0.92259], [0, 1, 2]), 4) == 1.8871, dosages[10 + 12 + 3, 0]
            assert truncate(dosages[10 + 12 + 3, 2], 4) == truncate(np.dot([0.08289, 0.83640, 0.08080], [0, 1, 2]), 4) == 0.9980, dosages[10 + 12 + 3, 2]
            assert truncate(dosages[10 + 12 + 3, 19], 4) == truncate(np.dot([0.01222, 0.01678, 0.97105], [0, 1, 2]), 4) == 1.9588, dosages[10 + 12 + 3, 19]

    def test_bgen_prefix(self):
        tmpdir = tempfile.mkdtemp()
        output_file = os.path.join(tmpdir, 'output.hdf5')

        options = [
            'python',
            self.convert_path,
            get_full_path('tests/data/set01_prefix'),
            output_file,
            '--bgen-prefix',
            'pref_',
        ]

        return_code = call(options)
        assert return_code == 0

        assert os.path.isfile(output_file)
        with h5py.File(output_file, 'r') as hdf5_file:
            # check dosages and its shape
            assert DOSAGES in hdf5_file
            dosages = hdf5_file[DOSAGES]
            assert dosages.shape == (7 + 10, 20)

            chrs = hdf5_file['variants/chr']
            rsids = hdf5_file['variants/rsids']
            positions = hdf5_file['variants/position']
            allele0s = hdf5_file['variants/allele0']
            allele1s = hdf5_file['variants/allele1']

            assert chrs.shape == (7 + 10,)
            assert rsids.shape == (7 + 10,)
            assert positions.shape == (7 + 10,)
            assert allele0s.shape == (7 + 10,)
            assert allele1s.shape == (7 + 10,)

            chrs_values = chrs[:]
            assert len(np.unique(chrs_values)) == 2
            assert 1 in chrs_values
            assert 2 not in chrs_values
            assert 3 in chrs_values

    def test_set00_samples_ids(self):
        tmpdir = tempfile.mkdtemp()
        output_file = os.path.join(tmpdir, 'output.hdf5')

        options = [
            'python',
            self.convert_path,
            get_full_path('tests/data/set00'),
            output_file,
        ]

        return_code = call(options)
        assert return_code == 0

        assert os.path.isfile(output_file)
        with h5py.File(output_file, 'r') as hdf5_file:
            samples_key = 'samples/ids'
            assert samples_key in hdf5_file, samples_key
            assert hdf5_file[samples_key].shape == (300,)
            assert hdf5_file[samples_key].chunks is None
            assert all(hdf5_file[samples_key][:].astype(str) == np.array([str(x) for x in range(1, 300 + 1)]))

    def test_set01_samples_ids(self):
        tmpdir = tempfile.mkdtemp()
        output_file = os.path.join(tmpdir, 'output.hdf5')

        options = [
            'python',
            self.convert_path,
            get_full_path('tests/data/set01'),
            output_file,
        ]

        return_code = call(options)
        assert return_code == 0

        assert os.path.isfile(output_file)
        with h5py.File(output_file, 'r') as hdf5_file:
            samples_key = 'samples/ids'
            assert samples_key in hdf5_file, samples_key
            assert hdf5_file[samples_key].shape == (20,)
            assert hdf5_file[samples_key].chunks is None
            assert all(hdf5_file[samples_key][:].astype(str) == np.array([str(x) for x in range(10, 200 + 1, 10)]))

    def test_default_dosages_chunks_when_less_than_10(self):
        tmpdir = tempfile.mkdtemp()
        output_file = os.path.join(tmpdir, 'output.hdf5')

        # first run with high scaleoffset option
        options = [
            'python',
            self.convert_path,
            get_full_path('tests/data/set01'),
            output_file,
            '--bgen-prefix', 'chr3i',
        ]

        return_code = call(options)
        assert return_code == 0

        assert os.path.isfile(output_file)
        with h5py.File(output_file, 'r') as hdf5_file:
            # check dosages and its shape
            assert DOSAGES in hdf5_file
            dosages = hdf5_file[DOSAGES]
            assert dosages.shape == (7, 20)

            assert dosages.chunks == (7, 20), dosages.chunks

            chrs = hdf5_file['variants/chr']
            rsids = hdf5_file['variants/rsids']
            positions = hdf5_file['variants/position']
            allele0s = hdf5_file['variants/allele0']
            allele1s = hdf5_file['variants/allele1']

            # chr3: check first snp
            assert chrs[0] == 3
            assert rsids[0] == 'rs3000000'
            assert positions[0] == 100
            assert allele0s[0] == 'T'
            assert allele1s[0] == 'A'
            assert truncate(dosages[0, 0], 4) == truncate(np.dot([0.00627, 0.94786, 0.04587], [0, 1, 2]), 4) == 1.0396, dosages[0, 0]
            assert truncate(dosages[0, 2], 4) == truncate(np.dot([0.82133, 0.17163, 0.00704], [0, 1, 2]), 4) == 0.1857, dosages[0, 2]
            assert truncate(dosages[0, 19], 4) == truncate(np.dot([0.90093, 0.07521, 0.02387], [0, 1, 2]), 4) == 0.1229, dosages[0, 19]

            # chr3: check middle snp
            assert chrs[3] == 3
            assert rsids[3] == 'rs3000003'
            assert positions[3] == 332
            assert allele0s[3] == 'G'
            assert allele1s[3] == 'A'
            assert truncate(dosages[3, 0], 4) == truncate(np.dot([0.03547, 0.04198, 0.92259], [0, 1, 2]), 4) == 1.8871, dosages[3, 0]
            assert truncate(dosages[3, 2], 4) == truncate(np.dot([0.08289, 0.83640, 0.08080], [0, 1, 2]), 4) == 0.9980, dosages[3, 2]
            assert truncate(dosages[3, 19], 4) == truncate(np.dot([0.01222, 0.01678, 0.97105], [0, 1, 2]), 4) == 1.9588, dosages[3, 19]

            # chr3: check last snp
            assert chrs[6] == 3
            assert rsids[6] == 'rs3000006'
            assert positions[6] == 585
            assert allele0s[6] == 'T'
            assert allele1s[6] == 'G'
            assert truncate(dosages[6, 1], 4) == truncate(np.dot([0.19479, 0.04479, 0.76042], [0, 1, 2]), 4) == 1.5656, dosages[6, 1]
            assert truncate(dosages[6, 17], 4) == truncate(np.dot([0.07464, 0.02689, 0.89846], [0, 1, 2]), 4) == 1.8238, dosages[6, 17]
            assert truncate(dosages[6, 19], 4) == truncate(np.dot([0.01770, 0.77147, 0.21083], [0, 1, 2]), 4) == 1.1931, dosages[6, 19]

    def test_default_dosages_chunks_when_greater_than_10(self):
        tmpdir = tempfile.mkdtemp()
        output_file = os.path.join(tmpdir, 'output.hdf5')

        # first run with high scaleoffset option
        options = [
            'python',
            self.convert_path,
            get_full_path('tests/data/set01'),
            output_file,
            '--bgen-prefix', 'chr2i',
        ]

        return_code = call(options)
        assert return_code == 0

        assert os.path.isfile(output_file)
        with h5py.File(output_file, 'r') as hdf5_file:
            # check dosages and its shape
            assert DOSAGES in hdf5_file
            dosages = hdf5_file[DOSAGES]
            assert dosages.shape == (12, 20)

            assert dosages.chunks == (10, 20), dosages.chunks

            chrs = hdf5_file['variants/chr']
            rsids = hdf5_file['variants/rsids']
            positions = hdf5_file['variants/position']
            allele0s = hdf5_file['variants/allele0']
            allele1s = hdf5_file['variants/allele1']

            # chr2: check first snp
            assert chrs[0] == 2
            assert rsids[0] == 'rs2000000'
            assert positions[0] == 100
            assert allele0s[0] == 'C'
            assert allele1s[0] == 'G'
            assert truncate(dosages[0, 0], 4) == truncate(np.dot([0.86648, 0.00133, 0.13219], [0, 1, 2]), 4) == 0.2657
            assert truncate(dosages[0, 1], 4) == truncate(np.dot([0.08567, 0.04659, 0.86774], [0, 1, 2]), 4) == 1.7820
            assert truncate(dosages[0, 19], 4) == truncate(np.dot([0.16810, 0.60427, 0.22762], [0, 1, 2]), 4) == 1.0595

            # chr2: check middle snp
            assert chrs[5] == 2
            assert rsids[5] == 'rs2000005'
            assert positions[5] == 473
            assert allele0s[5] == 'G'
            assert allele1s[5] == 'C'
            assert truncate(dosages[5, 0], 4) == truncate(np.dot([0.00502, 0.01640, 0.97858], [0, 1, 2]), 4) == 1.9735
            assert truncate(dosages[5, 5], 4) == truncate(np.dot([0.74327, 0.05947, 0.19729], [0, 1, 2]), 4) == 0.4540, dosages[5, 5]
            assert truncate(dosages[5, 19], 4) == truncate(np.dot([0.01201, 0.87458, 0.11345], [0, 1, 2]), 4) == 1.1014, dosages[5, 19]

    def test_specified_dosages_chunks_when_greater_than_99(self):
        tmpdir = tempfile.mkdtemp()
        output_file = os.path.join(tmpdir, 'output.hdf5')

        # first run with high scaleoffset option
        options = [
            'python',
            self.convert_path,
            get_full_path('tests/data/set00'),
            output_file,
            '--n-rows-chunk', '99',
        ]

        return_code = call(options)
        assert return_code == 0

        assert os.path.isfile(output_file)
        with h5py.File(output_file, 'r') as hdf5_file:
            # check dosages and its shape
            assert DOSAGES in hdf5_file
            dosages = hdf5_file[DOSAGES]
            assert dosages.shape == (250 + 150, 300)

            assert hdf5_file[DOSAGES].chunks == (99, 300)

            chrs = hdf5_file['variants/chr']
            rsids = hdf5_file['variants/rsids']
            positions = hdf5_file['variants/position']
            allele0s = hdf5_file['variants/allele0']
            allele1s = hdf5_file['variants/allele1']

            # chr 1: check first snp
            assert chrs[0] == 1
            assert rsids[0] == 'rs1'
            assert positions[0] == 100
            assert allele0s[0] == 'G'
            assert allele1s[0] == 'A'
            assert truncate(dosages[0, 0], 4) == truncate(np.dot([0.74909, 0.01333, 0.23758], [0, 1, 2]), 4) == 0.4884
            assert truncate(dosages[0, 2], 4) == truncate(np.dot([0.05437, 0.91567, 0.02999], [0, 1, 2]), 4) == 0.9756
            assert truncate(dosages[0, 3], 4) == truncate(np.dot([0.00650, 0.02577, 0.96773], [0, 1, 2]), 4) == 1.9612
            assert truncate(dosages[0, 5], 4) == truncate(np.dot([0.95803, 0.03895, 0.00302], [0, 1, 2]), 4) == 0.0449

            # chr 1: check second snp
            assert chrs[1] == 1
            assert rsids[1] == 'rs2'
            assert positions[1] == 181
            assert allele0s[1] == 'G'
            assert allele1s[1] == 'C'
            assert truncate(dosages[1, 0], 4) == truncate(np.dot([0.75232, 0.1172, 0.13048], [0, 1, 2]), 4) == 0.3781
            assert truncate(dosages[1, 299], 4) == truncate(np.dot([0.00937, 0.13421, 0.85642], [0, 1, 2]), 4) == 1.8470

            # chr 1: check last snp
            assert chrs[249] == 1
            assert rsids[249] == 'rs250'
            assert positions[249] == 18389
            assert allele0s[249] == 'T'
            assert allele1s[249] == 'C'
            assert truncate(dosages[249, 0], 4) == truncate(np.dot([0.04713, 0.94817, 0.00470], [0, 1, 2]), 4) == 0.9575
            assert truncate(dosages[249, 2], 4) == truncate(np.dot([0.07355, 0.50369, 0.42276], [0, 1, 2]), 4) == 1.3492
            assert truncate(dosages[249, 8], 4) == truncate(np.dot([0.01488, 0.07935, 0.90576], [0, 1, 2]), 4) == 1.8908
            assert truncate(dosages[249, 12], 4) == truncate(np.dot([0.95152, 0.02008, 0.02840], [0, 1, 2]), 4) == 0.0768

            # chr 2: check first snp
            assert chrs[250 + 0] == 2
            assert rsids[250 + 0] == 'rs2000000'
            assert positions[250 + 0] == 100
            assert allele0s[250 + 0] == 'A'
            assert allele1s[250 + 0] == 'G'
            assert truncate(dosages[250 + 0, 0], 4) == truncate(np.dot([0.9440, 0.02977, 0.02623], [0, 1, 2]), 4) == 0.0822
            assert truncate(dosages[250 + 0, 1], 4) == truncate(np.dot([0.06851, 0.85242, 0.07907], [0, 1, 2]), 4) == 1.0105
            assert truncate(dosages[250 + 0, 299], 4) == truncate(np.dot([0.08272, 0.89635, 0.02093], [0, 1, 2]), 4) == 0.9382

            # chr 2: check last snp
            assert chrs[250 + 149] == 2
            assert rsids[250 + 149] == 'rs2000149'
            assert positions[250 + 149] == 11226
            assert allele0s[250 + 149] == 'G'
            assert allele1s[250 + 149] == 'T'
            assert truncate(dosages[250 + 149, 0], 4) == truncate(np.dot([0.9462, 0.0535, 0.0003], [0, 1, 2]), 4) == 0.0541, dosages[250 + 149, 0]
            assert truncate(dosages[250 + 149, 1], 4) == truncate(np.dot([0.01373, 0.09532, 0.8909], [0, 1, 2]), 4) == 1.8771, dosages[250 + 149, 1]
            assert truncate(dosages[250 + 149, 299], 4) == truncate(np.dot([0.04675, 0.93974, 0.01351], [0, 1, 2]), 4) == 0.9667, dosages[250 + 149, 299]

    def test_specified_dosages_chunks_when_less_than_177(self):
        tmpdir = tempfile.mkdtemp()
        output_file = os.path.join(tmpdir, 'output.hdf5')

        # first run with high scaleoffset option
        options = [
            'python',
            self.convert_path,
            get_full_path('tests/data/set00'),
            output_file,
            '--bgen-prefix', 'chr2i',
            '--n-rows-chunk', '177',
        ]

        return_code = call(options)
        assert return_code == 0

        assert os.path.isfile(output_file)
        with h5py.File(output_file, 'r') as hdf5_file:
            # check dosages and its shape
            assert DOSAGES in hdf5_file
            dosages = hdf5_file[DOSAGES]
            assert dosages.shape == (150, 300)

            assert dosages.chunks == (150, 300), dosages.chunks

            chrs = hdf5_file['variants/chr']
            rsids = hdf5_file['variants/rsids']
            positions = hdf5_file['variants/position']
            allele0s = hdf5_file['variants/allele0']
            allele1s = hdf5_file['variants/allele1']

            # chr 2: check first snp
            assert chrs[0] == 2
            assert rsids[0] == 'rs2000000'
            assert positions[0] == 100
            assert allele0s[0] == 'A'
            assert allele1s[0] == 'G'
            assert truncate(dosages[0, 0], 4) == truncate(np.dot([0.9440, 0.02977, 0.02623], [0, 1, 2]), 4) == 0.0822
            assert truncate(dosages[0, 1], 4) == truncate(np.dot([0.06851, 0.85242, 0.07907], [0, 1, 2]), 4) == 1.0105
            assert truncate(dosages[0, 299], 4) == truncate(np.dot([0.08272, 0.89635, 0.02090], [0, 1, 2]), 4) == 0.9381, dosages[0, 299]

            # chr 2: check last snp
            assert chrs[149] == 2
            assert rsids[149] == 'rs2000149'
            assert positions[149] == 11226
            assert allele0s[149] == 'G'
            assert allele1s[149] == 'T'
            assert truncate(dosages[149, 0], 4) == truncate(np.dot([0.9462, 0.0534, 0.0003], [0, 1, 2]), 4) == 0.0540, dosages[149, 0]
            assert truncate(dosages[149, 1], 4) == truncate(np.dot([0.01373, 0.09532, 0.8909], [0, 1, 2]), 4) == 1.8771, dosages[149, 1]
            assert truncate(dosages[149, 299], 4) == truncate(np.dot([0.04675, 0.93974, 0.01351], [0, 1, 2]), 4) == 0.9667, dosages[149, 299]

    def test_scaleoffset(self):
        tmpdir = tempfile.mkdtemp()
        output_file = os.path.join(tmpdir, 'output.hdf5')

        # first run with high scaleoffset option
        options = [
            'python',
            self.convert_path,
            get_full_path('tests/data/set00'),
            output_file,
            '--scaleoffset',
            '8',
        ]

        return_code = call(options)
        assert return_code == 0

        assert os.path.isfile(output_file)
        orig_file_size = os.path.getsize(output_file)

        with h5py.File(output_file, 'r') as hdf5_file:
            assert hdf5_file[DOSAGES].scaleoffset == 8

        # now run with default scaleoffset option (4) and compare file sizes
        options = [
            'python',
            self.convert_path,
            get_full_path('tests/data/set00'),
            output_file,
        ]

        return_code = call(options)
        assert return_code == 0

        assert os.path.isfile(output_file)
        with h5py.File(output_file, 'r') as hdf5_file:
            dosages = hdf5_file[DOSAGES]
            assert dosages.scaleoffset == 4

            new_file_size = os.path.getsize(output_file)
            assert new_file_size * 1.7 < orig_file_size, (new_file_size, orig_file_size)

    def test_shrink_sample_ids(self):
        tmpdir = tempfile.mkdtemp()
        output_file = os.path.join(tmpdir, 'output.hdf5')

        # first run with no_shrink_ids option
        options = [
            'python',
            self.convert_path,
            get_full_path('tests/data/set00'),
            output_file,
            '--no-shrink-ids',
        ]

        return_code = call(options)
        assert return_code == 0

        assert os.path.isfile(output_file)
        orig_file_size = os.path.getsize(output_file)
        with h5py.File(output_file, 'r') as hdf5_file:
            samples_key = 'samples/ids'
            assert samples_key in hdf5_file, samples_key
            assert hdf5_file[samples_key].dtype == 'S30'
            assert hdf5_file[samples_key].shape == (300,)
            assert all(hdf5_file[samples_key][:].astype(str) == np.array([str(x) for x in range(1, 300 + 1)]))

        # now without it
        options = [
            'python',
            self.convert_path,
            get_full_path('tests/data/set00'),
            output_file,
        ]

        return_code = call(options)
        assert return_code == 0

        assert os.path.isfile(output_file)
        with h5py.File(output_file, 'r') as hdf5_file:
            samples_key = 'samples/ids'
            assert samples_key in hdf5_file, samples_key
            assert hdf5_file[samples_key].dtype == 'S3', hdf5_file[samples_key].dtype
            assert hdf5_file[samples_key].shape == (300,)
            assert all(hdf5_file[samples_key][:].astype(str) == np.array([str(x) for x in range(1, 300 + 1)]))

            new_file_size = os.path.getsize(output_file)
            assert new_file_size < orig_file_size, (new_file_size, orig_file_size)

    def test_no_sample_file(self):
        tmpdir = tempfile.mkdtemp()
        output_file = os.path.join(tmpdir, 'output.hdf5')

        options = [
            'python',
            self.convert_path,
            get_full_path('tests/data/set03_no_sample'),
            output_file,
        ]

        return_code, output = get_out(options)
        assert return_code != 0
        assert 'sample file' in output
        assert 'impv1.sample' in output

    def test_sample_file_option(self):
        tmpdir = tempfile.mkdtemp()
        output_file = os.path.join(tmpdir, 'output.hdf5')

        options = [
            'python',
            self.convert_path,
            get_full_path('tests/data/set04_sample'),
            output_file,
            '--sample-file', get_repository_path('set04_sample/impv2.sample')
        ]

        return_code, output = get_out(options)
        assert return_code == 0, output

        assert os.path.isfile(output_file)
        with h5py.File(output_file, 'r') as hdf5_file:
            samples_key = 'samples/ids'
            assert samples_key in hdf5_file, samples_key
            assert hdf5_file[samples_key].shape == (20,)
            assert all(hdf5_file[samples_key][:] == np.array([str(x) for x in range(100, 2000 + 1, 100)]))

    def test_no_compression(self):
        tmpdir = tempfile.mkdtemp()
        output_file = os.path.join(tmpdir, 'output.hdf5')

        # first run with no_shrink_ids option
        options = [
            'python',
            self.convert_path,
            get_full_path('tests/data/set00'),
            output_file,
            '--n-rows-chunk', '20',
        ]

        return_code = call(options)
        assert return_code == 0

        assert os.path.isfile(output_file)
        compressed_file_size = os.path.getsize(output_file)

        with h5py.File(output_file, 'r') as hdf5_file:
            dosages = hdf5_file[DOSAGES]
            assert dosages.shape == (250 + 150, 300)

            assert truncate(dosages[0, 0], 4) == truncate(np.dot([0.74909, 0.01333, 0.23758], [0, 1, 2]), 4) == 0.4884
            assert truncate(dosages[1, 299], 4) == truncate(np.dot([0.00937, 0.13421, 0.85642], [0, 1, 2]), 4) == 1.8470
            assert truncate(dosages[249, 8], 4) == truncate(np.dot([0.01488, 0.07935, 0.90576], [0, 1, 2]), 4) == 1.8908
            assert truncate(dosages[250 + 0, 1], 4) == truncate(np.dot([0.06851, 0.85242, 0.07907], [0, 1, 2]), 4) == 1.0105
            assert truncate(dosages[250 + 149, 299], 4) == truncate(np.dot([0.04675, 0.93974, 0.01351], [0, 1, 2]), 4) == 0.9667

        # now without it
        options = [
            'python',
            self.convert_path,
            get_full_path('tests/data/set00'),
            output_file,
            '--n-rows-chunk', '20',
            '--compression', 'disable',
        ]

        return_code = call(options)
        assert return_code == 0

        assert os.path.isfile(output_file)
        with h5py.File(output_file, 'r') as hdf5_file:
            dosages = hdf5_file[DOSAGES]
            assert dosages.shape == (250 + 150, 300)

            assert truncate(dosages[0, 0], 4) == truncate(np.dot([0.74909, 0.01333, 0.23758], [0, 1, 2]), 4) == 0.4884
            assert truncate(dosages[1, 299], 4) == truncate(np.dot([0.00937, 0.13421, 0.85642], [0, 1, 2]), 4) == 1.8470
            assert truncate(dosages[249, 8], 4) == truncate(np.dot([0.01488, 0.07935, 0.90576], [0, 1, 2]), 4) == 1.8908
            assert truncate(dosages[250 + 0, 1], 4) == truncate(np.dot([0.06851, 0.85242, 0.07907], [0, 1, 2]), 4) == 1.0105
            assert truncate(dosages[250 + 149, 299], 4) == truncate(np.dot([0.04675, 0.93974, 0.01351], [0, 1, 2]), 4) == 0.9667

        regular_file_size = os.path.getsize(output_file)
        # FIXME: new_file_size should be lesser than orig_file_size, but this could be related to the small
        # data set being used.
        assert compressed_file_size < regular_file_size, (compressed_file_size, regular_file_size)
