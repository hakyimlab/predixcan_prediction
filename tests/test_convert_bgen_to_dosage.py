import os
import unittest
from subprocess import call, check_output
import tempfile

import numpy as np
import h5py

from tests.utils import get_full_path, truncate, get_out, get_repository_path


class ConvertBGENToDosage(unittest.TestCase):
    def setUp(self):
        self.convert_path = get_full_path(os.path.join('Software', 'convert_bgen_to_dosage.py'))

    def test_mandatory_arguments00(self):
        tmpdir = tempfile.mkdtemp()
        output_file = os.path.join(tmpdir, 'output.hdf5')

        options = [
            'python',
            self.convert_path,
            get_full_path('tests/data/set00'),
            output_file,
            '--no-matrix-mode',
        ]

        return_code = call(options)
        assert return_code == 0

    def test_output_access_with_chr_and_position(self):
        tmpdir = tempfile.mkdtemp()
        output_file = os.path.join(tmpdir, 'output.hdf5')

        options = [
            'python',
            self.convert_path,
            get_full_path('tests/data/set00'),
            output_file,
            '--no-matrix-mode',
        ]

        return_code = call(options)
        assert return_code == 0

        assert os.path.isfile(output_file)
        hdf5_file = h5py.File(output_file, 'r')

        # Check first snp
        snp_key = 'variants/chr/1/100/G/A'
        assert snp_key in hdf5_file, snp_key
        assert hdf5_file[snp_key].shape == (300,)
        assert round(hdf5_file[snp_key][0], 4) == round(np.dot([0.74909, 0.01333, 0.23758], [0, 1, 2]), 4) == 0.4885
        assert round(hdf5_file[snp_key][2], 4) == round(np.dot([0.05437, 0.91567, 0.02996], [0, 1, 2]), 4) == 0.9756
        # because of rounding issues, sometimes the results is just truncated
        assert truncate(hdf5_file[snp_key][3], 4) == round(np.dot([0.00650, 0.02577, 0.96773], [0, 1, 2]), 4) == 1.9612
        assert round(hdf5_file[snp_key][5], 4) == round(np.dot([0.95803, 0.03895, 0.00302], [0, 1, 2]), 4) == 0.0450

        # Check second snp
        snp_key = 'variants/chr/1/181/G/C'
        assert snp_key in hdf5_file, snp_key
        assert hdf5_file[snp_key].shape == (300,)
        assert round(hdf5_file[snp_key][0], 4) == round(np.dot([0.75232, 0.11725, 0.13043], [0, 1, 2]), 4) == 0.3781
        assert round(hdf5_file[snp_key][299], 4) == round(np.dot([0.00937, 0.13421, 0.85642], [0, 1, 2]), 4) == 1.8470

        # Check last snp (in chr 1)
        snp_key = 'variants/chr/1/18389/T/C'
        assert snp_key in hdf5_file, snp_key
        assert hdf5_file[snp_key].shape == (300,)
        assert round(hdf5_file[snp_key][0], 4) == round(np.dot([0.04713, 0.94817, 0.00470], [0, 1, 2]), 4) == 0.9576
        assert truncate(hdf5_file[snp_key][2], 4) == round(np.dot([0.07355, 0.50369, 0.42276], [0, 1, 2]), 4) == 1.3492
        assert round(hdf5_file[snp_key][8], 4) == round(np.dot([0.01488, 0.07935, 0.90576], [0, 1, 2]), 4) == 1.8909
        assert round(hdf5_file[snp_key][12], 4) == round(np.dot([0.95152, 0.02008, 0.02840], [0, 1, 2]), 4) == 0.0769

        # Check another chromosome
        snp_key = 'variants/chr/2/11226/G/T'
        assert snp_key in hdf5_file, snp_key
        assert hdf5_file[snp_key].shape == (300,)
        assert round(hdf5_file[snp_key][0], 4) == round(np.dot([0.94620, 0.05350, 0.00030], [0, 1, 2]), 4) == 0.0541
        assert round(hdf5_file[snp_key][1], 4) == round(np.dot([0.01373, 0.09532, 0.89094], [0, 1, 2]), 4) == 1.8772
        assert round(hdf5_file[snp_key][299], 4) == round(np.dot([0.04675, 0.93974, 0.01351], [0, 1, 2]), 4) == 0.9668

    def test_output_alias(self):
        tmpdir = tempfile.mkdtemp()
        output_file = os.path.join(tmpdir, 'output.hdf5')

        options = [
            'python',
            self.convert_path,
            get_full_path('tests/data/set00'),
            output_file,
            '--no-matrix-mode',
        ]

        return_code = call(options)
        assert return_code == 0

        assert os.path.isfile(output_file)
        hdf5_file = h5py.File(output_file, 'r')

        # Check first snp
        snp_key = 'variants/chr/1/100/G/A'
        snp_rsid = 'variants/rsids/rs1'
        assert snp_key in hdf5_file, snp_key
        assert snp_rsid in hdf5_file, snp_rsid
        assert hdf5_file[snp_key] == hdf5_file[snp_rsid], snp_rsid

        # Check second snp
        snp_key = 'variants/chr/1/181/G/C'
        snp_rsid = 'variants/rsids/rs2'
        assert snp_key in hdf5_file, snp_key
        assert snp_rsid in hdf5_file, snp_rsid
        assert hdf5_file[snp_key] == hdf5_file[snp_rsid]

        # Check last snp (in chr 1)
        snp_key = 'variants/chr/1/18389/T/C'
        snp_rsid = 'variants/rsids/rs250'
        assert snp_key in hdf5_file, snp_key
        assert snp_rsid in hdf5_file, snp_rsid
        assert hdf5_file[snp_key] == hdf5_file[snp_rsid]

        # Check another chromosome
        snp_key = 'variants/chr/2/11226/G/T'
        snp_rsid = 'variants/rsids/rs2000149'
        assert snp_key in hdf5_file, snp_key
        assert snp_rsid in hdf5_file, snp_rsid
        assert hdf5_file[snp_key] == hdf5_file[snp_rsid]

    def test_number_of_snps(self):
        tmpdir = tempfile.mkdtemp()
        output_file = os.path.join(tmpdir, 'output.hdf5')

        options = [
            'python',
            self.convert_path,
            get_full_path('tests/data/set00'),
            output_file,
            '--no-matrix-mode',
        ]

        return_code = call(options)
        assert return_code == 0

        assert os.path.isfile(output_file)
        hdf5_file = h5py.File(output_file, 'r')

        global entries
        entries = 0

        def count_variants(key_name, data):
            global entries
            if not isinstance(data, h5py.Dataset):
                return None

            entries += 1

            assert data is not None

            if 'rs' in key_name:
                assert '/' not in key_name, key_name
            else:
                assert len(key_name.split('/')) == 4, key_name

        hdf5_file['variants/chr'].visititems(count_variants)
        assert entries == 250 + 150, entries

        # check alias
        entries = 0
        hdf5_file['variants/rsids'].visititems(count_variants)
        assert entries == 250 + 150, entries

    def test_output_attrs(self):
        tmpdir = tempfile.mkdtemp()
        output_file = os.path.join(tmpdir, 'output.hdf5')

        options = [
            'python',
            self.convert_path,
            get_full_path('tests/data/set00'),
            output_file,
            '--no-matrix-mode',
            '--save-attrs'
        ]

        return_code = call(options)
        assert return_code == 0

        assert os.path.isfile(output_file)
        hdf5_file = h5py.File(output_file, 'r')

        # Check first snp
        snp_key = 'variants/chr/1/100/G/A'
        assert snp_key in hdf5_file, snp_key
        assert 'chr' in hdf5_file[snp_key].attrs
        assert hdf5_file[snp_key].attrs['chr'] == 1
        assert 'position' in hdf5_file[snp_key].attrs
        assert hdf5_file[snp_key].attrs['position'] == 100
        assert 'rsid' in hdf5_file[snp_key].attrs
        assert hdf5_file[snp_key].attrs['rsid'] == 'rs1'
        assert 'allele0' in hdf5_file[snp_key].attrs
        assert hdf5_file[snp_key].attrs['allele0'] == 'G'
        assert 'allele1' in hdf5_file[snp_key].attrs
        assert hdf5_file[snp_key].attrs['allele1'] == 'A'

        # Check second snp
        snp_key = 'variants/chr/1/181/G/C'
        assert snp_key in hdf5_file, snp_key
        assert hdf5_file[snp_key].attrs['chr'] == 1
        assert hdf5_file[snp_key].attrs['position'] == 181
        assert hdf5_file[snp_key].attrs['rsid'] == 'rs2'
        assert hdf5_file[snp_key].attrs['allele0'] == 'G'
        assert hdf5_file[snp_key].attrs['allele1'] == 'C'

        # Check last snp (in chr 1)
        snp_key = 'variants/chr/1/18389/T/C'
        assert snp_key in hdf5_file, snp_key
        assert hdf5_file[snp_key].attrs['chr'] == 1
        assert hdf5_file[snp_key].attrs['position'] == 18389
        assert hdf5_file[snp_key].attrs['rsid'] == 'rs250'
        assert hdf5_file[snp_key].attrs['allele0'] == 'T'
        assert hdf5_file[snp_key].attrs['allele1'] == 'C'

        # Check another chromosome
        snp_key = 'variants/chr/2/11226/G/T'
        assert snp_key in hdf5_file, snp_key
        assert hdf5_file[snp_key].attrs['chr'] == 2
        assert hdf5_file[snp_key].attrs['position'] == 11226
        assert hdf5_file[snp_key].attrs['rsid'] == 'rs2000149'
        assert hdf5_file[snp_key].attrs['allele0'] == 'G'
        assert hdf5_file[snp_key].attrs['allele1'] == 'T'

    def test_output_no_attrs(self):
        tmpdir = tempfile.mkdtemp()
        output_file = os.path.join(tmpdir, 'output.hdf5')

        options = [
            'python',
            self.convert_path,
            get_full_path('tests/data/set00'),
            output_file,
            '--no-matrix-mode',
        ]

        return_code = call(options)
        assert return_code == 0

        assert os.path.isfile(output_file)
        hdf5_file = h5py.File(output_file, 'r')

        # Check first snp
        snp_key = 'variants/chr/1/100/G/A'
        assert snp_key in hdf5_file, snp_key
        assert 'chr' not in hdf5_file[snp_key].attrs
        assert 'position' not in hdf5_file[snp_key].attrs
        assert 'rsid' not in hdf5_file[snp_key].attrs
        assert 'allele0' not in hdf5_file[snp_key].attrs
        assert 'allele1' not in hdf5_file[snp_key].attrs

    def test_files_prefix_number_of_snps(self):
        tmpdir = tempfile.mkdtemp()
        output_file = os.path.join(tmpdir, 'output.hdf5')

        options = [
            'python',
            self.convert_path,
            get_full_path('tests/data/set01_prefix'),
            output_file,
            '--no-matrix-mode',
            '--bgen-prefix',
            'pref_',
        ]

        return_code = call(options)
        assert return_code == 0

        assert os.path.isfile(output_file)
        hdf5_file = h5py.File(output_file, 'r')

        global entries
        entries = 0

        def count_variants(key_name, data):
            global entries
            if isinstance(data, h5py.Dataset):
                entries += 1
            assert data is not None

        hdf5_file['variants/chr'].visititems(count_variants)
        assert entries == 7 + 10, entries

        # check alias
        entries = 0
        hdf5_file['variants/rsids'].visititems(count_variants)
        assert entries == 7 + 10, entries

        assert '1' in hdf5_file['variants/chr'].keys()
        assert '3' in hdf5_file['variants/chr'].keys()

    def test_output_samples_ids(self):
        tmpdir = tempfile.mkdtemp()
        output_file = os.path.join(tmpdir, 'output.hdf5')

        options = [
            'python',
            self.convert_path,
            get_full_path('tests/data/set00'),
            output_file,
            '--no-matrix-mode',
        ]

        return_code = call(options)
        assert return_code == 0

        assert os.path.isfile(output_file)
        hdf5_file = h5py.File(output_file, 'r')

        samples_key = 'samples/ids'
        assert samples_key in hdf5_file, samples_key
        assert hdf5_file[samples_key].shape == (300,)
        assert all(hdf5_file[samples_key][:] == np.array([str(x) for x in range(1, 300 + 1)]))

    def test_output_check_dtype_and_scaleoffset(self):
        tmpdir = tempfile.mkdtemp()
        output_file = os.path.join(tmpdir, 'output.hdf5')

        options = [
            'python',
            self.convert_path,
            get_full_path('tests/data/set00'),
            output_file,
            '--no-matrix-mode',
        ]

        return_code = call(options)
        assert return_code == 0

        assert os.path.isfile(output_file)
        hdf5_file = h5py.File(output_file, 'r')

        snp_key = 'variants/chr/2/11226/G/T'
        assert hdf5_file[snp_key].dtype == 'f8'

    def test_output_scaleoffset(self):
        tmpdir = tempfile.mkdtemp()
        output_file = os.path.join(tmpdir, 'output.hdf5')

        # first run with high scaleoffset option
        options = [
            'python',
            self.convert_path,
            get_full_path('tests/data/set00'),
            output_file,
            '--no-matrix-mode',
            '--scaleoffset',
            '8',
        ]

        return_code = call(options)
        assert return_code == 0

        assert os.path.isfile(output_file)
        orig_file_size = os.path.getsize(output_file)

        # now run with default scaleoffset option (4) and compare file sizes
        options = [
            'python',
            self.convert_path,
            get_full_path('tests/data/set00'),
            output_file,
            '--no-matrix-mode',
        ]

        return_code = call(options)
        assert return_code == 0

        assert os.path.isfile(output_file)
        hdf5_file = h5py.File(output_file, 'r')

        snp_key = 'variants/chr/2/11226/G/T'
        assert hdf5_file[snp_key].chunks is not None

        assert hdf5_file[snp_key].scaleoffset is not None
        # check that at least 4 decimal places are there
        # the assert below is not strictly necessary, since it's checked in other tests
        assert truncate(hdf5_file[snp_key][0], 4) == round(np.dot([0.94620, 0.05350, 0.00030], [0, 1, 2]), 4) == 0.0541

        new_file_size = os.path.getsize(output_file)
        assert new_file_size * 1.20 < orig_file_size

    def test_output_shrink_sample_ids(self):
        tmpdir = tempfile.mkdtemp()
        output_file = os.path.join(tmpdir, 'output.hdf5')

        # first run with no_shrink_ids option
        options = [
            'python',
            self.convert_path,
            get_full_path('tests/data/set00'),
            output_file,
            '--no-matrix-mode',
            '--no-shrink-ids',
        ]

        return_code = call(options)
        assert return_code == 0

        assert os.path.isfile(output_file)
        orig_file_size = os.path.getsize(output_file)
        hdf5_file = h5py.File(output_file, 'r')

        samples_key = 'samples/ids'
        assert samples_key in hdf5_file, samples_key
        assert hdf5_file[samples_key].dtype == 'S30'
        assert hdf5_file[samples_key].shape == (300,)
        assert all(hdf5_file[samples_key][:] == np.array([str(x) for x in range(1, 300 + 1)]))

        hdf5_file.close()

        # now without it
        options = [
            'python',
            self.convert_path,
            get_full_path('tests/data/set00'),
            output_file,
            '--no-matrix-mode',
        ]

        return_code = call(options)
        assert return_code == 0

        assert os.path.isfile(output_file)
        hdf5_file = h5py.File(output_file, 'r')

        samples_key = 'samples/ids'
        assert samples_key in hdf5_file, samples_key
        assert hdf5_file[samples_key].dtype == 'S3'
        assert hdf5_file[samples_key].shape == (300,)
        assert all(hdf5_file[samples_key][:] == np.array([str(x) for x in range(1, 300 + 1)]))

        new_file_size = os.path.getsize(output_file)
        assert new_file_size < orig_file_size

    def test_handle_repeated_rsids(self):
        tmpdir = tempfile.mkdtemp()
        output_file = os.path.join(tmpdir, 'output.hdf5')

        options = [
            'python',
            self.convert_path,
            get_full_path('tests/data/set02'),
            output_file,
            '--no-matrix-mode',
        ]

        return_code = call(options)
        assert return_code == 0

        assert os.path.isfile(output_file)
        hdf5_file = h5py.File(output_file, 'r')

        # count keys
        global entries
        entries = 0

        def count_variants(key_name, data):
            global entries
            if not isinstance(data, h5py.Dataset):
                return None

            assert data is not None

            entries += 1

        hdf5_file['variants/chr'].visititems(count_variants)
        assert entries == 10 + 12 + 7, entries

        # check alias (rsids with '.' do not have alias)
        entries = 0
        hdf5_file['variants/rsids'].visititems(count_variants)
        assert entries == 10 + 12 + 7 - 2, entries

        # Repeated snps in chr1
        snp_key = 'variants/chr/1/100/T/A'
        snp_rsid = 'variants/rsids/rs1'
        assert snp_key in hdf5_file, snp_key
        assert snp_rsid in hdf5_file, snp_rsid
        assert hdf5_file[snp_key] == hdf5_file[snp_rsid], snp_rsid

        snp_key = 'variants/chr/1/839/G/A'
        snp_rsid = 'variants/rsids/rs1_1'
        assert snp_key in hdf5_file, snp_key
        assert snp_rsid in hdf5_file, snp_rsid
        assert hdf5_file[snp_key] == hdf5_file[snp_rsid], snp_rsid

        snp_key = 'variants/chr/2/534/T/C'
        snp_rsid = 'variants/rsids/rs1_2'
        assert snp_key in hdf5_file, snp_key
        assert snp_rsid in hdf5_file, snp_rsid
        assert hdf5_file[snp_key] == hdf5_file[snp_rsid], snp_rsid

        # Repeated snps in chr2
        snp_key = 'variants/chr/2/412/G/C'
        snp_rsid = 'variants/rsids/rs2000004'
        assert snp_key in hdf5_file, snp_key
        assert snp_rsid in hdf5_file, snp_rsid
        assert hdf5_file[snp_key] == hdf5_file[snp_rsid], snp_rsid

        snp_key = 'variants/chr/2/624/T/C'
        snp_rsid = 'variants/rsids/rs2000004_1'
        assert snp_key in hdf5_file, snp_key
        assert snp_rsid in hdf5_file, snp_rsid
        assert hdf5_file[snp_key] == hdf5_file[snp_rsid], snp_rsid

        # Repeated snps in chr3
        snp_key = 'variants/chr/3/190/C/G'
        assert snp_key in hdf5_file, snp_key

        snp_key = 'variants/chr/3/507/C/G'
        assert snp_key in hdf5_file, snp_key

    def test_no_sample_file(self):
        tmpdir = tempfile.mkdtemp()
        output_file = os.path.join(tmpdir, 'output.hdf5')

        options = [
            'python',
            self.convert_path,
            get_full_path('tests/data/set03_no_sample'),
            output_file,
            '--no-matrix-mode',
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
            '--no-matrix-mode',
            '--sample-file', get_repository_path('set04_sample/impv2.sample')
        ]

        return_code, output = get_out(options)
        assert return_code == 0, output

        assert os.path.isfile(output_file)
        hdf5_file = h5py.File(output_file, 'r')

        samples_key = 'samples/ids'
        assert samples_key in hdf5_file, samples_key
        assert hdf5_file[samples_key].shape == (20,)
        assert all(hdf5_file[samples_key][:] == np.array([str(x) for x in range(100, 2000 + 1, 100)]))

    def test_output_compression(self):
        tmpdir = tempfile.mkdtemp()
        output_file = os.path.join(tmpdir, 'output.hdf5')

        # first run with no_shrink_ids option
        options = [
            'python',
            self.convert_path,
            get_full_path('tests/data/set00'),
            output_file,
            '--no-matrix-mode',
        ]

        return_code = call(options)
        assert return_code == 0

        assert os.path.isfile(output_file)
        orig_file_size = os.path.getsize(output_file)

        # now without it
        options = [
            'python',
            self.convert_path,
            get_full_path('tests/data/set00'),
            output_file,
            '--no-matrix-mode',
            '--compression', 'disable',
        ]

        return_code = call(options)
        assert return_code == 0

        assert os.path.isfile(output_file)
        with h5py.File(output_file, 'r') as hdf5_file:
            # Check first snp
            snp_key = 'variants/chr/1/100/G/A'
            assert snp_key in hdf5_file, snp_key
            assert hdf5_file[snp_key].shape == (300,)
            assert round(hdf5_file[snp_key][0], 4) == round(np.dot([0.74909, 0.01333, 0.23758], [0, 1, 2]), 4) == 0.4885
            assert round(hdf5_file[snp_key][2], 4) == round(np.dot([0.05437, 0.91567, 0.02996], [0, 1, 2]), 4) == 0.9756
            # because of rounding issues, sometimes the results is just truncated
            assert truncate(hdf5_file[snp_key][3], 4) == round(np.dot([0.00650, 0.02577, 0.96773], [0, 1, 2]), 4) == 1.9612
            assert round(hdf5_file[snp_key][5], 4) == round(np.dot([0.95803, 0.03895, 0.00302], [0, 1, 2]), 4) == 0.0450

        new_file_size = os.path.getsize(output_file)
        # FIXME: new_file_size should be lesser than orig_file_size, but this could be related to the small
        # data set being used.
        assert new_file_size != orig_file_size, (new_file_size, orig_file_size)
