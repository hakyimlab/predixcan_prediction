import math
from os.path import dirname, abspath, join
from tempfile import TemporaryFile
from subprocess import check_output, CalledProcessError


def get_repository_path(data_filename):
    directory = dirname(abspath(__file__))
    directory = join(directory, 'data/')
    return join(directory, data_filename)


def get_full_path(filename):
    root_dir = dirname(dirname(abspath(__file__)))
    return join(root_dir, filename)


def truncate(f, n=4):
    return math.floor(f * 10 ** n) / 10 ** n


def get_out(args):
    with TemporaryFile() as t:
        try:
            out = check_output(args, stderr=t)
            return 0, out
        except CalledProcessError as e:
            t.seek(0)
            return e.returncode, t.read()
