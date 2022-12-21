import numpy as np
import unittest

from dpgen2.fp.gaussian import (
    GaussianInputs,
    PrepGaussian,
    RunGaussian,
    dpdata,
)
from dargs import Argument

class TestPrepGaussian(unittest.TestCase):
    def test_prep_gaussian(self):
        inputs = GaussianInputs(
            keywords="force b3lyp/6-31g*",
        )
        ta = GaussianInputs.args()
        base = Argument("base", dict, ta)
        data = base.normalize_value(inputs.data, trim_pattern="_*")
        base.check_value(data, strict=True)
        system = dpdata.LabeledSystem(data={
            'atom_names': ['H'],
            'atom_numbs': [1],
            'atom_types': np.zeros(1, dtype=int),
            'cells': np.eye(3).reshape(1, 3, 3),
            'coords': np.zeros((1, 1, 3)),
            'energies': np.zeros(1),
            'forces': np.zeros((1, 1, 3)),
            'orig': np.zeros(3),
            'nopbc': True,
        })
        prep_gaussian = PrepGaussian()
        prep_gaussian.prep_task(
            conf_frame=system,
            inputs=inputs,
        )


class TestRunGaussian(unittest.TestCase):
    def test_run_gaussian(self):
        run_gaussian = RunGaussian()
        output = 'mock_output'
        out_name, log_name = run_gaussian.run_task(
            'g16',
            output,
        )
        assert out_name == output
        assert log_name == 'task.log'
