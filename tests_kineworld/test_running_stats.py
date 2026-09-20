import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('stats_test',Path(__file__).resolve().parents[1]/'dexbotic/data/dataset/dw05/transform/normalize.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class StatsTests(unittest.TestCase):
    def test_large_offset_small_variance_matches_numpy(self):
        x=1e9+np.random.default_rng(42).normal(size=(500,3))
        stats=module.RunningStats()
        for batch in np.array_split(x,7):stats.update(batch)
        actual=stats.get_statistics()
        np.testing.assert_allclose(actual.std,x.std(axis=0),rtol=2e-7,atol=1e-7)
        np.testing.assert_allclose(actual.mean,x.mean(axis=0),rtol=1e-14)

    def test_integer_square_does_not_overflow(self):
        x=np.array([[30000],[30002],[30004]],dtype=np.int16)
        stats=module.RunningStats();stats.update(x)
        np.testing.assert_allclose(stats.get_statistics().std,x.astype(float).std(axis=0))

    def test_bad_batches_leave_previous_statistics_intact(self):
        stats=module.RunningStats();stats.update(np.array([[1.,2.],[3.,4.]]))
        before=stats.get_statistics()
        for x in (np.empty((0,2)),np.array([[np.nan,2.]]),np.array([[np.inf,2.]]),np.ones((2,3)),np.ones((1,1,1))):
            with self.assertRaises(ValueError):stats.update(x)
            np.testing.assert_array_equal(before.mean,stats.get_statistics().mean)
            np.testing.assert_array_equal(before.std,stats.get_statistics().std)

if __name__=='__main__':unittest.main()
