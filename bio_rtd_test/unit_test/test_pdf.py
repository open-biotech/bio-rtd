import unittest
import numpy as np
from bio_rtd.utils.vectors import true_start, true_end
from bio_rtd import pdf, peak_shapes, logger


class GaussianTest(unittest.TestCase):

    @staticmethod
    def get_gauss_pdf_with_cutoff(t, rt_mean, sigma, cutoff) -> np.ndarray:
        p: np.ndarray = peak_shapes.gaussian(t, rt_mean, sigma)
        # cut at end
        p = p[:true_end(p >= cutoff * p.max())]
        # set to 0 at front
        p[:true_start(p >= cutoff * p.max())] = 0
        # return normalized profile (integral = 1)
        return p / p.sum() / t[1]

    # define inlet with single specie
    def run_gauss_fixed_dispersion(self, t, rt_mean_list, sigma_list, cutoff) -> None:
        for rt_mean in rt_mean_list:
            for sigma in sigma_list:
                p = self.get_gauss_pdf_with_cutoff(t, rt_mean, sigma, cutoff)
                pd = pdf.GaussianFixedDispersion(t, dispersion_index=sigma ** 2 / rt_mean, cutoff=cutoff)
                pd.log.log_level = pd.log.ERROR + 10
                pd.update_pdf(rt_mean=rt_mean)
                np.testing.assert_array_equal(pd.get_p(), p)
                pd.update_pdf(rt_mean=rt_mean / 2)
                p = self.get_gauss_pdf_with_cutoff(t, rt_mean / 2, sigma / np.sqrt(2), cutoff)
                np.testing.assert_array_almost_equal(pd.get_p(), p, 5)

    # define inlet with single specie
    def test_gauss_fixed_dispersion(self) -> None:
        t = np.linspace(0, 100, 1500)
        self.run_gauss_fixed_dispersion(t,
                                        sigma_list=[20, 5.5, 30, 80],
                                        rt_mean_list=[5, 10, 0.4, 100],
                                        cutoff=0.001)
        t = np.linspace(0, 900, 901)
        self.run_gauss_fixed_dispersion(t,
                                        sigma_list=[200, 5.5, 30, 880],
                                        rt_mean_list=[5, 0.4, 100],
                                        cutoff=0.001)

        pd = pdf.GaussianFixedDispersion(t, dispersion_index=5, cutoff=0.03)
        pd.log.log_level = pd.log.ERROR
        with self.assertRaises(RuntimeError):
            pd.update_pdf(rt_mean=6.6)
        pd.log.log_level = pd.log.ERROR + 10
        pd.update_pdf(rt_mean=6.6)
        p1 = pd.get_p().copy()
        pd.trim_and_normalize = False
        pd.update_pdf(rt_mean=6.6)
        p2 = pd.get_p()
        self.assertTrue(p1.size < p2.size)
        self.assertAlmostEqual(0.892, p2.sum() * t[1], 3)

    # define inlet with single specie
    def run_gauss_fixed_relative_with(self, t, rt_mean_list, sigma_list, cutoff, ignore_logger=True) -> None:
        for rt_mean in rt_mean_list:
            for sigma in sigma_list:
                p = self.get_gauss_pdf_with_cutoff(t, rt_mean, sigma, cutoff)
                pd = pdf.GaussianFixedRelativeWidth(t, relative_sigma=sigma / rt_mean, cutoff=cutoff)
                if ignore_logger:
                    pd.log.log_level = pd.log.ERROR + 10
                else:
                    pd.log = logger.StrictLogger()
                pd.update_pdf(rt_mean=rt_mean)
                np.testing.assert_array_equal(pd.get_p(), p)
                pd.update_pdf(rt_mean=rt_mean / 2)
                p = self.get_gauss_pdf_with_cutoff(t, rt_mean / 2, sigma / 2, cutoff)
                np.testing.assert_array_almost_equal(pd.get_p(), p, 5)

    # define inlet with single specie
    def test_gauss_fixed_relative_with(self) -> None:
        t = np.linspace(0, 100, 1500)
        self.run_gauss_fixed_relative_with(t,
                                           sigma_list=[20, 5.5, 30, 80],
                                           rt_mean_list=[5, 10, 0.4, 100],
                                           cutoff=0.001)
        t = np.linspace(0, 900, 901)
        self.run_gauss_fixed_relative_with(t,
                                           sigma_list=[200, 5.5, 30, 80],
                                           rt_mean_list=[5, 0.4, 100],
                                           cutoff=0.001)
        # Test logger binding.
        with self.assertRaises(RuntimeError):
            self.run_gauss_fixed_relative_with(t,
                                               sigma_list=[5.5],
                                               rt_mean_list=[5],
                                               cutoff=0.001,
                                               ignore_logger=False)


class EmgTest(unittest.TestCase):

    @staticmethod
    def get_emg_pdf_with_cutoff(t, rt_mean, sigma, skew, cutoff) -> np.ndarray:
        p: np.ndarray = peak_shapes.emg(t, rt_mean, sigma, skew)
        # cut at end
        p = p[:true_end(p >= cutoff * p.max())]
        # set to 0 at front
        p[:true_start(p >= cutoff * p.max())] = 0
        # return normalized profile (integral = 1)
        return p / p.sum() / t[1]

    # define inlet with single specie
    # noinspection DuplicatedCode
    def run_emg_fixed_dispersion(self, t, rt_mean_list, sigma_list, skew_list, cutoff, ignore_logger=True) -> None:
        for skew in skew_list:
            for sigma in sigma_list:
                for rt_mean in rt_mean_list:
                    if rt_mean < 1 / skew:  # peak max would be at t < 0
                        continue
                    p = self.get_emg_pdf_with_cutoff(t, rt_mean, sigma, skew, cutoff)
                    pd = pdf.ExpModGaussianFixedDispersion(t,
                                                           dispersion_index=sigma ** 2 / rt_mean,
                                                           skew=skew)
                    if ignore_logger:
                        pd.log.log_level = pd.log.ERROR + 10
                    else:
                        pd.log = logger.StrictLogger()
                    pd.cutoff_relative_to_max = cutoff
                    pd.update_pdf(rt_mean=rt_mean)
                    np.testing.assert_array_almost_equal(pd.get_p(), p, 3)
                    pd.update_pdf(rt_mean=rt_mean / 2)
                    p = self.get_emg_pdf_with_cutoff(t, rt_mean / 2, sigma / np.sqrt(2), skew, cutoff)
                    np.testing.assert_array_almost_equal(pd.get_p(), p, 3)
                    pd.update_pdf(rt_mean=rt_mean / 2, skew=skew / 2)
                    p = self.get_emg_pdf_with_cutoff(t, rt_mean / 2, sigma / np.sqrt(2), skew / 2, cutoff)
                    np.testing.assert_array_almost_equal(pd.get_p(), p, 3)

    # define inlet with single specie
    def test_emg_fixed_dispersion(self) -> None:
        t = np.linspace(0, 100, 1500)
        self.run_emg_fixed_dispersion(t,
                                      rt_mean_list=[5, 10, 0.4, 100],
                                      sigma_list=[2, 5.5, 30],
                                      skew_list=[1, 1 / 2, 1 / 4, 1 / 20],
                                      cutoff=0.001)
        t = np.linspace(0, 900, 901)
        self.run_emg_fixed_dispersion(t,
                                      rt_mean_list=[5, 0.4, 100],
                                      sigma_list=[2, 5.5, 30],
                                      skew_list=[1, 1 / 2, 1 / 4, 1 / 20],
                                      cutoff=0.001)
        # Make sure logger gets passed to emg.
        with self.assertRaises(RuntimeError):
            self.run_emg_fixed_dispersion(t,
                                          rt_mean_list=[5],
                                          sigma_list=[2],
                                          skew_list=[1],
                                          cutoff=0.001,
                                          ignore_logger=False)

    # define inlet with single specie
    # noinspection DuplicatedCode
    def run_emg_fixed_relative_width(self, t, rt_mean_list, sigma_list,
                                     skew_list, cutoff, ignore_logger=True) -> None:
        for skew in skew_list:
            for sigma in sigma_list:
                for rt_mean in rt_mean_list:
                    if rt_mean < 1 / skew:  # peak max would be at t < 0
                        continue
                    p = self.get_emg_pdf_with_cutoff(t,
                                                     rt_mean,
                                                     sigma,
                                                     skew,
                                                     cutoff)
                    pd = pdf.ExpModGaussianFixedRelativeWidth(
                        t,
                        sigma_relative=sigma / rt_mean,
                        tau_relative=1 / skew / rt_mean)
                    if ignore_logger:
                        pd.log.log_level = pd.log.ERROR + 10
                    else:
                        pd.log = logger.StrictLogger()
                    pd.cutoff_relative_to_max = cutoff
                    pd.update_pdf(rt_mean=rt_mean)
                    np.testing.assert_array_almost_equal(pd.get_p(), p, 3)
                    pd.update_pdf(rt_mean=rt_mean / 2)
                    p = self.get_emg_pdf_with_cutoff(t,
                                                     rt_mean / 2,
                                                     sigma / 2,
                                                     skew * 2,
                                                     cutoff)
                    np.testing.assert_array_almost_equal(pd.get_p(), p, 3)
                    pd.update_pdf(rt_mean=rt_mean / 2, skew=skew / 3)
                    p = self.get_emg_pdf_with_cutoff(t,
                                                     rt_mean / 2,
                                                     sigma / 2,
                                                     skew / 3,
                                                     cutoff)
                    np.testing.assert_array_almost_equal(pd.get_p(), p, 3)

    # define inlet with single specie
    def test_emg_fixed_relative_width(self) -> None:
        t = np.linspace(0, 100, 1500)
        self.run_emg_fixed_relative_width(t,
                                          rt_mean_list=[5, 10, 0.4, 100],
                                          sigma_list=[2, 5.5, 30],
                                          skew_list=[1, 1 / 2, 1 / 4, 1 / 20],
                                          cutoff=0.001)
        t = np.linspace(0, 900, 901)
        self.run_emg_fixed_relative_width(t,
                                          rt_mean_list=[5, 0.4, 100],
                                          sigma_list=[2, 5.5, 30],
                                          skew_list=[1, 1 / 2, 1 / 4, 1 / 20],
                                          cutoff=0.001)
        # Test binding with logger.
        with self.assertRaises(RuntimeError):
            self.run_emg_fixed_relative_width(t,
                                              rt_mean_list=[5],
                                              sigma_list=[2],
                                              skew_list=[1],
                                              cutoff=0.001,
                                              ignore_logger=False)


class TanksInSeriesTest(unittest.TestCase):

    @staticmethod
    def get_tanks_in_series_pdf_with_cutoff(t, rt_mean, n_tanks, cutoff) -> np.ndarray:
        p: np.ndarray = peak_shapes.tanks_in_series(t, rt_mean, n_tanks)
        # cut at end
        p = p[:true_end(p >= cutoff * p.max())]
        # set to 0 at front
        p[:true_start(p >= cutoff * p.max())] = 0
        # return normalized profile (integral = 1)
        return p / p.sum() / t[1]

    # define inlet with single specie
    # noinspection DuplicatedCode
    def run_tanks_in_series(self, t, rt_mean_list, n_tank_list, cutoff, ignore_logger=True) -> None:
        for n_tanks in n_tank_list:
            for rt_mean in rt_mean_list:
                p = self.get_tanks_in_series_pdf_with_cutoff(t, rt_mean, n_tanks, cutoff)
                pd = pdf.TanksInSeries(t, n_tanks)
                if ignore_logger:
                    pd.log.log_level = pd.log.ERROR + 10
                else:
                    pd.log = logger.StrictLogger()
                pd.cutoff_relative_to_max = cutoff
                pd.update_pdf(rt_mean=rt_mean)
                np.testing.assert_array_almost_equal(pd.get_p(), p, 3)
                pd.update_pdf(rt_mean=rt_mean / 2)
                p = self.get_tanks_in_series_pdf_with_cutoff(t, rt_mean / 2, n_tanks, cutoff)
                np.testing.assert_array_almost_equal(pd.get_p(), p, 3)

    # define inlet with single specie
    def test_tanks_in_series(self) -> None:
        t = np.linspace(0, 100, 1500)
        self.run_tanks_in_series(t,
                                 rt_mean_list=[5, 10, 0.4, 100],
                                 n_tank_list=[1, 2, 15, 30],
                                 cutoff=0.001)
        t = np.linspace(0, 900, 901)
        self.run_tanks_in_series(t,
                                 rt_mean_list=[5, 10, 400, 100],
                                 n_tank_list=[1, 2, 15, 30],
                                 cutoff=0.001)
        with self.assertRaises(RuntimeError):
            self.run_tanks_in_series(t,
                                     rt_mean_list=[20000],
                                     n_tank_list=[2],
                                     cutoff=0.001,
                                     ignore_logger=False)
