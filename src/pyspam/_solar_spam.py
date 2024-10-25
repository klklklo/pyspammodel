import numpy as np
import xarray as xr
import pyspam._misc as _m


class SolarSpam:
    '''
    Solar-SPAM model class.
    '''
    def __init__(self):
        self._dataset = _m.get_solar_spam_coeffs()
        self._coeffs = np.vstack((np.array(self._dataset['P1'], dtype=np.float64),
                                  np.array(self._dataset['P2'], dtype=np.float64),
                                  np.array(self._dataset['P3'], dtype=np.float64))).transpose()

    def _get_f107(self, f107):
        '''
        Method for creating the daily F10.7 index matrix that will be used to calculate the spectrum.
        Returns a matrix with rows [F10.7 ^ 2; F10.7; 1] for each passed value F10.7.
        :param f107: single value of the daily index F10.7 or an array of such values.
        :return: numpy array for model calculation.
        '''
        if isinstance(f107, float):
            return np.array([f107 ** 2, f107, 1], dtype=np.float64)[None, :]
        return np.vstack([np.array([x ** 2, x, 1]) for x in f107], dtype=np.float64)

    # def get_spectral_bands(self, f107):
    #     '''
    #     Model calculation method. Returns the xarray dataset values of radiation fluxes in all intervals
    #     of the spectrum of the interval 10-105 nm
    #     :param f107: single value of the daily index F10.7 or an array of such values
    #     :return: xarray Dataset [euv_flux_spectra, lband, uband, center]
    #     '''
    #     F107 = self._get_f107(f107)
    #     res = np.dot(self._coeffs, F107.T)
    #     return xr.Dataset(data_vars={'euv_flux_spectra': (('band_center', 'f107'), res),
    #                                  'lband': ('band_number', self._dataset['lband'].values),
    #                                  'uband': ('band_number', self._dataset['uband'].values),
    #                                  'center': ('band_number', self._dataset['center'].values)},
    #                       coords={'band_center': self._dataset['center'].values,
    #                               'f107': F107[:, 0],
    #                               'band_number': np.arange(20)})

    def get_spectral_bands(self, f107):
        '''
        Model calculation method. Returns the xarray dataset values of radiation fluxes in all intervals
        of the spectrum of the interval 10-105 nm
        :param f107: single value of the daily index F10.7 or an array of such values
        :return: xarray Dataset [euv_flux_spectra, lband, uband, center]
        '''
        F107 = self._get_f107(f107)
        res = np.dot(self._coeffs, F107.T)
        return xr.Dataset(data_vars={'euv_flux_spectra': (('band_center', 'f107'), res),
                                     'line_lambda': ('band_number', self._dataset['lambda'].values)},
                          coords={'band_center': self._dataset['lambda'].values,
                                  'f107': F107[:, 0],
                                  'band_number': np.arange(189)})

    def get_spectra(self, f107):
        '''
        Model calculation method. Combines the get_spectra_lines() and get_spectral_bands() methods
        :param f107: single value of the daily index F10.7 or an array of such values.
        :return: xarray Dataset [euv_flux_spectra, lband, uband, center]
        '''
        return self.get_spectral_bands(f107)
