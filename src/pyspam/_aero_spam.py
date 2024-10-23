import numpy as np
import xarray as xr
import pyspam._misc as _m
import math


class AeroSpam:
    '''
    Пипипупу
    '''
    def __init__(self):
        self._bands_dataset, self._lines_dataset = _m.get_aero_spam_coeffs()
        self._bands_coeffs = np.vstack((np.array(self._bands_dataset['P1'], dtype=np.float64),
                                        np.array(self._bands_dataset['P2'], dtype=np.float64),
                                        np.array(self._bands_dataset['P3'], dtype=np.float64))).transpose()

        self._lines_coeffs = np.vstack((np.array(self._lines_dataset['P1'], dtype=np.float64),
                                        np.array(self._lines_dataset['P2'], dtype=np.float64),
                                        np.array(self._lines_dataset['P3'], dtype=np.float64))).transpose()

    def _get_f107(self, f107):
        '''

        :param f107: single value of the daily index F10.7 or an array of such values
        :return:
        '''

        if isinstance(f107, float):
            return np.array([f107 ** 2, f107, 1], dtype=np.float64)[None, :]

        return np.vstack([np.array([x ** 2, x, 1]) for x in f107], dtype=np.float64)

    def get_spectral_lines(self, f107):
        '''
        Model calculation method. Returns the values of radiation fluxes in all lines
        of the spectrum of the interval 10-105 nm
        :param f107: single value of the daily index F10.7 or an array of such values
        :return: xarray Dataset [euv_flux_spectra, line_lambda]
        '''
        x = self._get_f107(f107)
        res = np.dot(self._lines_coeffs, x.T)
        return xr.Dataset(data_vars={'euv_flux_spectra': (('line_number', 'f107'), res),
                                     'line_lambda': ('line_number', self._lines_dataset['lambda'].values)},
                          coords={'line_number': np.arange(17),
                                  'f107': x[:, 1],
                                  })

    def get_spectral_bands(self, f107):
        '''
        Model calculation method. Returns the xarray dataset values of radiation fluxes in all intervals
        of the spectrum of the interval 10-105 nm
        :param f107: single value of the daily index F10.7 or an array of such values
        :return: xarray Dataset [euv_flux_spectra, lband, uband, center]
        '''
        x = self._get_f107(f107)
        res = np.dot(self._bands_coeffs, x.T)
        return xr.Dataset(data_vars={'euv_flux_spectra': (('band_center', 'f107'), res),
                                     'lband': ('band_number', self._bands_dataset['lband'].values),
                                     'uband': ('band_number', self._bands_dataset['uband'].values),
                                     'center': ('band_number', self._bands_dataset['center'].values)},
                          coords={'band_center': self._bands_dataset['center'].values,
                                  'f107': x[:, 1],
                                  'band_number': np.arange(20)})

    def get_spectra(self, lac):
        '''
        Model calculation method. Combines the get_spectra_lines() and get_spectral_bands() methods
        :param f107: single value of the daily index F10.7 or an array of such values
        :return: xarray Dataset [euv_flux_spectra], xarray Dataset [euv_flux_spectra, lband, uband, center]
        '''
        return self.get_spectral_bands(lac), self.get_spectral_lines(lac)
