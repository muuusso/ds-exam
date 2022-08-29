import os

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import simpson
from scipy.optimize import nnls
from sklearn.preprocessing import MinMaxScaler

# load reference spectra
def load_reference():
    reference_path = os.path.join('data', 'Database Raman')

    labels  = [] # name of the mineral
    domains = [] # wavenumber domain data
    spectra = [] # spectrum data

    for filename in os.listdir(reference_path):
        # avoid non spectra files
        if sum(s.isupper() for s in filename) < 2:
            labels.append(filename.split('.')[0])

            filename = os.path.join(reference_path, filename)
            domains.append(np.loadtxt(filename)[:, 0])
            spectra.append(np.loadtxt(filename)[:, 1])

    return labels, domains, spectra


# interpolate reference spectra to the target domain
# return the interpolated spectra and the interpolation domain
def interpolate_reference(domains, spectra, target_domain):
    # values choosen by inspection of reference spectra domains
    int_domain = target_domain[target_domain > 100]
    int_domain = int_domain[int_domain < 1200]

    # interpolated spectra
    int_spectra = np.zeros((len(spectra), len(int_domain)))

    for i, (domain, spectrum) in enumerate(zip(domains, spectra)):
        # "correcting" reference spectra for inexplicable negative values
        spectrum = np.where(spectrum < 0, 0, spectrum)
        spline = CubicSpline(domain[1:-1], spectrum[1:-1], extrapolate=False) 
        int_spectra[i] = spline(int_domain)
        # replacing out of reference spectra domain values with zero
        int_spectra[i] = np.nan_to_num(int_spectra[i])
        # normalize on target domain
        int_spectra[i] = int_spectra[i] / simpson(int_spectra[i], int_domain)

    return int_spectra, int_domain


# classify spectrum as linear combination of the reference spectra
# return a list with percentages of materials in spectrum and 
# a list with corresponding labels
# return -1 for every coefficient if unable to classify
def classify_spectrum(target_domain, target_spectrum):
    # target_domain : 1d numpy.array
    # target_spectrum : 1d numpy.array
    
    # load reference spectra
    labels, domains, spectra = load_reference()
    # interpolate to target domain
    int_spectra, int_domain = interpolate_reference(domains, 
                                                    spectra, 
                                                    target_domain)

    # scale between 0 and 1
    scaler = MinMaxScaler().fit(int_spectra)
    int_spectra = scaler.transform(int_spectra)
    
    # select target spectrum part corresponding to interpolation domain
    int_dom_ind = np.where((target_domain >= min(int_domain)) & 
                           (target_domain <= max(int_domain)))[0]
    target_spectrum = target_spectrum[int_dom_ind]

    # normalize and scale target spectrum
    target_spectrum = target_spectrum / simpson(target_spectrum, int_domain)
    target_spectrum = scaler.transform(target_spectrum.reshape(1, -1))

    # ls coefficients and residual
    ls, res = nnls(int_spectra.T, target_spectrum.flatten(), maxiter=1000)

    # discard coefficients if res > 3
    if res > 3:
        ls = np.zeros(ls.shape[0])

    # discard coefficients which contribute less than 20%
    perc = np.where((ls / np.sum(ls)) <= .2, 0, ls)
    perc = np.nan_to_num(perc / np.sum(perc), nan=-1)
    
    return perc, labels

# return an list of array indicating the percentage of materials
# the percentage position in the array correspond to the label 
# in the returned label list 
# return -1 for every coefficient if unable to classify
def classify_sample(sample_file):
    sample_domain = np.loadtxt(sample_file)[:, 0]

    # load reference spectra
    labels, domains, spectra = load_reference()
    # interpolate to target domain
    int_spectra, int_domain = interpolate_reference(domains, 
                                                    spectra, 
                                                    sample_domain)
    
    # select sample spectra part corresponding to interpolation domain
    int_dom_ind = np.where((sample_domain >= min(int_domain)) &
                           (sample_domain <= max(int_domain)))[0]

    # load sample spectra e normalize it on the interpolation domain
    sample_spectra = np.loadtxt(sample_file).T[1:][:, int_dom_ind]
    norm_sample_spectra = simpson(sample_spectra, int_domain)
    sample_spectra = sample_spectra / norm_sample_spectra[:, None]

    # scale between 0 and 1 every spectral line of the reference spectra
    scaler = MinMaxScaler().fit(int_spectra)
    int_spectra = scaler.transform(int_spectra)
    sample_spectra = scaler.transform(sample_spectra)

    percs = []
    for spectrum in sample_spectra:
        # ls coefficients and residual
        ls, res = nnls(int_spectra.T, spectrum, maxiter=1000)

        # discard coefficients if res > 3
        if res > 3:
            ls = np.zeros(ls.shape[0])

        # discard coefficients which contribute less than 20%
        ls = np.where((ls / np.sum(ls)) <= .2, 0, ls)
        percs.append(np.nan_to_num(ls / np.sum(ls), nan=-1))

    return percs, labels
