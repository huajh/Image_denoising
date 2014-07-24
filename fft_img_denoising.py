# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 14:03:19 2014

@author: huajh
"""

import numpy as np
from matplotlib import pyplot as plt

def plot_spectrum(F, amplify=1000, ax=None):
    """Normalise, amplify and plot an amplitude spectrum."""

    # Note: the problem here is that we have a spectrum whose histogram is
    # *very* sharply peaked at small values.  To get a meaningful display, a
    # simple strategy to improve the display quality consists of simply
    # amplifying the values in the array and then clipping.

    # Compute the magnitude of the input F (call it mag).  Then, rescale mag by
    # amplify/maximum_of_mag.
    mag = abs(F) 
    mag *= amplify/mag.max() 
    
    # Next, clip all values larger than one to one.
    mag[mag > 1] = 1 

    if ax is None: ax = plt.gca()
    ax.imshow(mag, plt.cm.Blues)


if __name__ == '__main__':    
    
    #fname = 'moonlanding.png'
    fname ='lenaNoise.jpg'
    # your code here, you should get this image shape when done:
    # Image shape: (474, 630)
    im = plt.imread(fname).astype(float)
    print "Image shape: %s" % str(im.shape)
        
        
    # Assign the 2d FFT to `F`
    #...
    F = np.fft.fft2(im)
    # Define the fraction of coefficients (in each direction) we keep
    keep_fraction = 0.1
    
    # Call ff a copy of the original transform.  Numpy arrays have a copy
    # method for this purpose.
    # ...
    ff = F.copy()
    # Set r and c to be the number of rows and columns of the array.
    # ....
    r,c = ff.shape
    # Set to zero all rows with indices between r*keep_fraction and
    # r*(1-keep_fraction):
    #... 
    ff[r*keep_fraction:r*(1-keep_fraction)] = 0
    # Similarly with the columns:
    #... 
    ff[:,c*keep_fraction:c*(1-keep_fraction)] = 0
    # im_new =   # new image here, from inverse FFT of truncated data
    im_new = np.fft.ifft2(ff).real
    # Now, create the plots and display
    fig, ax = plt.subplots(2, 2, figsize=(10,7))
    
    ax[0,0].set_title('Original image')
    ax[0,0].imshow(im, plt.cm.gray)
    
    ax[0,1].set_title('Fourier transform')
    plot_spectrum(F, ax=ax[0,1])
    
    ax[1,1].set_title('Filtered Spectrum')
    plot_spectrum(ff, ax=ax[1,1])
    
    ax[1,0].set_title('Reconstructed Image')
    ax[1,0].imshow(im_new, plt.cm.gray);
    
    fig.savefig("fft_image_denoising.pdf");