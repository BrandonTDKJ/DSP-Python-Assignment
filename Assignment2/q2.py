import numpy as np
import matplotlib.pyplot as plt

# Filter your ECG with the above FIR filter class using the coefficients from
# 1. by removing the 50Hz interference and the baseline wander with the
# highpass. Decide which cutoff frequencies are needed and provide
# explanations by referring to the spectra and/or fundamental frequencies
# of the ECG. Simulate realtime processing by feeding the ECG sample by
# sample into your FIR filter class. Make sure that the ECG looks intact and
# that it is not distorted (PQRST intact). Provide appropriate plots

def highpassDesign(sampling_rate,cutoff_frequencies,Frequency_Resolution):
    fs = sampling_rate
    M = int(fs/Frequency_Resolution)
    k = int(cutoff_frequencies/fs *M)
    X = np.ones(M)
    X[0:k] = 0
    X[M - k:M - 1] = 0
    return X




def bandstopDesign(sampling_rate,cutoff_frequencies,Frequency_Resolution):
        fs = sampling_rate
        M = int(fs / Frequency_Resolution)
        k1 = int(cutoff_frequencies[0]/fs * M)
        k2 = int(cutoff_frequencies[1] / fs * M)
        X = np.ones(M)
        X[k1:k2+1] = 0
        X[M-k2:M-k1+1] = 0
        X = np.real(X)
        return X


class FIRfilter:
    def __init__(self,_coefficients,Frequency_Resolution = 1):
        self.ntaps = len(_coefficients)  # ntaps is the number of coefficients
        self.coefficients = _coefficients  # define the variable called coefficients to save the coeffient of FIR
        self.buffer = np.zeros(self.ntaps)  # buffer is used to save the input value
        self.Frequency_Resolution = Frequency_Resolution
    def dofilter(self,v):
        for j in range(self.ntaps - 1):
            self.buffer[self.ntaps - 1 - j] = self.buffer[self.ntaps - 2 - j]  # as time goes by move the past value x(n) to x(n-1)
        self.buffer[0] = v  # assign the v to buffer[0] as the current input value

        return np.inner(self.buffer, self.coefficients)

if __name__ == '__main__':
    data = np.loadtxt('ECG_ugrad_matric_9.dat')
    Frequency_Resolution = 1
    OutputAfterHighpassFilter = np.zeros(len(data))
    OutputAfterBandStopFilter = np.zeros(len(data))

    # create bandstop filter
    cutoff_frequencies1 = [45, 55]
    coefficients1 = bandstopDesign(250, cutoff_frequencies1, Frequency_Resolution)
    x = np.fft.ifft(coefficients1)
    x = np.real(x)
    filter1 = FIRfilter(x)

    #create high pass filter
    cutoff_frequencies2 = 5
    coefficients2 = highpassDesign(250, cutoff_frequencies2, Frequency_Resolution)
    x = np.fft.ifft(coefficients2)
    x = np.real(x)
    filter2 = FIRfilter(x)

    #Processing of eliminating baseline wander
    for i in range(len(data)):
        OutputAfterHighpassFilter[i]= filter1.dofilter(data[i])

    for i in range(len(data)):
        OutputAfterBandStopFilter[i] = filter2.dofilter(OutputAfterHighpassFilter[i])


    plt.subplot(2, 2, 1)
    plt.title("original singal")
    plt.plot(data)

    plt.subplot(2, 2, 2)
    plt.title("output after eliminating baseline wander")
    plt.plot(OutputAfterHighpassFilter)

    plt.subplot(2, 2, 3)
    plt.title("output after eliminating 50Hz")
    plt.plot( OutputAfterBandStopFilter)
    plt.show()


