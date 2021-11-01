import numpy as np
import matplotlib.pyplot as plt


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


    def lns(self,error,learning_rate = 0.01):
        for j in range(self.ntaps):
            self.coefficients[j] =self.coefficients[j]+ error *learning_rate*self.buffer[j]

    def doFilterAdaptive(self, signal, noise, learningRate):
        cancellor = self.dofilter(noise)
        error = signal- cancellor
        for j in range(self.ntaps):
            self.coefficients[j] =self.coefficients[j]+ error *learningRate*self.buffer[j]

        return error

if __name__ == '__main__':
    data = np.loadtxt('ECG_ugrad_matric_9.dat')
    fs = 250
    noiseFrequency = 50
    Frequency_Resolution = 1
    cutoff_frequencies = 5
    cutoff_frequencies1 = [45, 55]
    OutputAfterHighpassFilter = np.zeros(len(data))
    OutputAfterLMSFilter = np.zeros(len(data))
    OutputAfterBandStopFilter = np.zeros(len(data))

    #elminating baseline wander
    coefficients= highpassDesign(250, cutoff_frequencies, Frequency_Resolution)
    x = np.fft.ifft(coefficients)
    x = np.real(x)
    HighPassFilter = FIRfilter(x)
    for i in range(len(data)):
        OutputAfterHighpassFilter[i] =  HighPassFilter.dofilter(data[i])



    # Create BandStop filter to filter the signal with 50Hz frequency
    coefficients1 = bandstopDesign(250, cutoff_frequencies1, Frequency_Resolution)
    x = np.fft.ifft(coefficients1)
    x = np.real(x)
    filter1 = FIRfilter(x)

    for i in range(len(data)):
        OutputAfterBandStopFilter[i] = filter1.dofilter(OutputAfterHighpassFilter[i])

    #create LMS filter
    AdaptiveFilter = FIRfilter(np.zeros(int(fs / 1)))
    for i in range(len(OutputAfterHighpassFilter)):
        noise = np.sin(2.0 * np.pi * noiseFrequency / fs * i)
        OutputAfterLMSFilter[i] = AdaptiveFilter.doFilterAdaptive(OutputAfterHighpassFilter[i],noise,0.01)

    plt.subplot(2, 2, 1)
    plt.title("original singal")
    plt.plot(data)

    plt.subplot(2, 2, 2)
    plt.title("Filter by LMS")
    plt.plot(OutputAfterLMSFilter)

    plt.subplot(2, 2, 3)
    plt.title("Filter by Normal filter")
    plt.plot(OutputAfterBandStopFilter)

    plt.show()


