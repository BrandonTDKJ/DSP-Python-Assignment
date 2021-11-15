import numpy as np
from matplotlib import pyplot as plt
from firfilter import highpassDesign,bandstopDesign, FIRfilter
if __name__ == '__main__':
    data = np.loadtxt('ecg.dat')
    fs = 250
    sample_rate = 250
    noiseFrequency = 50
    Frequency_Resolution = 1
    cutoff_frequencies = 2
    OutputAfterHighpassFilter = np.zeros(len(data))
    OutputAfterLMSFilter = np.zeros(len(data))
    t = np.arange(0, 5000)
    t = t / 250

    #elminating baseline wander
    coefficients= highpassDesign(250, cutoff_frequencies, Frequency_Resolution)
    HighPassFilter = FIRfilter(coefficients)
    for i in range(len(data)):
        OutputAfterHighpassFilter[i] =  HighPassFilter.dofilter(data[i])

    #create LMS filter
    AdaptiveFilter = FIRfilter(np.zeros(int(fs / 1)))
    for i in range(len(OutputAfterHighpassFilter)):
        noise = np.sin(2.0 * np.pi * noiseFrequency / fs * i)
        OutputAfterLMSFilter[i] = AdaptiveFilter.doFilterAdaptive(OutputAfterHighpassFilter[i],noise,0.01)


    plt.figure(1)
    plt.title("original singal")
    plt.plot(t,data)

    plt.figure(2)
    plt.title("Filter by Adaptive LMS")
    plt.plot(t,OutputAfterLMSFilter)

    plt.show()


