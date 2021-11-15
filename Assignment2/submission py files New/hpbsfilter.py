import numpy as np
from matplotlib import pyplot as plt
from firfilter import highpassDesign, bandstopDesign,FIRfilter
if __name__ == '__main__':
    data = np.loadtxt('ecg.dat')
    Frequency_Resolution = 1
    OutputAfterHighpassFilter = np.zeros(len(data))
    OutputAfterBandStopFilter = np.zeros(len(data))
    sample_rate = 250


    # create bandstop filter
    cutoff_frequencies1 = [49, 51]
    coefficients1 = bandstopDesign(250, cutoff_frequencies1, Frequency_Resolution)
    filter1 = FIRfilter(coefficients1)

    #create high pass filter
    cutoff_frequencies2 = 2
    coefficients2 = highpassDesign(250, cutoff_frequencies2, Frequency_Resolution)
    filter2 = FIRfilter(coefficients2)

    #Processing of eliminating baseline wander
    for i in range(len(data)):
        OutputAfterHighpassFilter[i]= filter1.dofilter(data[i])


    #eliminating 50Hz
    for i in range(len(data)):
        OutputAfterBandStopFilter[i] = filter2.dofilter(OutputAfterHighpassFilter[i])

    amplitude = np.array(OutputAfterBandStopFilter)

    # calculating the total number of samples
    total_samples = np.size(OutputAfterBandStopFilter)

    # calculating the time step between each sample
    time_step = 1 / sample_rate

    # calculating the time domain for the signal
    time_domain = np.linspace(0, (total_samples - 1) * time_step, total_samples)

    # calculating the frequency step size for the signal
    freq_step = sample_rate / total_samples

    # calculating the frequency domain for the signal
    freq_domain = np.linspace(0, (total_samples - 1) * freq_step, total_samples)
    freq_domain_plt = freq_domain[:int(total_samples / 2) + 1]

    # calculating the frequency response of the signal
    freq_mag = np.fft.fft(OutputAfterBandStopFilter)
    freq_mag_abs = np.abs(freq_mag) / total_samples
    freq_mag_abs_plt = 2 * freq_mag_abs[:int(total_samples / 2) + 1]
    freq_mag_dB = 20 * np.log10(freq_mag_abs_plt)

    plt.figure(4)
    plt.title("Spectrum of output after eliminating 50Hz")
    plt.xlabel("frequency")
    plt.ylabel("dB")
    plt.plot(freq_domain_plt, freq_mag_abs_plt)

    t = np.arange(0,5000)
    t = t/250
    plt.figure(1)
    plt.title("original signal")
    plt.plot(t,data)

    plt.figure(2)
    plt.title("output after eliminating baseline wander")
    plt.plot(t,OutputAfterHighpassFilter)

    plt.figure(3)
    plt.title("output after eliminating 50Hz")
    plt.plot(t,OutputAfterBandStopFilter)
    plt.show()

