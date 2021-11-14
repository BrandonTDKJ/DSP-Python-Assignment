import numpy as np
from matplotlib import pyplot as plt
from firfilter import highpassDesign, bandstopDesign, FIRfilter


def createWavelet():
    t = np.arange(-0.4, 0.4, 1 / 250)
    w = 250
    y = np.sin(w * (t - 0)) / (w * (t - 0))

    return y


if __name__ == '__main__':
    y1 = createWavelet()
    data = np.loadtxt('ecg.dat')
    PeakTimeForECG = np.zeros(200)
    # data_max = np.min(data)
    # data = data/data_max
    # plt.plot(data)
    # plt.show()
    PeakNum = 0
    Frequency_Resolution = 1
    OutputAfterHighpassFilter = np.zeros(len(data))
    OutputAfterBandStopFilter = np.zeros(len(data))
    res1 = np.zeros(len(data))
    res2 = np.zeros(len(data))
    t = np.arange(0, 5000)
    t = t / 250
    res3 = []
    # create bandstop filter
    cutoff_frequencies1 = [49, 51]
    coefficients1 = bandstopDesign(250, cutoff_frequencies1, Frequency_Resolution)
    filter1 = FIRfilter(coefficients1)

    # create high pass filter
    cutoff_frequencies2 = 2
    coefficients2 = highpassDesign(250, cutoff_frequencies2, Frequency_Resolution)
    filter2 = FIRfilter(coefficients2)

    # Processing of eliminating baseline wander
    for i in range(len(data)):
        OutputAfterHighpassFilter[i] = filter1.dofilter(data[i])

    for i in range(len(data)):
        OutputAfterBandStopFilter[i] = filter2.dofilter(OutputAfterHighpassFilter[i])

    template1 = OutputAfterBandStopFilter[3490:3690]  # create template
    template2 = createWavelet()
    fir_coeff1 = template1[::-1]  # reverse time
    fir_coeff2 = template2[::-1]

    template_t = np.arange(-100, 100)
    template_t = template_t / 250

    plt.figure(1)

    plt.subplot(2, 1, 1)
    plt.xlabel("time")
    plt.ylabel("Amplitude")
    plt.title("Template")
    plt.plot(template_t, fir_coeff1)

    plt.subplot(2, 1, 2)
    plt.xlabel("time")
    plt.ylabel("Amplitude")
    plt.title("Wavelet")
    plt.plot(template_t, fir_coeff2)

    filter1 = FIRfilter(fir_coeff1)
    filter2 = FIRfilter(fir_coeff2)

    for i in range(len(data)):
        res2[i] = filter2.dofilter(OutputAfterBandStopFilter[i])

        res2[i] = res2[i] * res2[i]

    plt.figure(2)
    plt.xlabel("time")
    plt.ylabel("Amplitude")
    plt.title("Output with R peaks detected")
    plt.plot(t, res2)

    for i in range(len(data)):
        if res2[i] <= 1.0e-5:
            res2[i] = 0
        else:
            PeakTimeForECG[PeakNum] = i
            PeakNum += 1

    PeakTimeForECG = PeakTimeForECG[:PeakNum]
    PeakTimeForECG = PeakTimeForECG / 250

    res3.append(PeakTimeForECG[0])
    for i in range(1, len(PeakTimeForECG)):
        if PeakTimeForECG[i] - PeakTimeForECG[i - 1] > 0.3:
            res3.append(PeakTimeForECG[i])

    InverseInterval = []
    y_output = []
    x_output = []
    for i in range(1, len(res3)):
        InverseInterval.append((1 / (res3[i] - res3[i - 1])))
    # FirstValue = InverseInterval[0]
    # InverseInterval.insert(0,FirstValue)
    for i in range(len(InverseInterval)):
        y_output.append(InverseInterval[i])
        y_output.append(InverseInterval[i])
    x_output.append(0)
    for i in range(1, len(y_output), 2):
        x_output.append(res3[int(i / 2)])
        x_output.append(res3[int(i / 2)])

    x_output = x_output[:-1]

    t1 = np.arange(0, 20)
    plt.figure(3)
    plt.xlabel("time")
    plt.ylabel("Amplitude")
    plt.title("Output  after thresholding")
    plt.plot(t, res2)

    plt.figure(4)
    plt.plot(x_output, y_output)
    plt.xlabel("time(s)")
    plt.ylabel("(/s)")
    plt.title("Momentary Heartrate")

    plt.show()
