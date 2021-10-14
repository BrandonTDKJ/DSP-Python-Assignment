import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile


max_val = 32768
cutoff = 1000



def detector(FilePath):
    # loading the sample rate of the signal along with the data points
    sample_rate, data = wavfile.read(FilePath)

    # loading the data of the sample into an array
    amplitude = np.array(data)
    amplitude_norm = amplitude / max_val

    print(np.max(amplitude))
    print(np.max(amplitude_norm))

    # calculating the total number of samples
    total_samples = np.size(amplitude)

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
    freq_mag = np.fft.fft(amplitude_norm) / total_samples
    freq_mag_abs = np.abs(freq_mag)
    freq_mag_abs_plt = 2 * freq_mag_abs[:int(total_samples / 2) + 1]
    freq_mag_dB = 20 * np.log10(freq_mag_abs_plt)

    plt.subplot(2, 2, 1)
    plt.plot(freq_domain_plt, freq_mag_abs_plt)
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.plot(freq_domain_plt, freq_mag_dB)
    left = int(100 / freq_step)
    right = int(150 / freq_step)

    TheMaxAmplitude= np.max(freq_mag_abs_plt[left:right])
    print(TheMaxAmplitude)
    if(TheMaxAmplitude>0.028):
        return "a"
    if(TheMaxAmplitude>0.018 and TheMaxAmplitude<0.025):
        return"i"
    else:
        return "e"

if __name__ == '__main__':
    FilePath = 'Resources/DSP_sound_i.wav'
    ReturnValue = detector(FilePath)
    print("The vowel is "+ReturnValue+"  according to the vowel detector")