#importing the necessary modules
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile


#maximum value of a 16 bit PCM
max_val = 32768
cutoff = 1000

#loading the sample rate of the signal along with the data points
sample_rate, data = wavfile.read('DSP_sentence_recording_removed_noise.wav')

#loading the data of the sample into an array
amplitude = np.array(data) 

# calculating the total number of samples
total_samples = np.size(amplitude) 

#calculating the time step between each sample
time_step = 1 / sample_rate

#calculating the time domain for the signal
time_domain = np.linspace(0, (total_samples - 1) * time_step, total_samples)

#calculating the frequency step size for the signal
freq_step = sample_rate / total_samples

#calculating the frequency domain for the signal
freq_domain = np.linspace(0, (total_samples - 1) * freq_step, total_samples)
freq_domain_plt = freq_domain[:int(total_samples / 2) + 1]

#calculating the frequency response of the signal
freq_mag = np.fft.fft(amplitude)
freq_mag_abs = np.abs(freq_mag ) / total_samples
freq_mag_abs_plt = 2 * freq_mag_abs[:int(total_samples / 2) + 1]
freq_mag_dB = 20 * np.log10(freq_mag_abs_plt)

#calculate the number of samples within our desired range of frequencies
 
freq_mag_rec = np.copy(freq_mag)

#signal reconstructed using ifft
amp_rec = np.fft.ifft(freq_mag_rec)

#graphing the plot of the signal
plt.subplot(2,2,1)
plt.plot(time_domain, amplitude / max_val)

#graphing the frequency response of the signal
plt.subplot(2,2,2)
plt.plot(freq_domain_plt, freq_mag_abs_plt)

#graphing the frequency response of the signal in logarithmic scale
plt.subplot(2,2,3)
plt.plot(freq_domain_plt, freq_mag_dB)

plt.xscale('log')
plt.yscale('log')

#graphing the plot of the signal after reconstruction
plt.subplot(2,2,4)
plt.plot(time_domain, amp_rec/max_val)


wavfile.write("DSP_sentence_recording_output.wav", 44100, np.float32(amp_rec/max_val))
#display all of the graphs for the signal
plt.show()



