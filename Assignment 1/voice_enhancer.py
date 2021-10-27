#importing the necessary modules
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile


#maximum value of a 16 bit PCM
max_val = 32767
cutoff = 1000

#loading the sample rate of the signal along with the data points
sample_rate, data = wavfile.read('original.wav')

#loading the data of the sample into an array
amplitude = np.array(data)

amplitude_norm = amplitude / max_val

# calculating the total number of samples
total_samples = np.size(amplitude) 
print(total_samples)

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

pos_x = int(200 / freq_step)
pos_y = int(1000 / freq_step)

freq_mag = np.fft.fft(amplitude_norm)
freq_mag_norm = freq_mag /total_samples
freq_mag_abs = np.abs(freq_mag_norm)
freq_mag_abs_plt = 2 * freq_mag_abs[:int(total_samples / 2) + 1]

freq_mag_dB = 20 * np.log10(freq_mag_abs_plt)

#calculate the number of samples within our desired range of frequencies

freq_mag_rec = np.copy(freq_mag)
freq_mag_rec[pos_x:pos_y] = freq_mag_rec[pos_x:pos_y] * 1.5
freq_mag_rec[total_samples - pos_y: total_samples - pos_x] = freq_mag_rec[total_samples - pos_y: total_samples - pos_x] * 1.5

amp_rec = np.fft.ifft(freq_mag_rec)


#graphing the plot of the signal
plt.figure("Plot of Normalized amplitude vs. Time")
plt.plot(time_domain, amplitude_norm)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude Normalized")
plt.grid()

#graphing the frequency response of the signal in logarithmic scale
plt.figure("Plot of frequency spectrum with logarhithmic scales")
plt.plot(freq_domain_plt, freq_mag_dB)
plt.xscale('log')
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (dB)")
plt.grid()

wavfile.write("enhanced.wav", 44100, np.float32(amp_rec))
#display all of the graphs for the signal

plt.show()
