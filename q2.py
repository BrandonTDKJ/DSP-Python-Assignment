

# Create a Python FIR filter class which implements an FIR filter which has
# a method of the form value dofilter(self,value) where both the value
# argument and return value are scalars and not vectors (!) so that it can
# be used in a realtime system. The constructor of the class takes the
# coefficients as its input:
import numpy as np
import pylab as pl
import scipy as sp
import scipy.signal as signal1
import Assignment2.q1 as q1
import matplotlib.pyplot as plt


def doFilterAdaptive(signal, noise, learningRate, fs):
    y = np.empty((len(signal)))
    f = FIRfilter(np.zeros(int(fs / 1)))
    for i in range(len(signal)):
        ref_noise = np.sin(2.0 * np.pi * noise / fs * i)
        cancellor = f.dofilter1(ref_noise)
        output_signal = signal[i] - cancellor
        f.lns(output_signal, learningRate)
        y[i] = output_signal
    plt.plot(y)
    plt.show()


class FIRfilter:
    def __init__(self,_coefficients,Frequency_Resolution = 1):
        self.ntaps = len(_coefficients)#ntaps is the number of coefficients
        self.coefficients = _coefficients#define the variable called coefficients to save the coeffient of FIR
        self.buffer = np.zeros(self.ntaps)#buffer is used to save the input value
        self.Frequency_Resolution = Frequency_Resolution


    def dofilter(self,v):
        results = np.zeros(len(v)+self.ntaps-1) #define the output

        for i in range(len(v)+self.ntaps-1):#the length of convolution operation is M+N-1
            for j in range(self.ntaps-1):# in order to acheive the convolution operation,we need to make the input signal move
                self.buffer[self.ntaps-1-j] = self.buffer[self.ntaps-2-j]#as time goes by move the past value x(n) to x(n-1)
            if(i<len(v)):
                self.buffer[0] = v[i] #assign the v to buffer[0] as the current input value
            else:#when all of the input are recorded,we need to add zero to make up the signal to achieve the conn
                self.buffer[0] = 0
            results[i] = np.inner(self.buffer,self.coefficients)#One for loop is to calculate the current value in the convolution result

        return results

    def dofilterLMS(self, v):#The filter is used to LMS because LMS is to update the filter
        # coefficients in each convolution process.

        for j in range(self.ntaps - 1):
            self.buffer[self.ntaps - 1 - j] = self.buffer[self.ntaps - 2 - j]  # as time goes by move the past value x(n) to x(n-1)
        self.buffer[0] = v  # assign the v to buffer[0] as the current input value

        return np.inner(self.buffer, self.coefficients)


    def q2Test(self): #which is the test unit for q2
        data = np.loadtxt('ECG_ugrad_matric_9.dat')
        data_fft = np.fft.fft(data)

        sampling_rate = 250
        cutoff_frequencies = [45, 55]
        cutoff_frequencies1 = [95, 110]
        Frequency_Resolution = 1
        coefficients = q1.bandstopDesign(sampling_rate, cutoff_frequencies, Frequency_Resolution)
        coefficients1 = q1.bandstopDesign(sampling_rate, cutoff_frequencies1, Frequency_Resolution)
        x = np.fft.ifft(coefficients)
        x = np.real(x)
        filter1 = FIRfilter(x)

        result = filter1.dofilter(data)
        result_fft = np.fft.fft(result)

        total_samples = np.size(data_fft)
        freq_step = sampling_rate / total_samples

        freq_domain = np.linspace(0, (total_samples - 1) * freq_step, total_samples)
        plt.subplot(2, 2, 1)
        plt.plot(data)
        plt.subplot(2, 2, 2)
        plt.plot(result)
        plt.subplot(2, 2, 3)
        plt.plot(freq_domain, data_fft)
        plt.subplot(2, 2, 4)
        plt.plot(freq_domain, result_fft)

        plt.show()

        return result

    def lns(self,error,learning_rate = 0.01):
        for j in range(self.ntaps):
            self.coefficients[j] =self.coefficients[j]+ error *learning_rate*self.buffer[j]

    def doFilterAdaptive(self,signal, noise, learningRate, fs):
        y = np.empty((len(signal))) #Y is used to record the error signal which is reference signal minus input signal
        f = FIRfilter(np.zeros(int(fs / 1))) #create the filter with random coeffientient at first with M length
        for i in range(len(signal)):
            ref_noise = np.sin(2.0 * np.pi * noise / fs * i) #create the input signal with 50Hz
            cancellor = f.dofilterLMS(ref_noise) #output signal
            output_signal = signal[i] - cancellor#error signal
            f.lns(output_signal, learningRate)#by using feedback loop to update coeffiencient of FIRfilter
            y[i] = output_signal
        plt.plot(y)
        plt.show()

    def heartbeat_detection(self,signal):

        sampling_rate = 250
        cutoff_frequencies = [45, 55]
        cutoff_frequencies1 = [95, 110]
        Frequency_Resolution = 1
        coefficients = q1.bandstopDesign(sampling_rate, cutoff_frequencies, Frequency_Resolution)
        coefficients1 = q1.bandstopDesign(sampling_rate, cutoff_frequencies1, Frequency_Resolution)
        x = np.fft.ifft(coefficients)
        x = np.real(x)
        filter1 = FIRfilter(x)
        result = filter1.dofilter(signal) #remove 50Hz
        plt.plot(result)
        plt.show()

        template =result[200:400] #create template
        plt.plot(template)
        plt.show()
        fir_coeff = template[::-1] #reverse time
        plt.plot(fir_coeff)
        plt.show()
        filter = FIRfilter(fir_coeff)
        results = filter.dofilter(result)
        results = results*results#square the result to improve signal/noise
        plt.plot(results)
        plt.show()




if __name__ == '__main__':
    v = np.array([1,2,3,4,5,6,7,8,9,12,1,5,6,7])
    coeffiency = np.array([1,1,1,0,0,0,1,1,1,0,0,2,4,5])
    c = signal1.convolve(v,coeffiency)
    print(c)
    filter1 = FIRfilter(coeffiency)
    result = filter1.dofilter(v)
    print(result)











