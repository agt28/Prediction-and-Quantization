import matplotlib.pyplot as plt
import scipy.signal as sps
import numpy as np
import math

'''
Prediction model

The model functions as the signal processing system that features an IIR filter
that can predict the squence given.


'''
class PredictionModel(object):

    def __init__(self, x_n, coefficients, optimal_coefficients):
        self.x_n = np.array(x_n) # x[n]
        self.x_hat = 0 # x^[n]
        self.error = 0 # e[n]
        self.coefficients = np.array(coefficients) # {a}
        self.optimal_coefficients = np.array(optimal_coefficients) # {a*}
        self.find_energy = lambda x: np.dot(x,x)
        self.sgn = lambda x: 1 / (1 + math.exp(-x))

    def setSequence(self, x, coefficients):
        self.x_n = x
        self.coefficients = coefficients

    def getPrediction(self):
        """
        Prediction using the second approach
        """
        numerator = self.coefficients
        denominator = [1]
        self.error = sps.lfilter(numerator, denominator, self.x_n) # E(z) = A(z)X(z)
        self.x_hat = self.x_n - self.error # x_hat[n] = x[n] – e[n]

    def printPrediction(self):
        print("x[n] Values:  ", self.x_n)
        print("X^ Values:    ", self.x_hat)
        print("Error Values: ", self.error)

        XE = self.find_energy(self.x_n)
        XHE = self.find_energy(self.x_hat)
        prediction_gain = 10*np.log10(XE / XHE)
        print("Prediction Gain: ", prediction_gain)

        plt.style.use('seaborn-white')

        plt.figure(figsize=[15,25])
        plt.subplot(4,1,1)
        plt.plot(self.x_n, color='#264653', label='Sequence (E = '+ str(XE) + 'J )')
        plt.plot(self.x_hat, color='#2A9D8F', label='Prediction (E = ' + str(XHE) + 'J )')
        plt.plot(self.error, color='#E76F51', label='Error')
        plt.title("Time Squence")
        plt.xlabel('Time[s]')
        plt.ylabel('Magnitude')
        plt.grid()
        plt.legend(loc='upper right')

    def printCoefficients(self):
        plt.figure(figsize=[15,25])
        plt.subplot(4,1,1)
        plt.plot(self.coefficients, color='#264653', label='Manuel {a}')
        plt.title("Manuel coefficients")
        plt.xlabel('Time[s]')
        plt.ylabel('Magnitude')
        plt.grid()
        plt.legend(loc='upper right')

        plt.figure(figsize=[15,25])
        plt.subplot(4,1,1)
        plt.plot(self.optimal_coefficients, color='#2A9D8F', label='Optimized {a*}')
        plt.title("Optimized Coefficients")
        plt.xlabel('Time[s]')
        plt.ylabel('Magnitude')
        plt.grid()
        plt.legend(loc='upper right')

        denominator = self.coefficients
        numerator = [1]
        z, p, k = sps.tf2zpk(numerator, denominator)

        theta = np.linspace(-np.pi,np.pi,201)
        fig = plt.figure(figsize=[25,25])
        ax1 = fig.add_subplot(4,1,1)
        ax1.set_title("Unit Circle - Manuel Coefficients")
        ax1.grid()
        ax1.plot(np.cos(theta),np.sin(theta))
        ax1.scatter(np.real(z),np.imag(z), color='red')
        ax1.scatter(np.real(p),np.imag(p),color='green',marker='x')
        ax1.set_aspect('equal')

        denominator= self.optimal_coefficients
        z, p, k = sps.tf2zpk(numerator, denominator)

        fig = plt.figure(figsize=[25,25])
        ax1 = fig.add_subplot(4,1,1)
        ax1.set_title("Unit Circle - Optimized Coefficients")
        plt.grid()
        ax1.plot(np.cos(theta),np.sin(theta))
        ax1.scatter(np.real(z),np.imag(z), color='red')
        ax1.scatter(np.real(p),np.imag(p),color='green',marker='x')
        ax1.set_aspect('equal')

    def coeQuantizer2bin(self, coefficients):
        newCoe = []

        for c in coefficients:
            if c < 0:
                newCoe.append(-1)
            elif 0 < c and c < 0.5:
                newCoe.append(0.1 * self.sgn(c))
            elif 0.75 < c and c < 1.0:
                newCoe.append(0.85 * self.sgn(c))
            else:
                newCoe.append(c)

        return newCoe

    def plotCoefficients(self, coefficients):
        plt.figure(figsize=[15,25])
        plt.subplot(4,1,1)
        plt.plot(coefficients, color='#264653', label='Manuel {a}')
        plt.title("Coefficients Values")
        plt.xlabel('Time[s]')
        plt.ylabel('Magnitude')
        plt.grid()
        plt.legend(loc='upper right')

        denominator = coefficients
        numerator = [1]
        z, p, k = sps.tf2zpk(numerator, denominator)

        theta = np.linspace(-np.pi,np.pi,201)
        fig = plt.figure(figsize=[25,25])
        ax1 = fig.add_subplot(4,1,1)
        ax1.set_title("Unit Circle - Coefficients")
        ax1.grid()
        ax1.plot(np.cos(theta),np.sin(theta))
        ax1.scatter(np.real(z),np.imag(z), color='red')
        ax1.scatter(np.real(p),np.imag(p),color='green',marker='x')
        ax1.set_aspect('equal')



if __name__ == '__main__':
    """
    Visualize random data
    """
    x_n = [1.0, -1.0, 0.0, 2.0, 1.0, -1.0, -2.0, 0.0, 1.0, 0.0,] # Input x[n]
    coefficients = [0.8570, -0.2500, 1.0000] # {a}
    model = PredictionModel(x_n, coefficients)



    """
    Visualize Random Dataloader
    """
    # showrandom_batch()
