import pickle
import matplotlib.pyplot as plt
import numpy as np
import Constants
import StepStrideFrequencyModuleConstants

if __name__ == "__main__":
    with open("p_spectrum.pkl","r") as f:
        data = pickle.load(f)
        y = data[('6b104542-0b0a-4421-a490-568f66c7ebc4', 'Immediately before Parkinson medication')]
        x = np.linspace(0,Constants.sampling_frequency/2,StepStrideFrequencyModuleConstants.sample_points)
        plt.plot(x,y)
        plt.show()

