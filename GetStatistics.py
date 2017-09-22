import Utilities as utilities
import StepStrideFrequencyModuleConstants
import pickle
import matplotlib.pyplot as plt
import numpy as np
import math
import sys

def main():
    X, recordIds = utilities.load_full_data()
    print "Computing Fourier transform and power spectrum..."
    X1 = []
    Y1 = []
    Z1 = []
    for x in X:
        (fx, x1) = utilities.fourier_transform(x["x"])
        (fy, y1) = utilities.fourier_transform(x["y"])
        (fz, z1) = utilities.fourier_transform(x["z"])
        # a[0] = 0
        x1 = utilities.remove_noise(x1, 0.5)
        y1 = utilities.remove_noise(y1, 0.5)
        z1 = utilities.remove_noise(z1, 0.5)
        # interpolate
        (fx, x1) = utilities.interpolate_data(fx, x1)
        (fy, y1) = utilities.interpolate_data(fy, y1)
        (fz, z1) = utilities.interpolate_data(fz, z1)
        x1 = utilities.squared_signal(x1)
        y1 = utilities.squared_signal(y1)
        z1 = utilities.squared_signal(z1)
        X1.append(x1)
        Y1.append(y1)
        Z1.append(z1)

    print "Estimating the quantities in X-axis..."
    half_energies = []
    amplitudes = []
    frequencies = []
    j = 1
    for z in X1:
        f = StepStrideFrequencyModuleConstants.F
        start_index = next(x[0] for x in enumerate(f) if x[1] > 0.5)
        end_index = next(x[0] for x in enumerate(f) if x[1] > 2.5)
        print start_index
        print end_index
        with open("z.pkl", "w") as file:
            pickle.dump(z, file)
        diff = z - max(z)
        if np.any(diff):
            indices = utilities.get_peaks_between(z, start_index=start_index, end_index=end_index, thres=0.3,
                                                  min_dist=1)
            print "Done till this point."
            for i in range(len(indices)):
                left_idx = indices[i - 1] if i - 1 >= 0 else 0
                right_idx = indices[i + 1] if i + 1 < len(indices) else len(f) - 1
                half_energies.append(utilities.get_half_energy(z, f, indices[i], left_idx, right_idx))
                a = np.sqrt(z)
                amplitudes.append(a[indices[i]])
                frequencies.append(f[indices[i]])
        print "Done " + str(j) + "!"
        j = j + 1

    with open("half-energies_x.pkl","w") as f:
        pickle.dump(half_energies,f)

    with open("amplitudes_x.pkl","w") as f:
        pickle.dump(amplitudes,f)

    with open("frequencies_x.pkl","w") as f:
        pickle.dump(frequencies,f)

    print "----Energy----"
    print np.mean(np.array(half_energies))
    print np.std(np.array(half_energies))
    print "--------------"

    print "----Amp-------"
    print np.mean(np.array(amplitudes))
    print np.std(np.array(amplitudes))
    print "--------------"

    print "-----Freq-----"
    print np.mean(np.array(frequencies))
    print np.std(np.array(frequencies))
    print "--------------"

    print "Estimating the quantities in Y-axis..."
    half_energies = []
    amplitudes = []
    frequencies = []
    j = 1
    for z in Y1:
        f = StepStrideFrequencyModuleConstants.F
        start_index = next(x[0] for x in enumerate(f) if x[1] > 0.5)
        end_index = next(x[0] for x in enumerate(f) if x[1] > 2.5)
        print start_index
        print end_index
        with open("z.pkl", "w") as file:
            pickle.dump(z, file)
        diff = z - max(z)
        if np.any(diff):
            indices = utilities.get_peaks_between(z, start_index=start_index, end_index=end_index, thres=0.3,
                                                  min_dist=1)
            print "Done till this point."
            for i in range(len(indices)):
                left_idx = indices[i - 1] if i - 1 >= 0 else 0
                right_idx = indices[i + 1] if i + 1 < len(indices) else len(f) - 1
                half_energies.append(utilities.get_half_energy(z, f, indices[i], left_idx, right_idx))
                a = np.sqrt(z)
                amplitudes.append(a[indices[i]])
                frequencies.append(f[indices[i]])
        print "Done " + str(j) + "!"
        j = j + 1

    with open("half-energies_y.pkl","w") as f:
        pickle.dump(half_energies,f)

    with open("amplitudes_y.pkl","w") as f:
        pickle.dump(amplitudes,f)

    with open("frequencies_y.pkl","w") as f:
        pickle.dump(frequencies,f)

    print "----Energy----"
    print np.mean(np.array(half_energies))
    print np.std(np.array(half_energies))
    print "--------------"

    print "----Amp-------"
    print np.mean(np.array(amplitudes))
    print np.std(np.array(amplitudes))
    print "--------------"

    print "-----Freq-----"
    print np.mean(np.array(frequencies))
    print np.std(np.array(frequencies))
    print "--------------"

    print "Estimating the quantities in Z-axis..."
    half_energies = []
    amplitudes = []
    frequencies = []
    j = 1
    for z in Z1:
        f = StepStrideFrequencyModuleConstants.F
        start_index = next(x[0] for x in enumerate(f) if x[1] > 0.5)
        end_index = next(x[0] for x in enumerate(f) if x[1] > 2.5)
        print start_index
        print end_index
        with open("z.pkl","w") as file:
            pickle.dump(z,file)
        diff = z - max(z)
        if np.any(diff):
            indices = utilities.get_peaks_between(z,start_index=start_index,end_index=end_index,thres=0.3,min_dist=1)
            print "Done till this point."
            for i in range(len(indices)):
                left_idx = indices[i - 1] if i - 1 >= 0 else 0
                right_idx = indices[i + 1] if i + 1 < len(indices) else len(f) - 1
                half_energies.append(utilities.get_half_energy(z, f, indices[i], left_idx, right_idx))
                a = np.sqrt(z)
                amplitudes.append(a[indices[i]])
                frequencies.append(f[indices[i]])
        print "Done " + str(j) + "!"
        j = j + 1

    with open("half-energies_z.pkl","w") as f:
        pickle.dump(half_energies,f)

    with open("amplitudes_z.pkl","w") as f:
        pickle.dump(amplitudes,f)

    with open("frequencies_z.pkl","w") as f:
        pickle.dump(frequencies,f)

    print "----Energy----"
    print np.mean(np.array(half_energies))
    print np.std(np.array(half_energies))
    print "--------------"

    print "----Amp-------"
    print np.mean(np.array(amplitudes))
    print np.std(np.array(amplitudes))
    print "--------------"

    print "-----Freq-----"
    print np.mean(np.array(frequencies))
    print np.std(np.array(frequencies))
    print "--------------"

    sys.exit()
    plt.subplot(3,1,1)
    plt.hist(np.array(half_energies), bins='auto')
    plt.title("Energy distribution")
    plt.xlim((0,1000))

    plt.subplot(3,1,2)
    plt.hist(np.array(amplitudes),bins='auto')
    plt.title("Amplitudes")

    plt.subplot(3,1,3)
    plt.hist(np.array(frequencies),bins='auto')
    plt.title("Frequencies")
    plt.show()

    with open('half-energies.pkl','w') as f:
        pickle.dump(half_energies,f)

if __name__ == "__main__":
    main()