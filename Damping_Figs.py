import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Parameters
m = 5.3e-9  # mass in kg
k = 0.4708  # spring constant in N/m
b = 1.4272e-6  # damping coefficient in kg/s
F_max = 60e-9  # maximum force in N
omega_0 = np.sqrt(k / m)  # resonant frequency in rad/s

# Differential equation for a given frequency omega
def model_omega(X, t, omega, b):
    x, v = X
    F_t = F_max * np.sin(omega * t)
    dxdt = v
    dvdt = (F_t - b * v - k * x) / m
    return [dxdt, dvdt]

# Function to simulate and calculate the Q-factor and FWHM for a given damping coefficient
def simulate_system(b):
    frequencies = np.linspace(0.8 * omega_0, 1.2 * omega_0, 200)
    amplitudes = []

    X0 = [0, 0]  # initial condition
    t = np.linspace(0, 0.01, 10000)  # time from 0 to 10 ms with 10000 steps

    # Calculate the amplitude for each frequency
    for omega in frequencies:
        sol = odeint(model_omega, X0, t, args=(omega, b))
        steady_state = sol[8000:, 0]  # take a part of the solution where the oscillation is stable
        amplitude = (np.max(steady_state) - np.min(steady_state)) / 2
        amplitudes.append(amplitude)

    max_amplitude = max(amplitudes)
    half_max = max_amplitude / 2
    indices = np.where(np.array(amplitudes) > half_max)[0]
    fwhm = frequencies[indices[-1]] - frequencies[indices[0]]
    Q_factor = omega_0 / fwhm

    return max_amplitude, Q_factor, fwhm


xc = np.linspace(-0.2, 0.2, 500)
results = [[], [], [], []]  # damping_coefficient, amplitude, Q_factor, FWHM

for i in range(len(xc)):
    factor = xc[i]
    b_varied = b * (1 + factor)
    amplitude, Q_factor, fwhm = simulate_system(b_varied)
    results[0].append(b_varied)  # damping_coefficient
    results[1].append(amplitude)  # amplitude
    results[2].append(Q_factor)  # Q_factor
    results[3].append(fwhm)  # FWHM
    

fig, ax1 = plt.subplots()

# Plot the amplitude on the first y-axis
ax1.plot(xc, results[1], 'C1')
ax1.set_xlabel('Variation in damping coefficient (%)')
ax1.set_ylabel('Amplitude (m)', color='C1')
ax1.tick_params('y')

# Create a second y-axis that shares the same x-axis
ax2 = ax1.twinx()

# Plot the damping coefficient on the second y-axis
ax2.plot(xc, results[0], 'C0')  # blue solid line
ax2.set_ylabel('Damping Coefficient', color='C0')
ax2.tick_params('y')

plt.savefig(('Amplitude.png'), dpi=600)
plt.clf()

plt.plot(xc, results[2])
plt.xlabel('Variation in damping coefficient (%)')
plt.ylabel('Q-factor')
plt.savefig(('Q-factor.png'), dpi=600)
plt.clf()

plt.plot(xc, results[3])
plt.xlabel('Variation in damping coefficient (%)')
plt.ylabel('FWHM (rad/s)')
plt.savefig(('FWHM.png'), dpi=600)
plt.clf()