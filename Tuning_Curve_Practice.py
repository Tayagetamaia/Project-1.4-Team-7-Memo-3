import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Parameters
m = 5.3e-9  # mass in kg
k = 0.4708  # spring constant in N/m
b = 1.4272e-6  # damping coefficient in kg/s
F_max = 60e-9  # maximum force in N
omega_0 = np.sqrt(k / m)  # resonant frequency in rad/s
nu_0 = omega_0 / (2 * np.pi)  # resonant frequency in Hz

# Differential equation for a given frequency omega
def model_omega(X, t, omega):
    x, v = X
    F_t = F_max * np.sin(omega * t)
    dxdt = v
    dvdt = (F_t - b * v - k * x) / m
    return [dxdt, dvdt]

# Frequency array around the resonant frequency
frequencies = np.linspace(0.8 * omega_0, 1.2 * omega_0, 200)
amplitudes = []

# Initial condition
X0 = [0, 0]  # initial condition
t = np.linspace(0, 0.01, 10000)  # time from 0 to 10 ms with 10000 steps

# Calculate the amplitude for each frequency
for omega in frequencies:
    sol = odeint(model_omega, X0, t, args=(omega,))
    steady_state = sol[8000:, 0]  # take a part of the solution where the oscillation is stable
    amplitude = (np.max(steady_state) - np.min(steady_state)) / 2
    amplitudes.append(amplitude)

# Plot the tuning curve
plt.plot(frequencies, amplitudes)
plt.xlabel('Frequency (rad/s)')
plt.ylabel('Amplitude (m)')
plt.title('Tuning Curve of the Drive Mode')
plt.savefig('Tuning_Curve_practice', dpi=600)
plt.show()

# Determine the FWHM
max_amplitude = max(amplitudes)
half_max = max_amplitude / 2
indices = np.where(np.array(amplitudes) > half_max)[0]
fwhm = frequencies[indices[-1]] - frequencies[indices[0]]

# Determine the factor (ν0 / FWHM)
factor = nu_0 / fwhm

print(f"Maximale amplitude: {max_amplitude:.2e} m")
print(f"FWHM: {fwhm:.2e} rad/s")
print(f"Factor (ν0 / FWHM): {factor:.2e}")