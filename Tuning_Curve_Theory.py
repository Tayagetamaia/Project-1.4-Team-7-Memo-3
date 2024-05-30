import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Parameters
m = 5.3e-9  # massa in kg
k = 0.4708  # veerconstante in N/m
b = 1.4272e-6  # dempingsfactor in kg/s
F_max = 60e-9  # amplitude van de oscillerende kracht in N
omega_0 = np.sqrt(k / m)  # resonantiefrequentie in rad/s

# Differentiaalvergelijking voor een gegeven frequentie omega
def model_omega(X, t, omega):
    x, v = X
    F_t = F_max * np.sin(omega * t)
    dxdt = v
    dvdt = (F_t - b * v - k * x) / m
    return [dxdt, dvdt]

# Frequentiearray rond de resonantiefrequentie
frequencies = np.linspace(0.8 * omega_0, 1.2 * omega_0, 200)
amplitudes = []


X0 = [0, 0]  # initiële conditie
t = np.linspace(0, 0.06, 10000)  # tijd van 0 tot 10 ms met 10000 stappen

# Bereken de amplitude voor elke frequentie
for omega in frequencies:
    sol = odeint(model_omega, X0, t, args=(omega,))
    steady_state = sol[8000:, 0]  
    amplitude = (np.max(steady_state) - np.min(steady_state)) / 2
    amplitudes.append(amplitude)

# Plot de tuning-kromme
plt.plot(frequencies, amplitudes)
plt.xlabel('Frequentie (rad/s)')
plt.ylabel('Amplitude (m)')
plt.title('Tuning-kromme van de Drive Mode')


# Bepaal de FWHM
max_amplitude = max(amplitudes)
half_max = max_amplitude / 2
indices = np.where(np.array(amplitudes) > half_max)[0]
fwhm = frequencies[indices[-1]] - frequencies[indices[0]]
peak = np.argmax(amplitudes)

FWHM_Line = np.linspace(0, 1, len(amplitudes)) 

for i in range(len(FWHM_Line)):
    FWHM_Line[i] = amplitudes[indices[0]]

plt.plot(frequencies, FWHM_Line, linestyle='dashed', color='grey', zorder=-1)

plt.savefig('Tuning-Kromme.png', dpi=600)

plt.show()

print(f"Maximale amplitude: {max_amplitude:.2e} m")
print(f"FWHM: {fwhm:.2e} rad/s")
print(frequencies[peak])
