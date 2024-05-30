import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Parameters
m = 5.3e-9  # massa in kg
k = 0.4708  # veerconstante in N/m
b = 1.4272e-6  # dempingsfactor in kg/s
F_max = 60e-9  # amplitude van de oscillerende kracht in N
omega_0 = np.sqrt(k / m)  # resonantiefrequentie in rad/s

# Differentiaalvergelijking
def model(X, t):
    x, v = X
    F_t = F_max * np.sin(omega_0 * t)
    dxdt = v
    dvdt = (F_t - b * v - k * x) / m
    return [dxdt, dvdt]

# Tijdsarray
t = np.linspace(0, 0.06, 10000)  # tijd van 0 tot 10 ms met 10000 stappen

# Oplossing
X0 = [0, 0]  # initiële conditie
sol = odeint(model, X0, t)

# Grafiek
plt.plot(t, sol[:, 0])
plt.xlabel('Tijd (s)')
plt.ylabel('Uitwijking (m)')
plt.title('Uitwijking van de Massa als Functie van de Tijd')
plt.savefig('U_t.png', dpi=600)
plt.show()

# Bepaal de amplitude
steady_state = sol[8000:, 0]  # neem een deel van de oplossing waar de oscillatie stabiel is
amplitude = np.max(steady_state) - np.min(steady_state) / 2
print(f"Amplitude van de trilling: {amplitude:.2e} m")

# Statische uitwijking
x_static = F_max / k
print(f"Statische uitwijking: {x_static:.2e} m")

# Versterkingsfactor
Q_factor = amplitude / x_static
print(f"Versterkingsfactor (Q-factor): {Q_factor:.2f}")