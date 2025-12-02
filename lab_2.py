import numpy as np
import matplotlib.pyplot as plt
import time

n = 2  
N = 10 + n  # N = 12
k_vals = np.arange(N)

signal = np.array([
    0.5, 0.7, 1.0, 0.4, 0.2, 0.0,
    -0.2, -0.3, -0.4, -0.1, 0.3, 0.6
])

#Частина I
# 1. Функція для обчислення одного члена ряду Фур’є (тригонометрична форма)
def fourier_term(signal, k, N):
    n = np.arange(N)
    Ak = np.sum(signal * np.cos(2 * np.pi * k * n / N))
    Bk = -np.sum(signal * np.sin(2 * np.pi * k * n / N))
    return Ak, Bk

# 2. Обчислення коефіцієнтів Ck = Ak + jBk
start_time = time.perf_counter()
C = np.zeros(N, dtype=complex)

mult_count = 0
add_count = 0

for k in range(N):
    Ak, Bk = fourier_term(signal, k, N)
    C[k] = Ak + 1j * Bk
    mult_count += 2 * N  # множення у sin і cos
    add_count += 2 * (N - 1)

end_time = time.perf_counter()

print("Частина I")
print(f"Час обчислення ДПФ: {end_time - start_time:.10f} c")
print(f"Кількість множень: {mult_count}")
print(f"Кількість додавань: {add_count}")

# 3. Побудова графіків спектрів
amplitude = np.abs(C)
phase = np.angle(C)

plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.title("Амплітудний спектр |Ck|")
plt.stem(amplitude, linefmt='r-', basefmt='k-')
plt.xlabel("k")
plt.ylabel("|Ck|")

plt.subplot(2, 1, 2)
plt.stem(phase)
plt.title("Фазовий спектр arg(Ck)")
plt.xlabel("k")
plt.ylabel("Фаза (рад)")
plt.tight_layout()
plt.show()

#Частина II 
# Відтворення аналогового сигналу з 8 відліків
N2 = 96 + n  # = 98
binary = list(format(N2, '08b')) # Перетворюємо у 8-бітовий код

# Для парного варіанту вставляємо 0 у 8-й розряд(найстарший біт)
binary[0] = '0'
print("8-бітове двійкове число:", "".join(binary))

# Перетворюємо в набір з 8 відліків
samples = np.array([int(bit) for bit in binary], dtype=float)
print("8 відліків сигналу:", samples)

# 2. ДПФ 
C2 = np.zeros(8, dtype=complex)

for k in range(8):
    for m in range(8):
        C2[k] += samples[m] * np.exp(-1j * 2 * np.pi * k * m / 8)

amp2 = np.abs(C2)
phase2 = np.angle(C2)

# Побудова спектрів
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.stem(amp2, linefmt='r-', basefmt='k-')
plt.title("Амплітудний спектр |Cn| (8 відліків)")

plt.subplot(2, 1, 2)
plt.stem(phase2)
plt.title("Фазовий спектр arg(Cn)")
plt.tight_layout()
plt.show()

# 3. Відновлення сигналу s(t)
t = np.linspace(0, 1, 400)
s_t = np.zeros_like(t, dtype=complex)

for k in range(8):
    s_t += C2[k] * np.exp(1j * 2 * np.pi * k * t)

plt.figure(figsize=(10, 4))
plt.plot(t, s_t.real)
plt.title("Відтворений аналоговий сигнал s(t)")
plt.xlabel("t")
plt.ylabel("s(t)")
plt.grid(True)
plt.show()

# Частина III
print("\nЧастина III")

# Обернене ДПФ 
s_rec = np.zeros(8, dtype=complex)

for m in range(8):
    for k in range(8):
        s_rec[m] += C2[k] * np.exp(1j * 2 * np.pi * k * m / 8)
    s_rec[m] /= 8

print("Відновлені відліки s(nTδ):")
for i, val in enumerate(s_rec):
    print(f"s({i}) = {val.real:.4f} + {val.imag:.4f}j")

print("\nАналітичні відліки:")
print(f"s(0) = {s_rec[0].real:.4f}")
print(f"s(1) = {s_rec[1].real:.4f}")
