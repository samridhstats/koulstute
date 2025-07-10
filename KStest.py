import numpy as np
import pandas as pd

np.random.seed(789)

rej_rate_u_true = np.zeros(100)
rej_rate_u_false = np.zeros(100)

# A) Loop for correct null tests
for b in range(100):
    print(f"Bootstrapping {b + 1} out of 100")

    x_t = np.random.uniform(-1, 1, 1050)
    y_t_plus_1 = np.random.exponential(scale=(0.4 + 0.2 * x_t))

    # Drop first 50 observations
    x_t = x_t[50:]
    y_t_plus_1 = y_t_plus_1[50:]

    u_t_plus_1 = y_t_plus_1 - 0.4 - 0.2 * x_t

    s_inv_inf = 1 / np.sqrt(np.mean(u_t_plus_1**2))

    def v_n_psi(m):
        indicator = (x_t <= m).astype(float)
        a = u_t_plus_1 * indicator
        return np.abs(np.sum(a) / np.sqrt(1000))

    vec_value = [v_n_psi(m) for m in x_t]
    KS_stat_1 = s_inv_inf * np.max(vec_value)

    rej_rate_u_true[b] = 1 if KS_stat_1 > 2.22 else 0

print("Mean rejection rate when null is true:", np.mean(rej_rate_u_true))

# B) Loop for incorrect null tests
for b in range(100):
    print(f"Bootstrapping {b + 1} out of 100")

    x_t = np.random.uniform(-1, 1, 1050)
    y_t_plus_1 = np.random.exponential(scale=(0.4 + 0.2 * x_t))

    x_t = x_t[50:]
    y_t_plus_1 = y_t_plus_1[50:]

    u_t_plus_1 = np.random.normal(1, 9, 1000)  # Null is false here

    s_inv_inf = 1 / np.sqrt(np.mean(u_t_plus_1**2))

    def v_n_psi(m):
        indicator = (x_t <= m).astype(float)
        a = u_t_plus_1 * indicator
        return np.abs(np.sum(a) / np.sqrt(1000))

    vec_value = [v_n_psi(m) for m in x_t]
    KS_stat_1 = s_inv_inf * np.max(vec_value)

    rej_rate_u_false[b] = 1 if KS_stat_1 > 2.22 else 0

print("Mean rejection rate when null is false:", np.mean(rej_rate_u_false))
