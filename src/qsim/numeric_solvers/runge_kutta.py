def rungeKutta(f, t, h, state):
    k1 = runge_kutta_k1(f, t, state)
    k2 = runge_kutta_k2(f, t, h, state, k1)
    k3 = runge_kutta_k3(f, t, h, state, k2)
    k4 = runge_kutta_k4(f, t, h, state, k3)
    return state + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def runge_kutta_k1(f, t, state):
    return f(t, state)


def runge_kutta_k2(f, t, h, state, k1):
    return f(t + h / 2, state + h * (k1 / 2))


def runge_kutta_k3(f, t, h, state, k2):
    return f(t + h / 2, state + h * (k2 / 2))


def runge_kutta_k4(f, t, h, state, k3):
    return f(t + h, state + h * k3)


def rungeKuttaG(
    f,
    h,
    state,
    a=[[0, 0, 0, 0], [1 / 2, 0, 0, 0], [0, 1 / 2, 0, 0], [0, 0, 1, 0]],
    b=[1 / 6, 1 / 3, 1 / 3, 1 / 6],
):
    k = [f(state)]

    for i in range(len(b) - 1):
        k.append(k_n(f, h, state, a[i + 1][: i + 1], k))

    for i, b_i in enumerate(b):
        state += h * b_i * k[i]

    return state


def k_n(f, h, state, a, k_n_minus_1):
    new_state = state
    for i, a_i in enumerate(a):
        new_state += h * a_i * k_n_minus_1[i]
    return f(new_state)
