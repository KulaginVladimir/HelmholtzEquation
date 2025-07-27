import numpy as np
import cmath


def calculate_RTA(lambda0, boundaries, epsilons, theta=0, polarization="s"):
    """
    Вычисляет амплитудные коэффициенты отражения (R), пропускания (T) и поглощения (A)
    для слоистой структуры с произвольным профилем диэлектрической проницаемости.

    Параметры:
        lambda0      : Длина волны в вакууме
        boundaries   : Массив границ слоев [z0, z1, z2, ..., zN]
        epsilons     : Массив диэлектрических проницаемостей для каждого слоя
        theta        : Угол падения в градусах (по умолчанию 0)
        polarization : Поляризация ('s' или 'p', по умолчанию 's')

    Возвращает:
        R, T (комплексные амплитуды), A (действительное поглощение)
    """
    # Проверка входных данных
    n_layers = len(epsilons)
    if len(boundaries) != n_layers + 1:
        raise ValueError("Количество границ должно быть len(epsilons) + 1")

    # Константы и параметры
    k0 = 2 * np.pi / lambda0
    theta_rad = np.deg2rad(theta)
    kx = k0 * np.sin(theta_rad)
    z1 = boundaries[0]
    zN = boundaries[-1]

    # Функция для вычисления kz с правильным выбором ветви
    def calc_kz(eps):
        expr = k0**2 * eps - kx**2
        return np.sqrt(expr)

    # Волновые числа в вакууме
    eps_vacuum = 1.0
    kz0 = calc_kz(eps_vacuum)

    # Вычисление kz для каждого слоя с правильным выбором ветви
    kz_list = [calc_kz(eps) for eps in epsilons]

    # Инициализация матрицы переноса
    M = np.eye(2, dtype=complex)

    # Первый интерфейс: вакуум -> первый слой
    if polarization == "s":
        # s-поляризация: используем kz
        k_left, k_right = kz0, kz_list[0]
    else:  # p-поляризация
        # p-поляризация: используем p = kz/eps
        k_left = kz0 / eps_vacuum
        k_right = kz_list[0] / epsilons[0]

    r = (k_left - k_right) / (k_left + k_right)
    t = 2 * k_left / (k_left + k_right)
    I = np.array([[1, r], [r, 1]], dtype=complex) / t
    M = M @ I

    # Обработка каждого слоя
    for i in range(n_layers):
        # Матрица распространения
        d = boundaries[i + 1] - boundaries[i]
        kz = kz_list[i]
        P = np.array(
            [[np.exp(-1j * kz * d), 0], [0, np.exp(1j * kz * d)]], dtype=complex
        )
        M = M @ P

        # Следующий интерфейс
        if i < n_layers - 1:
            # Интерфейс между слоями
            kz_left, kz_right = kz_list[i], kz_list[i + 1]
            eps_left, eps_right = epsilons[i], epsilons[i + 1]
        else:
            # Последний интерфейс: слой -> вакуум
            kz_left, kz_right = kz_list[i], kz0
            eps_left, eps_right = epsilons[i], eps_vacuum

        if polarization == "s":
            k_left, k_right = kz_left, kz_right
        else:  # p-поляризация
            k_left = kz_left / eps_left
            k_right = kz_right / eps_right

        r = (k_left - k_right) / (k_left + k_right)
        t = 2 * k_left / (k_left + k_right)
        I = np.array([[1, r], [r, 1]], dtype=complex) / t
        M = M @ I

    # Вычисление коэффициентов
    T = np.exp(1j * kz0 * (zN - z1)) / M[0, 0]

    if polarization == "s":
        R = (M[1, 0] / M[0, 0]) * np.exp(2j * kz0 * z1)
    else:  # p-поляризация
        R = -(M[1, 0] / M[0, 0]) * np.exp(2j * kz0 * z1)

    # Поглощение
    R2 = np.abs(R) ** 2
    T2 = np.abs(T) ** 2
    A = 1 - R2 - T2

    return R, T, A


# Параметры структуры
R, T, A = calculate_RTA(
    lambda0=0.5,
    boundaries=[0, 1],
    epsilons=[-2.25 + 1j],  # стекло
    theta=45,
    polarization="s",
)
print(np.abs(R) ** 2, np.abs(T) ** 2, A)
