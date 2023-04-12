import pandas as pd
import numpy as np
from hyppo.ksample import Energy, MMD


chat_id = 487727948 # Ваш chat ID, не меняйте название переменной

def solution(x: np.array, y: np.array) -> bool:
    # Измените код этой функции
    # Это будет вашим решением
    # Не меняйте название функции и её аргументы
    p_value = MMD(compute_kernel="rbf", gamma=1).test(x, y)[1]

    # Если p-уровень значимости меньше заданного уровня значимости, отклоняем гипотезу однородности выборок
    return p_value < 0.01
