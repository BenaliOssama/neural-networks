import numpy as np
from sklearn.metrics import log_loss

def log_loss_custom(y_true, y_pred):
    total = y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    return -np.mean(total)

y_true = np.array([0, 1, 1, 0, 1])
y_pred = np.array([0.1, 0.8, 0.6, 0.5, 0.3])

print(log_loss_custom(y_true, y_pred))
print(log_loss(y_true, y_pred))
# Both expected: 0.5472899351247816
