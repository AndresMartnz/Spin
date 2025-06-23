import numpy as np

def rkf45_step(f, y, t, h, *args):
    """Método Runge-Kutta-Fehlberg de 5to orden con paso adaptativo."""
    
    k1 = f(t, y, *args)
    k2 = f(t + h * (1 / 4), y + h * (1 / 4) * k1, *args)
    k3 = f(t + h * (3 / 8), y + h * (3 / 32) * k1 + h * (9 / 32) * k2, *args)
    k4 = f(t + h * (12 / 13), y + h * (1932 / 2197) * k1 - h * (7200 / 2197) * k2 + h * (7296 / 2197) * k3, *args)
    k5 = f(t + h, y + h * (439 / 216) * k1 - 8 * h * k2 + h * (3680 / 513) * k3 - h * (845 / 4104) * k4, *args)
    k6 = f(t + h * (1 / 2), y - h * (8 / 27) * k1 + 2 * h * k2 - h * (3544 / 2565) * k3 + h * (1859 / 4104) * k4 - h * (11 / 40) * k5, *args)

    # Solución de 5to orden (más precisa)
    y_next_5th = y + h * (16 / 135 * k1 + 6656 / 12825 * k3 + 28561 / 56430 * k4 - 9 / 50 * k5 + 2 / 55 * k6)

    # Solución de 4to orden (para estimar el error)
    y_next_4th = y + h * (25 / 216 * k1 + 1408 / 2565 * k3 + 2197 / 4104 * k4 - 1 / 5 * k5)

    # Estimación del error
    error = np.abs(y_next_5th - y_next_4th)

    return y_next_5th, error

def solve_rkf45(f, y0, t_eval, tol, dt_max, *args):
    """Método de integración adaptativo basado en RKF45."""
    y = y0
    solution = np.zeros((len(t_eval), len(y0)))  # Guardar la solución en t_eval
    solution[0] = y0
    
    t_current = t_eval[0]
    h = dt_max  # Empezamos con el paso máximo

    for i in range(1, len(t_eval)):
        t_next_eval = t_eval[i]  # Próximo tiempo en t_eval
        
        while t_current < t_next_eval:
            # Si el próximo paso dt es mayor que la distancia a t_next_eval, ajustarlo
            h = min(h, t_next_eval - t_current)
            
            # Ejecutar un paso del método RKF45
            y_next, error = rkf45_step(f, y, t_current, h, *args)
            
            # Ajustar el tamaño del paso con base en el error
            if np.max(error) < tol:
                # Aceptar el paso
                t_current += h
                y = y_next
            else:
                # Rechazar el paso y reducir el tamaño del paso
                h *= 0.9 * (tol / np.max(error)) ** 0.25  # Ajustar el paso
            
            # Si el error es pequeño, podemos aumentar el tamaño del paso para el próximo paso
            if np.max(error) < tol / 10:
                h = min(h * 1.5, dt_max)  # No exceder dt_max
        
        # Guardar la solución en el tiempo t_eval[i]
        solution[i] = y

    return solution


def rk4_step(f, y, t, dt, *args):
    k1 = f(t, y, *args)
    k2 = f(t + dt / 2, y + dt * k1 / 2, *args)
    k3 = f(t + dt / 2, y + dt * k2 / 2, *args)
    k4 = f(t + dt, y + dt * k3, *args)
    return y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

def solve_rk4(f, y0, t_eval, dt_max, *args):
    y = y0
    solution = np.zeros((len(t_eval), len(y0)))  # Guardar la solución en t_eval
    solution[0] = y0

    t_current = t_eval[0]
    
    for i in range(1, len(t_eval)):
        t_next_eval = t_eval[i]  # Próximo tiempo en t_eval
        
        while t_current < t_next_eval:
            # Si el próximo paso dt es mayor que la distancia a t_next_eval, ajustarlo
            dt = min(dt_max, t_next_eval - t_current)
            
            # Hacer un paso de integración de tamaño dt
            y = rk4_step(f, y, t_current, dt, *args)
            t_current += dt
        
        # Guardar la solución en el tiempo t_eval[i]
        solution[i] = y

    return solution