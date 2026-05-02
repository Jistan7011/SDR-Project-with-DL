import cmath
from matplotlib.pylab import *

tau = 2 * pi

def average(readings):
    base = e ** (1j * tau / 360)   # 1 degree rotation (unit complex)

    total = 0j
    for r in readings:
        v = r[1] * base ** r[0]    # (direction_deg, speed)
        total += v
        arrow(0, 0, v.real, v.imag, head_width=0.05, head_length=0.05,
              fc='r', ec='r', length_includes_head=True)

    result = total / len(readings)

    arrow(0, 0, result.real, result.imag, head_width=0.05, head_length=0.05,
          fc='b', ec='b', length_includes_head=True)

    xlim((-1.5, 1.5))
    ylim((-1.5, 1.5))
    xlabel('Real')
    ylabel('Imaginary')

    # fixes:
    # 1) handle zero-vector (direction undefined)
    if abs(result) < 1e-12:
        return (None, 0.0)

    # 2) wrap direction to [0, 360)
    direction_deg = cmath.log(result, base).real % 360.0
    speed = abs(result)
    return (direction_deg, speed)


# tests (same as screenshot style)
average(((12, 1), (15, 1), (13, 1), (9, 1), (16, 1)))
average(((358, 1), (1, 1), (359, 1), (355, 1), (2, 1)))
average(((210, 15), (290, 10), (10, 1), (90, 3), (170, 6)))
