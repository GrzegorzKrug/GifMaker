import time

from functools import wraps


def measure_real_time_decorator(fun):

    @wraps(wrapped=fun)
    def wrapper(*a, **kw):
        fmt = ">4.1f"
        time0 = time.perf_counter()
        res = fun(*a, **kw)
        t_end = time.perf_counter()
        dur = t_end - time0
        if dur < 1e-3:
            timeend = f"{dur * 1000000:{fmt}} us"
        elif dur < 1:
            timeend = f"{dur * 1000:{fmt}} ms"
        else:
            timeend = f"{dur:{fmt}} s"
        print(f"{fun.__name__} exec time: {timeend}: ")

        return res

    return wrapper
