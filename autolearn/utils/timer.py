import time


TIME_BUDGET = None


def set_time_budget(time_budget):
    global TIME_BUDGET
    TIME_BUDGET = TimeBudget(time_budget)


def get_time_budget():
    global TIME_BUDGET
    return TIME_BUDGET


def _wrapped_method(m, return_dict, args, kwargs):
    res = m(*args, **kwargs)
    return_dict["res"] = res
    return return_dict


class TimeOutError(Exception):
    pass


class TimeBudget:
    def __init__(self, time_budget):
        self._time_budget = time_budget
        self._start_time = time.time()

    def reset(self):
        self._start_time = time.time()

    @property
    def remain(self):
        escape_time = time.time() - self._start_time
        return self._time_budget - escape_time

    @remain.setter
    def remain(self, value):
        self._time_budget = value

    def timing(self, seconds=None, frac=1.0):
        if seconds is None:
            seconds = self.remain * frac
        else:
            seconds = min(seconds, self.remain * frac)
        return TimeBudget(seconds)

    def check(self):
        if self.remain < 0:
            raise TimeOutError(f"Time out {self.remain: 0.4f}")

    def __add__(self, other):
        # self._time_budget += other
        return self

    def __sub__(self, other):
        # self._time_budget -= other
        return self

    def __str__(self):
        return str(self.remain)

    def __repr__(self):
        return repr(self.remain)

    def __format__(self, format_spec):
        return format(self.remain, format_spec)
