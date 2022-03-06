import functools
import time
from enum import IntEnum

log_level = IntEnum("LOG_LEVEL", ["error", "info", "debug"])
log_level_map = {
    "error": log_level.error,
    "info": log_level.info,
    "debug": log_level.debug
}
nesting_level = 0
global_level = log_level.info


def log(entry, level="info"):
    global nesting_level, global_level, log_level_map
    level = log_level_map.get(level, log_level.info)
    if level <= global_level:
        space = "-" * (4 * nesting_level)
        print(f"{space}{entry}")


def timeit(method):
    @functools.wraps(method)
    def timed(*args, **kw):
        global nesting_level

        class_name = ""
        if len(args) > 0 and hasattr(args[0], '__class__'):
            class_name = f"{args[0].__class__.__name__}."
        log(f"Start [{class_name}{method.__name__}]:")
        nesting_level += 1

        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()

        nesting_level -= 1
        log(f"End   [{class_name}{method.__name__}]. Time elapsed: {end_time - start_time:0.4f} sec.")

        return result

    return timed
