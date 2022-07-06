from itertools import repeat


def apply_args_and_kwargs(func, args, kwargs):
    return func(*args, **kwargs)


def starmap_with_kwargs(pool, func, iterable_args, iterable_kwargs):
    iterable_for_starmap = zip(repeat(func), iterable_args, iterable_kwargs)
    return pool.starmap(apply_args_and_kwargs, iterable_for_starmap)
