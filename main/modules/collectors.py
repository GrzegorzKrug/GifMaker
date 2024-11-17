from functools import wraps


# print("IMPORTING collectors====")

_instances = {}


class Singleton(type):
    def __call__(cls, *args, **kwargs):
        print(f"== Checking instance: cls {cls}")

        name = cls.__name__
        # print("==", name)
        if name not in _instances:
            print(f"== == Creating instance: {name}")
            # print(f"Current keys: {_instances.keys()}")
            instance = super().__call__(*args, **kwargs)
            _instances[name] = instance

        # print(f"{__name__}, CUR INSTANCE({_instances[cls].type_}): {_instances.keys()}")
        return _instances[name]


class FunctionCollector(metaclass=Singleton):
    def __init__(self):
        # print("======= Creating empty dict. ")
        self._keys = {}
        self.type_ = "base"
        # Name-function relation

        self.arguments = {}

        # Dict with variables types, limits
        # Type, minval, maxval, Number of vars, Labels

    def adder(self, fkey, *args):
        """
        Function to register function for usage.
        arg - Tuple
                type: python keyword
                min: number
                max: number
                argnumber: int
                label: str (if argnumer==1), else list of strings
                ]
        """
        def wrapper(func):
            name = fkey.lower()
            if name in self._keys:
                # print(f"{self._keys.keys()}")
                raise KeyError(f"Function already registered as: {name}")
            # if len(args) != 3:
            #     raise ValueError(f"3 params required to register filter: {fkey}")
            args_ = list(args)

            for i, a in enumerate(args_):
                a = list(a)
                args_[i] = a
                assert len(a) == 5, f"Arguments does not have 4 params: {name}, but {a}"
                nvars = a[3]
                if nvars == 1:
                    assert isinstance(a[4], str), f"Variable should be single string: {name}"
                    a[4] = [a[4]]
                else:
                    assert nvars == len(
                            a[4]), f"Variable requires list of strings: {name}, list size:{nvars}"

            self._keys[name] = func
            self.arguments[name] = args_

            # print(f"Register filter: {name}")

            @wraps(wrapped=func)
            def inner_wrapper(*a, **kw):
                out = func(*a, **kw)
                return out

            return inner_wrapper

        return wrapper

    def get_default(self, fkey):
        params = []
        for dtype, mmin, mmax, nvars, _ in self.arguments[fkey]:
            if nvars == 1:
                if dtype is str or dtype is bool:
                    params.append(mmin)
                elif mmin <= 0 <= mmax:
                    params.append(0)
                elif mmin <= 1 <= mmax:
                    params.append(1)
                else:
                    params.append(mmin)
            else:
                inner_p = []
                for n in range(nvars):
                    if dtype is str or dtype is bool:
                        params.append(mmin)
                    elif mmin <= 0 <= mmax:
                        inner_p.append(0)
                    elif mmin <= 1 <= mmax:
                        inner_p.append(1)
                    else:
                        inner_p.append(mmin)
                params.append(inner_p)
        return params

    # @classmethod
    # def __class_getitem__(cls, item):
    #     return cls.keys.get(item, None)
    @property
    def keys(self):
        return sorted(list(self._keys.keys()))

    def __getitem__(self, item):
        return self._keys.get(item, None)

    def __str__(self):
        return f"FunctionCollector({self.type_}): keys({len(self._keys)}), {self._keys.keys()}"


class SequenceModSingleton(FunctionCollector):
    # __metaclass__ = Singleton

    def __init__(self):
        super().__init__()
        self.type_ = "modifier"
        # print("INIT: ", self.type_)


class PiplineModifiersSingleton(FunctionCollector):
    def __init__(self):
        super().__init__()
        self.type_ = "pipe"
        # print("INIT: ", self.type_)


SequenceModifiers = SequenceModSingleton()
# Seq = SequenceModSingleton()
PipeLineModifiers = PiplineModifiersSingleton()
