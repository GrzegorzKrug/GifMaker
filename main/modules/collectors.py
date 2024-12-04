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

        self.arguments = dict()
        self.defaults = dict()

        # Dict with variables types, limits
        # Type, minval, maxval, Number of vars, Labels

    def adder(self, fkey, *args):
        """
        Function to register function for usage.
        arg - Tuple
                type: python keyword
                ar1: number min / str: default
                ar2: number max / list str [options]
                argnumber: int
                label: str (if argnumebr==1), else list of strings
                ]
        defaultArgTuple - Tuple (optional)
                None: Literal None (must be provided to mark default)
                arg0Type: type
                ...
                argNType: type
        """
        def wrapper(func):
            "Checking Wrapper?"
            name = fkey.lower()
            if name in self._keys:
                # print(f"{self._keys.keys()}")
                raise KeyError(f"Function already registered as: {name}")
            # if len(args) != 3:
            #     raise ValueError(f"3 params required to register filter: {fkey}")
            argsFixed = list(args)
            hasDefault = False
            default_vals = None

            if len(argsFixed) > 0 and isinstance(argsFixed[-1][0], (type(None), )):
                hasDefault = True
                default_vals = argsFixed[-1][1:]
                argsFixed = argsFixed[:-1]

            for i, curArg in enumerate(argsFixed):
                curArg = list(curArg)
                argsFixed[i] = curArg

                assert len(curArg) == 5, \
                    f"Arguments does not have 4 params: {name}, but {len(curArg)}: {curArg}"
                varN = curArg[3]

                "Checking Label strings"
                if varN == 1:
                    assert isinstance(curArg[4], str), \
                        f"Variable should be single string: {name}"
                    curArg[4] = [curArg[4]]
                else:
                    assert varN == len(
                        curArg[4]), f"Variable requires list of strings: {name}, list size:{varN}"
                    for lb in curArg[4]:
                        assert isinstance(lb, str), \
                            f"Variable in list be string. Function {name}, got: {lb}"

            self._keys[name] = func
            self.arguments[name] = argsFixed
            if hasDefault:
                areDefValidType = True
                validatedDefs = []

                if len(argsFixed) != len(default_vals):
                    print(
                        f"Default value miss match from parameters for: {name}, Arg required: {len(argsFixed)} but provided: {len(default_vals)}")
                    areDefValidType = False
                    raise ValueError
                else:
                    for i, value in enumerate(default_vals):
                        varN = argsFixed[i][3]
                        thisType = argsFixed[i][0]

                        try:
                            if varN == 1:
                                temp = thisType(value)
                                validatedDefs.append(temp)
                            elif varN > 1:
                                tmpList = []
                                for v in value:
                                    temp = thisType(v)
                                    tmpList.append(temp)

                                validatedDefs.append(tmpList)

                        except Exception as err:
                            print(
                                f"Failed to convert default parameter to type: {thisType} from: {value} in method: {name}")
                            areDefValidType = False
                            raise ValueError(err)
                            break

                if areDefValidType:
                    self.defaults[name] = validatedDefs

            # print(f"Register filter: {name}")

            @wraps(wrapped=func)
            def inner_wrapper(*a, **kw):
                out = func(*a, **kw)
                return out

            return inner_wrapper

        return wrapper

    def get_default(self, fkey):
        "Gets defult params"
        if fkey in self.defaults:
            return self.defaults[fkey].copy()

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
