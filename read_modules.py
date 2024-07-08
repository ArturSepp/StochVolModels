import stochvolmodels.pricers.logsv.affine_expansion

this = dir(stochvolmodels.pricers.logsv.affine_expansion)
print(this)
for x in this:
    if not any(y in x for y in ['__', 'Dict']):
        print(f"{x},")


print('##############################')
import inspect

all_functions = inspect.getmembers(stochvolmodels.pricers.logsv.affine_expansion, inspect.isfunction)
for x in all_functions:
    if not any(y in x for y in ['run_unit_test', 'njit', 'NamedTuple', 'dataclass', 'skew', 'kurtosis', 'abstractmethod']):
        print(f"{x[0]},")