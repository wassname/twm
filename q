[0;31mSignature:[0m [0mNewType[0m[0;34m([0m[0mname[0m[0;34m,[0m [0mtp[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;31mSource:[0m   
[0;32mdef[0m [0mNewType[0m[0;34m([0m[0mname[0m[0;34m,[0m [0mtp[0m[0;34m)[0m[0;34m:[0m[0;34m[0m
[0;34m[0m    [0;34m"""NewType creates simple unique types with almost zero[0m
[0;34m    runtime overhead. NewType(name, tp) is considered a subtype of tp[0m
[0;34m    by static type checkers. At runtime, NewType(name, tp) returns[0m
[0;34m    a dummy function that simply returns its argument. Usage::[0m
[0;34m[0m
[0;34m        UserId = NewType('UserId', int)[0m
[0;34m[0m
[0;34m        def name_by_id(user_id: UserId) -> str:[0m
[0;34m            ...[0m
[0;34m[0m
[0;34m        UserId('user')          # Fails type check[0m
[0;34m[0m
[0;34m        name_by_id(42)          # Fails type check[0m
[0;34m        name_by_id(UserId(42))  # OK[0m
[0;34m[0m
[0;34m        num = UserId(5) + 1     # type: int[0m
[0;34m    """[0m[0;34m[0m
[0;34m[0m[0;34m[0m
[0;34m[0m    [0;32mdef[0m [0mnew_type[0m[0;34m([0m[0mx[0m[0;34m)[0m[0;34m:[0m[0;34m[0m
[0;34m[0m        [0;32mreturn[0m [0mx[0m[0;34m[0m
[0;34m[0m[0;34m[0m
[0;34m[0m    [0mnew_type[0m[0;34m.[0m[0m__name__[0m [0;34m=[0m [0mname[0m[0;34m[0m
[0;34m[0m    [0mnew_type[0m[0;34m.[0m[0m__supertype__[0m [0;34m=[0m [0mtp[0m[0;34m[0m
[0;34m[0m    [0;32mreturn[0m [0mnew_type[0m[0;34m[0m[0;34m[0m[0m
[0;31mFile:[0m      ~/miniforge3/lib/python3.9/typing.py
[0;31mType:[0m      function