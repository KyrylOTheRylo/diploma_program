from enum import IntEnum


class Starter(IntEnum):
    a = 1
    b = 2


print(type(Starter.a.value))

b = 1,
print(type(b))
