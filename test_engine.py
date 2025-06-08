# 属性查找
# 对象=》类=》父类=》object
class Mytype(type):
    age = 18

    def __call__(self, *args, **kwargs):
        # print(object.__new__ is self.__new__) # True
        obj = self.__new__(self)
        self.__init__(obj, *args, **kwargs)
        return obj


class Animal(object):
    # age = 17
    pass


class Human(Animal):
    # age = 16
    pass


class Chinese(Human, metaclass=Mytype):
    # age = 15
    pass


class ScPerson(Chinese):
    # age = 14
    pass


obj = ScPerson()
# print(obj.age)
# print(ScPerson.age)
print(type(Human))
