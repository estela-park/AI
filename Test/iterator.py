mytuple = ("apple", "banana", "cherry")
myit = iter(mytuple)

print(myit)
# <tuple_iterator object at 0x00000200F8B54CD0>
print(next(myit))
# apple
print(next(myit))
# banana
print(next(myit))
# cherry