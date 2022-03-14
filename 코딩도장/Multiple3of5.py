# 10미만의 자연수에서 3과 5의 배수를 구하면 3,5,6,9이다. 이들의 총합은 23이다.
# 1000미만의 자연수에서 3,5의 배수의 총합을 구하라.

lst = []
array = []
# for i in range(0, 1001):
#     if i % 3 == 0 :
#         lst.append(i)
#     elif i % 5 == 0:
#         lst.append(i)

# for i in lst:
#     if i not in array:
#         array.append(i)
# result = 0

# for i in array:
#     result = result + i

# print(result)

for i in range(0, 1001):
    if i % 3 == 0 or i % 5 == 0:
        if i not in array:
            array.append(i)

print(sum(array))