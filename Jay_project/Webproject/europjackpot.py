import random

lst = [i for i in range(1,51)]
lis = [i for i in range(1,13)]

result = []
bonus = []

while len(result) < 5:
    num = random.choice(lst)
    if num not in result:
        result.append(num)
    else:
        pass

while len(bonus) < 2:
    bon = random.choice(lis)
    if bon not in bonus:
        bonus.append(bon)
    else:
        pass

print(result, bonus)