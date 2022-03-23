import random

origin = []
bonus = []

while True:
    num = random.randint(1,51)
    if num not in origin:
        origin.append(num)
    else:
        pass
    if len(origin) == 5:
        break

while True:
    bon = random.randint(1,13)
    if bon not in bonus:
        bonus.append(bon)
    else:
        pass
    if len(bonus) == 2:
        break

print(origin, bonus)