n = int(input())
name = []

for i in range(0, n):
    x = input()
    name.append(x)

for i in range(len(name)):
    sen = "Hallo {}!".format(name[i])
    print(sen)