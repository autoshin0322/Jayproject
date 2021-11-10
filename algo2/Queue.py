# Stern-Aufgabe 4.10
# Pseudo code for Queue

s1 = []
s2 = []

def ENQUEUE(x):
    s1.push(x)                          # x in stack 1

def DEQUEUE():
    if len(s2) == 0 and len(s1) > 0:    # if stack 2 is empty and stack 1 not empty
        while not s1.IsEmpty():
            s2.push(s1.pop())
        x = s2.pop()
        print(x)
        return x
    else:
        x = s2.pop()                    # if stack 2 is not empty
        print(x)
        return x

def ISEMPTY():
    if s1.IsEmpty & s2.IsEmpty:         # wenn stack 1 and stack 2 are not empty, denn Queue is empty
        print("Queue is empty")
    else:
        print("Queue is not empty")