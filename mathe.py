sum = 0
all_sum = 0
po_sum = 0
for i in range(0,37):
    all_sum += i * i
    if i % 2 == 0:
        sum += i
        po_sum += i * i
        
print(sum, po_sum, all_sum)