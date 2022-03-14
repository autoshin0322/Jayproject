# 1. 김씨와 이씨는 몇 명인가요?
# 2. "이재영" 이란 이름이 몇 번 반복되나요?
# 3. 중복을 제거한 이름을 출력
# 4. 중복을 제거한 이름을 오름차순으로 정렬하여 출력

lst = ['이유덕','이재영','권종표','이재영','박민호','강상희','이재영','김지완','최승혁','이성연','박영서','박민호','전경헌','송정환','김재성','이유덕','전경헌']

kim = 0
lee = 0

for i in lst:
    if i[0] == '이':
        kim += 1
    elif i[0] == '김':
        lee += 1

print("이씨는 %d명, 김씨는 %d명이다" % (lee, kim))

leejy = 0

for jy in lst:
    if jy == '이재영':
        leejy += 1
    
print("이재영은 %d 명이다" % (leejy))

personlst = []

for person in lst:
    if person not in  personlst:
        personlst.insert(0, person)

print(personlst)
personlst.sort()
print(personlst)