# 프로그램 실행 순서
# 1. 입력할 정수의 개수를 사용자로부터 입력 받는다.
# 2. 입력받은 정수의 개수만큼 정수를 입력받는다.
# 3. 입력받은 정수의 합과 평균 값을 출력한다.
# 4. 할당된 메모리공간을 비운다.

# 요구사항
# 1. 메모리공간은 사용자의 입력 수의 따라 변동된다.
# 2. 사용한 공간은 마지막에 비워야 한다.
# 3. 배열을 사용하면 안된다.

numberOfInteger = input("정수의 개수를 입력해주세요: ")

number = int(numberOfInteger)

count = 0
sum = 0
while True:
    if number > count:
        Input = input("정수를 입력해주세요: ")
        Integer = int(Input)
        sum += Integer
        count += 1
    else:
        break

avg = sum / number

print("합은 %d, 평균은 %d이다." % (sum, avg))

del sum, avg, number, count