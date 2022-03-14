# 일전에 뭐 게임 회사에서 본 간단한 퀴즈 테스트 입니다.

# 0~9까지의 문자로 된 숫자를 입력 받았을 때, 이 입력 값이 0~9까지의 숫자가 각각 한 번 씩만 사용된 것인지 확인하는 함수를 구하시오.

# sample inputs: 0123456789 01234 01234567890 6789012345 012322456789
# sample outputs: true false false true false

lst = input("정수를 입력하세요: ").split()

result_lst = []
for number in lst:
    mid_result = []
    for i in range(0, 10):
        if number.count(str(i)) != 1:
            mid_result.insert(0, False)
        else:
            mid_result.insert(0, True)
    if False in mid_result:
        result_lst.append(False)
    else:
        result_lst.append(True)

print(result_lst)