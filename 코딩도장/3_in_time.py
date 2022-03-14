hour = [3, 13, 23]
hourWithoutThree = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22]
ThreeInMinute = [3, 13, 23, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 43, 53]

countThreeInMinute = len(ThreeInMinute)

countMinute = len(hourWithoutThree)
countHour = len(hour)
SecForMinute = countMinute * countThreeInMinute * 60
SecForHour = countHour * 60 * 60

print(SecForHour + SecForMinute)