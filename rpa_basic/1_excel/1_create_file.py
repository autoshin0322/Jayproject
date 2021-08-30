from openpyxl import Workbook

wb = Workbook() # 새 워크북을 생성
ws = wb.active # 현재 활성화된 sheet 가져옴
ws.title = "Kyungseo Sheet" # sheet 의 이름을 변경
wb.save("sample_for_kyungseo.xlsx")
wb.close()