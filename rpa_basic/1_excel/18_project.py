from openpyxl import Workbook

wb = Workbook()
ws = wb.active

ws.title="8월"

ws.append(("날짜", "금액", "사용처"))

wb.save("Buchhaltung.xlsx")