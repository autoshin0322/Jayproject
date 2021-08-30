from openpyxl import load_workbook
wb = load_workbook("sample.xlsx")
ws = wb.active

# ws.insert_rows(8) # 8번째 줄이 비워짐 
# ws.insert_rows(8, 5) # 8번째 줄 위치에 5줄 추가

ws.insert_cols(2) # B번째 열이 비워짐(새 빈 열 추가)
ws.insert_cols(2, 3) #B번째 열부터 3칸 추가
wb.save("sample_insert_cols.xlsx")