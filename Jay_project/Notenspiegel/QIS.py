import time
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from openpyxl import Workbook
from openpyxl.styles import Font, Border, Side, PatternFill, Alignment
from openpyxl.styles.alignment import Alignment

wb = Workbook()
ws = wb.active

url = "https://qis.server.uni-frankfurt.de/qisserver/rds?state=user&type=8&topitem=functions&breadCrumbSource=portal"

options = webdriver.ChromeOptions()
options.headless = True
options.add_argument("window-size=1920x1080")

browser = webdriver.Chrome(options=options)
browser.get(url)
browser.maximize_window()

browser.find_element_by_xpath("//*[@id='wrapper']/div[3]/a").click()

id = browser.find_element_by_xpath("//*[@id='asdf']")
pw = browser.find_element_by_xpath("//*[@id='fdsa']")

# login
id.send_keys("s6010479")
pw.send_keys("Wogusrne1!")
browser.find_element_by_xpath("//*[@id='loginForm:login']").click()

# Pruefungsverwaltung
browser.find_element_by_xpath("//*[@id='makronavigation']/ul/li[3]/a").click()

# Notenspiegel
browser.find_element_by_xpath("//*[@id='wrapper']/div[6]/div[2]/div/form/div/ul/li[3]/a").click()

# PO2019
browser.find_element_by_xpath("//*[@id='wrapper']/div[6]/div[2]/form/ul/li/a[2]/img").click()
# browser.find_element_by_xpath("//*[@id='wrapper']/div[6]/div[2]/form/ul/li/ul/li[2]/a[1]/img").click()

# records
scores = []
for i in range(10, len(browser.find_elements_by_class_name("qis_records"))):
    a = i % 9
    if a == 1 or a == 3 or a == 5 or a == 6 or a == 7 or a == 0:
        scores.append(browser.find_elements_by_class_name("qis_records")[i].text)

for i in range(0, len(scores)):
    a = i % 6
    if a == 0:
        Bestanden = str(scores[i+2])
        if Bestanden != "bestanden":
            continue
        else: ws.append([scores[i], scores[i+1], scores[i+3], scores[i+4]])

ws.append(["","","",""])
wb.save("Notenspielgel.xlsx")

browser.quit()