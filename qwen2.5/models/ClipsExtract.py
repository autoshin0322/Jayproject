import pandas as pd

# 1️⃣ 원본 CSV 파일 로드
clips_df = pd.read_csv("clips.csv")

# 2️⃣ 필요한 열(index, label)만 선택
true_df = clips_df[["index", "label"]]

# 3️⃣ 새 CSV로 저장
true_df.to_csv("true.csv", index=False)

print("✅ true.csv 파일이 생성되었습니다.")
print(true_df.head())  # 확인용: 앞부분 5개 행 출력
