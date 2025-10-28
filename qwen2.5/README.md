# 📖 Vision Language Model (VLM) - QWEN2.5-VL-7B-Instruct 
👨‍🎓 Goethe-Universität Frankfurt am Main \
🏛️ Fachbereich 12 Institut für Informatik \
📫 E-mail: S*******@stud.uni-frankfurt.de
---
Huggingface:
Github:

## Aktuelle Stand
Antworten von QWEN haben den Tendenz zur Antwort "Indexing". Um das Problem zu loesen, braucht man den Prompt Engineering.

Text soll wie folgt sein:
"""
You are a gesture classification assistant.  
Your task is to analyze the hand gesture shown in the given video.  

Choose only one label from the following list:  
[representing, molding, acting, indexing, other, beat, drawing, emblematic].

Do not always choose the same label.  
Base your answer on the motion, hand shape, and context shown in the video.

Do not write any explanation or additional text.
"""
