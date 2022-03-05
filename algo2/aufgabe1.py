class test:
    def __init__(self, name):
        self.name = "fdsafsa"
        
    def dict(self):
        test.dictionary = "ㄹㅇㄴㄹㅇㄴㅁㄹㄴㅇㅁ"
        
    def dich_ruf(self):
        a = test.dictionary
        
        print(a)

x = test("name")

x.dict()
x.dich_ruf()