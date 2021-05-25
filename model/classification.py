
class classification:
    #
    def __init__(self, classifier = '', **kwargs) -> None:
        self.classifier = classifier
        for item in kwargs.items():
            print(item)
            
a = classification('asd', time = 6, name = 6)
# print(a.time)
# 
# print(a.name)