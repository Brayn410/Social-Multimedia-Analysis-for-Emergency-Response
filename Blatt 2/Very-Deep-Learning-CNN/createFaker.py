from os import listdir
from os.path import isfile, join

faker = open("./dog-breed-identification-dataset/faker.csv", "w")
faker.write("id,breed\n")
mypath = "./dog-breed-identification-dataset/test/"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
for name in onlyfiles:
    prefix = name.split('.')[0]
    faker.write(prefix + ",dingo\n")
faker.close()
