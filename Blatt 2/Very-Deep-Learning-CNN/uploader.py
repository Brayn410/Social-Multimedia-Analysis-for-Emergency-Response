import torch
from dataset import dataset
from dataset import ConvertDataset
import torch.nn as nn
#from torch.autograd import Variable
#import numpy as np
import platform
import os
import config as cf
from torchvision import transforms

systemName = platform.system()
if systemName == 'Linux':
    dirSlash = '/'
else:
    dirSlash = '\\'
cur_dir = os.getcwd()
TEST_IMAGES = cur_dir + dirSlash + 'dog-breed-identification-dataset' + dirSlash + 'test' + dirSlash
IMG_EXT = '.jpg'
TRAIN_DATA = cur_dir + dirSlash + 'dog-breed-identification-dataset' + dirSlash + 'faker.csv'


if __name__ == '__main__':

    header = "id,affenpinscher,afghan_hound,african_hunting_dog,airedale,american_staffordshire_terrier,appenzeller,australian_terrier,basenji,basset,beagle,bedlington_terrier,bernese_mountain_dog,black-and-tan_coonhound,blenheim_spaniel,bloodhound,bluetick,border_collie,border_terrier,borzoi,boston_bull,bouvier_des_flandres,boxer,brabancon_griffon,briard,brittany_spaniel,bull_mastiff,cairn,cardigan,chesapeake_bay_retriever,chihuahua,chow,clumber,cocker_spaniel,collie,curly-coated_retriever,dandie_dinmont,dhole,dingo,doberman,english_foxhound,english_setter,english_springer,entlebucher,eskimo_dog,flat-coated_retriever,french_bulldog,german_shepherd,german_short-haired_pointer,giant_schnauzer,golden_retriever,gordon_setter,great_dane,great_pyrenees,greater_swiss_mountain_dog,groenendael,ibizan_hound,irish_setter,irish_terrier,irish_water_spaniel,irish_wolfhound,italian_greyhound,japanese_spaniel,keeshond,kelpie,kerry_blue_terrier,komondor,kuvasz,labrador_retriever,lakeland_terrier,leonberg,lhasa,malamute,malinois,maltese_dog,mexican_hairless,miniature_pinscher,miniature_poodle,miniature_schnauzer,newfoundland,norfolk_terrier,norwegian_elkhound,norwich_terrier,old_english_sheepdog,otterhound,papillon,pekinese,pembroke,pomeranian,pug,redbone,rhodesian_ridgeback,rottweiler,saint_bernard,saluki,samoyed,schipperke,scotch_terrier,scottish_deerhound,sealyham_terrier,shetland_sheepdog,shih-tzu,siberian_husky,silky_terrier,soft-coated_wheaten_terrier,staffordshire_bullterrier,standard_poodle,standard_schnauzer,sussex_spaniel,tibetan_mastiff,tibetan_terrier,toy_poodle,toy_terrier,vizsla,walker_hound,weimaraner,welsh_springer_spaniel,west_highland_white_terrier,whippet,wire-haired_fox_terrier,yorkshire_terrier\n"
    
    f = open("result.csv", "w")
    f.write(header)

    classes = ("affenpinscher", "afghan_hound", "african_hunting_dog", "airedale", "american_staffordshire_terrier", "appenzeller", "australian_terrier", "basenji", "basset", "beagle", "bedlington_terrier", "bernese_mountain_dog", "black-and-tan_coonhound", "blenheim_spaniel", "bloodhound", "bluetick", "border_collie", "border_terrier", "borzoi", "boston_bull", "bouvier_des_flandres", "boxer", "brabancon_griffon", "briard", "brittany_spaniel", "bull_mastiff", "cairn", "cardigan", "chesapeake_bay_retriever", "chihuahua", "chow", "clumber", "cocker_spaniel", "collie,curly-coated_retriever", "dandie_dinmont", "dhole", "dingo", "doberman", "english_foxhound", "english_setter", "english_springer", "entlebucher", "eskimo_dog,flat-coated_retriever", "french_bulldog", "german_shepherd", "german_short-haired_pointer", "giant_schnauzer", "golden_retriever", "gordon_setter", "great_dane", "great_pyrenees", "greater_swiss_mountain_dog", "groenendael", "ibizan_hound", "irish_setter", "irish_terrier", "irish_water_spaniel", "irish_wolfhound", "italian_greyhound", "japanese_spaniel", "keeshond", "kelpie", "kerry_blue_terrier", "komondor", "kuvasz", "labrador_retriever", "lakeland_terrier", "leonberg", "lhasa", "malamute", "malinois", "maltese_dog", "mexican_hairless", "miniature_pinscher", "miniature_poodle", "miniature_schnauzer", "newfoundland", "norfolk_terrier", "norwegian_elkhound", "norwich_terrier", "old_english_sheepdog", "otterhound", "papillon", "pekinese", "pembroke", "pomeranian", "pug", "redbone", "rhodesian_ridgeback", "rottweiler", "saint_bernard", "saluki", "samoyed", "schipperke", "scotch_terrier", "scottish_deerhound", "sealyham_terrier", "shetland_sheepdog", "shih-tzu", "siberian_husky", "silky_terrier", "soft-coated_wheaten_terrier", "staffordshire_bullterrier", "standard_poodle", "standard_schnauzer", "sussex_spaniel", "tibetan_mastiff", "tibetan_terrier", "toy_poodle", "toy_terrier", "vizsla,walker_hound", "weimaraner", "welsh_springer_spaniel", "west_highland_white_terrier", "whippet", "wire-haired_fox_terrier", "yorkshire_terrier")

    use_cuda = torch.cuda.is_available()
    net = torch.load('.' + dirSlash + 'test.pt')
    net.eval()
    trainloader, testloader2, outputs, inputs = dataset('dog-breed')
    transformations = transforms.Compose([transforms.Resize(size=(224,224)), transforms.ToTensor()])
    trueTest_set = ConvertDataset(TRAIN_DATA, TEST_IMAGES, IMG_EXT, transformations)
    testloader = torch.utils.data.DataLoader(trueTest_set, batch_size=cf.batch_size, shuffle=False, num_workers=1)
    criterion = nn.CrossEntropyLoss()
    #dataiter = iter(testloader)
    #images, labels = dataiter.next()
    #print('GT: ', ' '.join('%5s' % classes[labels[j]] for j in range(8)))
    #correct = 0 
    #total = 0
    #with torch.no_grad():
    #print(testloader2)
    #print(testloader)
    for idx, data in enumerate(testloader):
        print(idx)
        images, labels, paths = data
        outputs = net(images)
        #print(outputs.data)
        for counter, oneImage in enumerate(outputs.data):
            imgName = paths[counter].split(dirSlash)[-1].split('.')[0]
            resultValues = []
            for oneClassLoss in oneImage.data:
                if len(resultValues) >= 120:
                    break
                else:
                    val = oneClassLoss.item()
                resultValues.append(val)
            resultValues = [(c-min(resultValues))/(max(resultValues)-min(resultValues)) for c in resultValues]
            resultValues = [c/sum(resultValues) for c in resultValues]  
            resultValues = [str(c) for c in resultValues]          
            f.write(imgName + ',' + ','.join(resultValues) + '\n')
    f.close()
    os.system("zip result.csv.zip result.csv")
    os.system("kaggle competitions submit dog-breed-identification -f result.csv.zip -m \"Testing submission system\"")
    

