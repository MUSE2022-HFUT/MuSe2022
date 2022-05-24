import pandas as pd
import os
from loss_def_res18 import MVFace
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image, ImageFile
import csv
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# face_dir
dataset_path = '/userdata/langjunjie/datasets/MUSE2022/c2_muse_reaction/feats/Faces'

def main():

    # feature_save_path
    save_path = '/userdata/langjunjie/datasets/MUSE2022/c2_muse_reaction/feats/Feature_DAN'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])  

    model = MVFace(backbone='resnet18',num_class_1=3 ,num_class_2_list=[2,2,3,7,7])
    model.to(device)
    checkpoint = torch.load('affectnet7_best_acc0.6527.pth.tar')
    model.load_state_dict(checkpoint['model_state_dict']) 

        

    
    for sequence_id in tqdm( range(len(os.listdir(dataset_path))) ) :

        sequence_id = '%05d' % sequence_id

        image_full_path_list = read_image_path(sequence_id,dataset_path)
            

        dataset = ImageList(image_full_path_list,data_transforms)

        data_loader = torch.utils.data.DataLoader(dataset, batch_size=dataset.__len__(), shuffle=False, num_workers=4)
        
        with torch.no_grad():         
            for i, (input) in enumerate(data_loader): 
                input = input.to(device) 
                out = model(input)
        
        file = open(os.path.join(save_path,sequence_id+'.csv'),'a+')
        writer = csv.writer(file)
        writer.writerow(['File_ID','timestamp']+list(range(1,513)))
        for j in range(len(image_full_path_list)):
            time_stamp = image_full_path_list[j].split('/')[-1].split('.')[0]
            feature = out[j].cpu().numpy().tolist() if len(image_full_path_list) > 1 else out.cpu().numpy().tolist()
            writer.writerow( ['['+sequence_id+']',time_stamp] + feature )
        
        file.close()
                


class ImageList(data.Dataset):
    def __init__(self, image_full_path_list, transform=None):
        self.imgList = image_full_path_list
        self.transform = transform
        self.totensor = transforms.ToTensor()


    def __getitem__(self, index):
        imgPath = self.imgList[index]
        img = self.PIL_loader(imgPath)

        if self.transform is not None:
            img = self.transform(img)

        return img

    def getit(self):
        imgPath = self.imgList[0]
        img = self.PIL_loader(imgPath)

        if self.transform is not None:
            img = self.transform(img)

        return img


    def PIL_loader(self,path):
        try:
            with open(path, 'rb') as f:
                return Image.open(f).convert('RGB')
        except IOError:
            print('Cannot load image ' + path)       
       
    
    def __len__(self):
        return len(self.imgList)



def read_image_path(sequence_id,dataset_path):
    sequence_path = os.path.join(dataset_path,sequence_id)
    image_name_list = os.listdir(sequence_path)
    if 'all_features.pickle' in image_name_list:
        image_name_list.remove('all_features.pickle')
    image_name_list_nojpg = [i.split('.')[0] for i in image_name_list]
    image_name_list_nojpg = sorted(image_name_list_nojpg,key=int)
    image_name_list = [i+'.jpg' for i in image_name_list_nojpg]

    image_full_path_list = [os.path.join(sequence_path,i) for i in image_name_list]
    return image_full_path_list

if __name__ == "__main__":
    
    main()
    print("Process has finished!")