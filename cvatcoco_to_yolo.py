#! /usr/bin/env python3

from ultralytics.data.converter import convert_coco
import os
import zipfile
import shutil
import glob
import random

#export from CVAT as COCO1.1
#comes in as ziped file with annotations/instances_default.json


def convert_coco_n_move(download_dir, file_name, save_dir):
    """From a dip file in a download directory, extract the contents 
    and convert the COCO labels to YOLO labels saving the labels in save_dir"""
    with zipfile.ZipFile(os.path.join(download_dir, file_name), 'r') as zip_ref:
        zip_ref.extractall(download_dir)
        #annotation folder in downloads

    downloaded_labels_dir = os.path.join(download_dir, 'annotations')
    lable_dir = os.path.join(save_dir, 'labels')
    coco_labels = os.path.join(lable_dir, 'default')

    convert_coco(labels_dir=downloaded_labels_dir, save_dir=save_dir,
                    use_segments=True, use_keypoints=False, cls91to80=False)

    for filename in os.listdir(coco_labels):
        source = os.path.join(coco_labels, filename)
        destination = os.path.join(lable_dir, filename)
        if os.path.isfile(source):
            shutil.move(source, destination)

def fill_in_blank_text_files(images_folder, labels_folder):
    """From an image folder, create a blank text file for each image in the labels folder that does not already exist."""
    image_files = [os.path.splitext(file)[0] for file in os.listdir(images_folder)]
    label_files = [os.path.splitext(file)[0] for file in os.listdir(labels_folder)]
    missing_files = [file for file in image_files if file not in label_files]
    print(f'number of missing files: {len(missing_files)}')
    for file_name in missing_files:
        with open(os.path.join(labels_folder, file_name+'.txt'), 'w') as file:
            pass
        print(f"added {file_name}.txt to labels folder")
    print ("blank text files added")

def split_train_val_test(dir, train_ratio, test_ratio, valid_ratio):
    def check_ratio(test_ratio,train_ratio,valid_ratio):
        if(test_ratio>1 or test_ratio<0): ValueError(test_ratio,f'test_ratio must be > 1 and test_ratio < 0, test_ratio={test_ratio}')
        if(train_ratio>1 or train_ratio<0): ValueError(train_ratio,f'train_ratio must be > 1 and train_ratio < 0, train_ratio={train_ratio}')
        if(valid_ratio>1 or valid_ratio<0): ValueError(valid_ratio,f'valid_ratio must be > 1 and valid_ratio < 0, valid_ratio={valid_ratio}')
        if not((train_ratio+test_ratio+valid_ratio)==1): ValueError("sum of train/val/test ratio must equal 1")
    check_ratio(test_ratio,train_ratio,valid_ratio)

    imagelist = glob.glob(os.path.join(dir,'images', '*.jpg'))
    txtlist = glob.glob(os.path.join(dir, 'labels', '*.txt'))
    txtlist.sort()
    imagelist.sort()
    imgno = len(txtlist) 
    noleft = imgno

    validimg, validtext, testimg, testtext = [], [], [], []

    # function to seperate files into different lists randomly while retaining the same .txt and .jpg name in the specific type of list      
    def seperate_files(number,newimglist,newtxtlist,oldimglist,oldtxtlist):
        for i in range(int(number)):
            r = random.randint(0, len(oldtxtlist) - 1)
            newimglist.append(oldimglist[r])
            newtxtlist.append(oldtxtlist[r])
            oldimglist.remove(oldimglist[r])
            oldtxtlist.remove(oldtxtlist[r])
        return oldimglist, oldtxtlist

    imagelist, txtlist = seperate_files(imgno*valid_ratio,validimg,validtext,imagelist,txtlist)
    imagelist, txtlist = seperate_files(imgno*test_ratio,testimg,testtext,imagelist,txtlist)

    # function to preserve symlinks of src file, otherwise default to copy
    def copy_link(src, dst):
        if os.path.islink(src):
            linkto = os.readlink(src)
            os.symlink(linkto, os.path.join(dst, os.path.basename(src)))
        else:
            shutil.copy(src, dst)
    # function to make sure the directory is empty
    def clean_dirctory(savepath):
        if os.path.isdir(savepath):
            shutil.rmtree(savepath)
        os.makedirs(savepath, exist_ok=True)
    # function to move a list of files, by cleaning the path and copying and preserving symlinks
    def move_file(filelist,savepathbase,savepathext):
        output_path = os.path.join(savepathbase, savepathext)
        #clean_dirctory(output_path)
        os.makedirs(output_path, exist_ok=True)
        for i, item in enumerate(filelist):
            copy_link(item, output_path)

    move_file(txtlist,dir,'labels/train')
    move_file(imagelist,dir,'images/train')
    move_file(validtext,dir,'labels/val')
    move_file(validimg,dir,'images/val')
    move_file(testtext,dir,'labels/test')
    move_file(testimg,dir,'images/test')

    print("split complete")

def main():
    file_name = 'cslics_desktop_dec_coco.zip'
    save_dir = '/home/java/Downloads/cslics_dektop_dec'
    download_dir = '/home/java/Downloads'
    split = True #got to get images in dir before split
    fill_in = True #if want to fill in blank text files
    #convert_coco_n_move(download_dir, file_name, save_dir)

    save_dir = '/home/java/Java/data/cslics_desktop_data/'
    images_folder = os.path.join(save_dir,'images')
    labels_folder = os.path.join(save_dir,'labels')
    #fill_in_blank_text_files(images_folder, labels_folder)

    train_ratio = 0.70
    test_ratio = 0.15
    valid_ratio = 0.15
    save_dir = '/home/java/Java/data/cslics_aloripedes_n_amtenuis_jan_2000/200_short'
    split_train_val_test(save_dir, train_ratio, test_ratio, valid_ratio)

if __name__ == "__main__":
    main()