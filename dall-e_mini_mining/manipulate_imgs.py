import os
import shutil
from PIL import Image

def rename():
    trim = True
    target_dir = "raw_fake_dataset"
    
    to_delete = []
    for class_dir in os.listdir(target_dir):
        if class_dir == ".DS_Store":
            continue
        cnt = 1
        print(f"Renaming class {class_dir}")
        for img in os.listdir(os.path.join(target_dir, class_dir)):
            if not trim:
                os.rename(os.path.join(target_dir, class_dir, img), os.path.join(target_dir, class_dir, f"{cnt}.jpeg"))
            
            if trim:
                if cnt > 6000:
                    to_delete.append(os.path.join(target_dir, class_dir, img))
                else:
                    os.rename(os.path.join(target_dir, class_dir, img), os.path.join(target_dir, class_dir, f"{cnt}.jpeg"))
            cnt += 1
            
    for item in to_delete:
        os.remove(item)
        
        
def copy():
    src = "fake_dataset_2"
    dst = "fake_dataset"
    for class_dir in os.listdir(src):
        if class_dir == ".DS_Store":
            continue
        cnt = 1
        print(f"Copying class {class_dir}")
        for img in os.listdir(os.path.join(src, class_dir)):
            shutil.copy(os.path.join(src, class_dir, img), os.path.join(dst, class_dir, img))
            
            
def resize():
    target_dir = "raw_fake_dataset_60000"
    
    for class_dir in os.listdir(target_dir):
        if class_dir == ".DS_Store":
            continue

        print(f"Resizing class {class_dir}")
        for img_name in os.listdir(os.path.join(target_dir, class_dir)):
            img = Image.open(os.path.join(target_dir, class_dir, img_name))
            os.remove(os.path.join(target_dir, class_dir, img_name))
            img = img.resize((32, 32))
            img_name, ext = img_name.split(".")
            img.save(os.path.join(target_dir, class_dir, f"{img_name}.png"))
            
            
def separate_test_data():
    target_dir = "raw_fake_dataset_60000_32x32"
    
    os.makedirs(f"{target_dir}_test_data", exist_ok=True)
    
    for class_dir in os.listdir(target_dir):
        if class_dir == ".DS_Store":
            continue
        
        os.makedirs(os.path.join(f"{target_dir}_test_data", class_dir), exist_ok=True)

        print(f"class {class_dir}")
        file_list = os.listdir(os.path.join(target_dir, class_dir))
        
        for img_name in file_list:
            img_num, ext = img_name.split(".")
            if int(img_num) > 5000:
                src = os.path.join(target_dir, class_dir, img_name)
                dst = os.path.join(f"{target_dir}_test_data", class_dir, img_name)
                
                shutil.copy(src, dst)
                os.remove(src)
            
            


#copy()   
#rename()
#resize()
#trim()
separate_test_data()
