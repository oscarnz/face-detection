import zipfile
from io import StringIO
from PIL import Image
import imghdr
import gzip
import os
import tqdm

#IMFD
# !gdown --id 1MdZ4-oMULv3tH4EBOJBodtwkz_LDTWZW

zippedImgs = zipfile.ZipFile('/content/IMFD.zip')
inflist = zippedImgs.infolist()
width = 128
height = 128
new_folder = '/content/Incorrect_Mask'+'_'+str(width)+'_'+str(height)

for f in  tqdm.tqdm(inflist):
  
   if not f.is_dir():
    
    ifile = zippedImgs.open(f)
    img = Image.open(ifile)
    new_img = img.resize((width,height))
    filename = ifile.name.split('/')[-1].split('.')[0]
    new_img.save(os.path.join(new_folder, filename +'_resized'+'_'+str(width)+'_'+str(height)+'.jpg'), 'JPEG', optimize=True)
    # display(img)

'''
rm -rf 'Incorrect_Mask_128_128'

from google.colab import drive
drive.mount('/content/gdrive')

!cp -r "/content/Incorrect_Mask_128_128" "/content/gdrive/MyDrive/DATASET"

!wget CMFD.zip 'https://docs.google.com/uc?export=download&id=18KLD4n69F68wQKOnclu4PwGwI7UAm2cS&confirm=t'

drive.flush_and_unmount()
'''