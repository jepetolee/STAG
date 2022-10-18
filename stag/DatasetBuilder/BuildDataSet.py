from MakeCsv2Img import *
import torch
from PIL import Image
import torchvision.transforms as transforms
'''
ex> 'C:/Users/Adminstrator/DataStorage/BTC_USDT/'
'''
URL_Slash ='/'
def detailed_dataset_root(crypto_name,time_interval):
    return IMGFileMainRoot + URL_Slash + crypto_name + URL_Slash +time_interval+URL_Slash


class Dataset:
    def __init__(self, crypto_name, time_interval ):
        self.data_main_root = detailed_dataset_root(crypto_name,time_interval)
        self.size = None

    def call_image_tensor(self,timesteps):
        natural_sigend_timesteps = str(timesteps+1)
        image_url = self.data_main_root+natural_sigend_timesteps+'.jpg'
        image_data = Image.open(image_url)
        transform = transforms.ToTensor()
        return transform(image_data)
