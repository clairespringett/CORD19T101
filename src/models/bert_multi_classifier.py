from dotenv import load_dotenv
load_dotenv()

import os
system_path=os.getenv("sys_path")

import sys 
sys.path.append(system_path) #this will allow us to define where to find modules 



import tensorflow as tf
import bert
import tensorflow_hub as hub

from conf.local import config