import sys

sys.path.append('C:/Users/harry/Desktop/dp3_src/Layers/A.py')
'''python import模块时， 是在sys.path里按顺序查找的。
sys.path是一个列表，里面以字符串的形式存储了许多路径。
使用A.py文件中的函数需要先将他的文件路径放到sys.path中'''
from Layers import A

a = A.A()
a.add()






