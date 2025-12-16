"""
自定义一个可计算包含若干数list中数的平均值的函数。
即如果你定义的函数为pjs()，那么执行pjs([1,3,4,6])后，应该返回平均数3.5。
【提示：len()可用来计算list的长度，用法如len([1,2,3,4])】。
"""
import numpy as np

def pjs(lt):
    return np.mean(lt)

print(pjs([1,3,4,6]))