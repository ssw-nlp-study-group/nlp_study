from machine_learning import svd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import requests
from io import BytesIO


class ImgCompresser():
    def __init__(self, image):
        image = image / 255
        self.imag_shape = image.shape
        self.image = image.reshape((self.imag_shape[0], -1))
        self.U, self.sigma, self.VT = svd.svd(self.image)

    def compress(self, sval_nums):
        # 提取前sval_nums个特征数据
        compressed = self.U[:, 0:sval_nums] @ self.sigma[0:sval_nums, 0:sval_nums] @ self.VT[0:sval_nums, :]
        return compressed.reshape(self.imag_shape)


# 从网络下载图片
img_url = 'https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fb-ssl.duitang.com%2Fuploads%2Fitem%2F201501%2F26%2F20150126093300_uHzYv.jpeg&refer=http%3A%2F%2Fb-ssl.duitang.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=jpeg?sec=1617412512&t=796988fe549e34704175fee730e8625a'

response = requests.get(img_url)
response = response.content

BytesIOObj = BytesIO()
BytesIOObj.write(response)

# image = Image.open('bingguo.jpg')
image = Image.open(BytesIOObj)

image = np.array(image)
compresser = ImgCompresser(image)

image = image.reshape((image.shape[0], -1, 3))
fix, ax = plt.subplots(2, 5)

ax[0, 0].set(title='300/300')
ax[0, 0].imshow(image)
ax[0, 1].set(title='1/300')
ax[0, 1].imshow(compresser.compress(sval_nums=1))
ax[0, 2].set(title='3/300')
ax[0, 2].imshow(compresser.compress(sval_nums=3))
ax[0, 3].set(title='5/300')
ax[0, 3].imshow(compresser.compress(sval_nums=5))
ax[0, 4].set(title='7/300')
ax[0, 4].imshow(compresser.compress(sval_nums=7))
ax[1, 0].set(title='10/300')
ax[1, 0].imshow(compresser.compress(sval_nums=10))
ax[1, 1].set(title='13/300')
ax[1, 1].imshow(compresser.compress(sval_nums=13))
ax[1, 2].set(title='16/300')
ax[1, 2].imshow(compresser.compress(sval_nums=16))
ax[1, 3].set(title='19/300')
ax[1, 3].imshow(compresser.compress(sval_nums=19))
ax[1, 4].set(title='30/300')
ax[1, 4].imshow(compresser.compress(sval_nums=30))
plt.show()
