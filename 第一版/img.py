from PIL import Image
import numpy
#import glob
import os
import sys
class AllImg():

    #初始化函数
    def __init__(self,file_paths):
        self.file_paths = file_paths
        self.total = numpy.size(self.file_paths)
        self.cur_imp_frame = 0
        self.cur_pro_frame = 0
        self.cur_del_frame = 0
        self.num = 0
        self.new_imgs = []
        self.imgs = []
        self.tags = [[],[],[]]
        for i in range(150):
            self.import_frame()

    #运行函数
    def work(self):
        i = 0
        while i < self.total:
            self.import_frame()
            self.process_frames()
            self.del_frame()
            i += 1
    #导入图片函数
    def import_frame(self):
        if self.cur_imp_frame < self.total:
            self.imgs.append(OneImg(self.file_paths[self.cur_imp_frame],self.cur_imp_frame,self.tags))
            self.tags[1].extend(self.imgs[min(self.cur_imp_frame,300)].addr_light[0])
            self.tags[2].extend(self.imgs[min(self.cur_imp_frame,300)].addr_light[1])
            for i in range(len(self.tags[1])-len(self.tags[0])):
                self.tags[0].append(self.cur_imp_frame)
            self.cur_imp_frame += 1

    #删除图片信息函数
    def del_frame(self):
        if self.cur_imp_frame >= 301 :
            del self.imgs[0]
            num = 0
            for  i in range(len(self.tags[0])):
                if self.tags[0][i] > self.cur_del_frame:
                    break
                num = i
            self.tags[0] = self.tags[0][num:]
            self.tags[1] = self.tags[1][num:]
            self.tags[2] = self.tags[2][num:]
            self.cur_del_frame += 1
    # 处理图片函数
    # 将当前在处理的图片检查是否存在亮点,若存在,则导出范围51*51像素,前后各150帧图片
    def process_frames(self):
        if self.imgs[min(self.cur_pro_frame,150)].addr_light.size > 0:
            self.imgs_frame = min(self.cur_pro_frame,150)
            for i in range(self.imgs[self.imgs_frame].addr_light.shape[1]):
                self.arr_frame(self.imgs[self.imgs_frame].addr_light[0][i],self.imgs[self.imgs_frame].addr_light[1][i])
        print(self.cur_pro_frame)
        self.cur_pro_frame += 1
    #导出一个亮点的图片
    def arr_frame(self,x,y):
        self.new_imgs = []
        for i in range(min(self.cur_pro_frame+150,300,self.total-self.cur_del_frame)):
            self.new_imgs.append(self.one_frame(x,y,i))
        if x > 10 and y>10 and x < self.imgs[0].image.shape[0]-10 and y < self.imgs[0].image.shape[1]-10:
            self.num += 1
            self.new_path = f"{fname_pre}_{self.num}_{self.cur_pro_frame}_{x}_{y}"
            os.makedirs(self.new_path)
            for i in range(min(self.cur_pro_frame+150,300,self.total-self.cur_del_frame)):
                self.new_imgs[i].save(f"{self.new_path}/{i}.tiff")
    #导出一张图片
    def one_frame(self, x, y, frame):
        self.top = max(0, x-25)
        self.bottom = min(x+25, self.imgs[frame].image.shape[0]-1)
        self.left = max(0, y-25)
        self.right = min(y+25, self.imgs[frame].image.shape[1]-1)
        self.new_frame = self.imgs[frame].image[self.top:self.bottom,self.left:self.right]
        return Image.fromarray(self.new_frame)





class OneImg():
    # 初始化图片，使得灰度图片亮度以二维数组形式存储
    def __init__(self,file_path,frame,tags):
        self.image = numpy.array(Image.open(file_path).convert('L'))
        self.get_light_addr(frame,tags)
    # 获得该张图片符合亮度要求的位置
    def get_light_addr(self,frame,tags):
        # 初步筛选:筛选所有亮度大于100的点
        self.addr_light= numpy.array(numpy.where(self.image > 225))
        ##位置修正
        y1 = 220 
        y2 = 460
        x1 = 0
        x2 = 600
        self.addr_light_cor = [[],[]]
        for i in range(self.addr_light.shape[1]):
            if self.addr_light[0][i]>y1 and self.addr_light[0][i]<y2 and self.addr_light[1][i]>x1 and self.addr_light[1][i]<x2:
                self.addr_light_cor[0].append(self.addr_light[0][i])
                self.addr_light_cor[1].append(self.addr_light[1][i])
        self.addr_light = numpy.array(self.addr_light_cor)

        # 杂点修正:将某点范围5*5方框内亮度和小于一定值点去除
        self.addr_light_cor = [[],[]]
        for n in range(self.addr_light.shape[1]):
            x = self.addr_light[0][n]
            y = self.addr_light[1][n]
            sum = 0
            for i in range(5):
                for j in range(5):
                    sum += self.image[max(min(x-2+i,self.image.shape[0]-1),0)][max(0,min(y-2+j,self.image.shape[1]-1))]
            
            if sum > 1600:
                self.addr_light_cor[0].append(self.addr_light[0][n])
                self.addr_light_cor[1].append(self.addr_light[1][n])

        self.addr_light = numpy.array(self.addr_light_cor)

        # 空间重复修正:将相邻亮点去除
        self.addr_light_cor = [[],[]]
        self.flags = numpy.zeros(self.addr_light.shape[1])
        for i in range(self.addr_light.shape[1]):
            if self.flags[i] == 0 :
                self.addr_light_cor[0].append(self.addr_light[0][i])
                self.addr_light_cor[1].append(self.addr_light[1][i])
                for j in range(self.addr_light.shape[1]):
                    if(self.flags[j] == 0)and((self.addr_light[0][i] - self.addr_light[0][j])**2 + (self.addr_light[1][i] - self.addr_light[1][j])**2 <=25 ) :
                        self.flags[j] = 1
        self.addr_light = numpy.array(self.addr_light_cor)

        #时间重复修正: 如果该范围在150fps出现过,则去除
        self.addr_light_cor = [[],[]]
        for i in range(self.addr_light.shape[1]):
            p = 0
            for j in range(len(tags[0])):
                if (tags[0][j] + 150 > frame) and ((tags[1][j]-5 < self.addr_light[0][i])and(tags[1][j]+5 > self.addr_light[0][i])  )and ((tags[2][j]-5 < self.addr_light[1][i])and(tags[2][j] + 5 > self.addr_light[1][i])  ):
                    p = 1
            if p==0:
                self.addr_light_cor[0].append(self.addr_light[0][i])
                self.addr_light_cor[1].append(self.addr_light[1][i])
        self.addr_light = numpy.array(self.addr_light_cor)



#读取当前目录下tiff文件
#file_paths = glob.glob("*"+'*.tiff')
'''
with Image.open('./Result of PH7ZNOFPS108-1.32-1.tif') as img:
    # 获取页面数量
    pages = img.n_frames
 
    # 遍历每一页，并将其保存为单独的图像
    for i in range(pages):
        # 调到特定页面
        img.seek(i)
        print(i)
        # 将页面保存为单独的图像文件
        img.convert('L').save(f'page_{i+1}.tiff')
'''
fname_pre = sys.argv[1]
file_paths = []
for i in range(1,10001,1):
    file_paths.append(f"VimbaImage_{i}.tiff")
ALLIMG = AllImg(file_paths)
ALLIMG.work()
