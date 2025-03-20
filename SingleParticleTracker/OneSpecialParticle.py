import os
from matplotlib import pyplot as plt
import numpy as np
from tifffile import imwrite
from tqdm import tqdm
class OneSpecialParticle:
    def __init__(self, save_path = "./result/",frames = None,dpi = 400):
        self.dpi = dpi
        self.save_path = save_path
        self.frames = np.array(frames)
    def run(self,tracks,track_tags,order_special ):
        #print(tracks[0])
        for i ,track in enumerate(tqdm(tracks,desc = "特殊粒子保存：",unit="条")):
            indices = self.find_indices(track_tags["gred_sum"][order_special["gred_sum"][i]],1)
            filtered_indices = self.filter_indices_by_interval(indices)
            for j in filtered_indices:
                self.process(track[j])

    def find_indices(self,lst, target):
        return [index for index, value in enumerate(lst) if value == target]
    def filter_indices_by_interval(self,indices):
        if not indices:
            return []  # 如果列表为空，直接返回空列表

        filtered_indices = [indices[0]]  # 从第一个下标开始
        last_index = indices[0]

        for index in indices[1:]:
            if index - last_index >= 100:
                filtered_indices.append(index)
                last_index = index
        return filtered_indices
    def process(self,pos):
        fps_s = max(0,pos[0]-50)
        fps_e = min(self.frames.shape[0],pos[0]+50)
        x_s = max(0,round(pos[2])-25)
        y_s = max(0,round(pos[1])-25)
        x_e = min(self.frames.shape[2],round(pos[2])+25)
        y_e = min(self.frames.shape[1],round(pos[1])+25)
        new_frame = self.frames[fps_s:fps_e,y_s:y_e,x_s:x_e]
        if y_e+x_e - y_s - x_s>90:
            path = f"{self.save_path}/{pos[0]}_{round(pos[1])}_{round(pos[2])}"
            self.save_frames_as_tiff(new_frame,path)
            x_s = max(0,round(pos[2])-2)
            y_s = max(0,round(pos[1])-2)
            x_e = min(self.frames.shape[2],round(pos[2])+2)
            y_e = min(self.frames.shape[1],round(pos[1])+2)
            new_frame = self.frames[fps_s:fps_e,y_s:y_e,x_s:x_e]
            path = f"{self.save_path}/{pos[0]}_{round(pos[1])}_{round(pos[2])}"
            self.light_denisty(new_frame,path)

    def save_frames_as_tiff(self,new_frame, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        for i in range(new_frame.shape[0]):
            frame = new_frame[i] 
            frame_filename = os.path.join(output_folder, f"{i}.tiff")  
            imwrite(frame_filename, frame)
    def light_denisty(self,frame,path):
        if not os.path.exists(path):
            os.makedirs(path)
        y = np.sum(frame,axis=(1,2))
        plt.plot([4*i for i in range(len(y))],y)
        plt.scatter([4*i for i in range(len(y))],y,marker='D')
        plt.tick_params(axis='both', direction='in', which='both')
        plt.locator_params(axis='x', nbins=5)  # x轴最多显示5个刻度
        plt.locator_params(axis='y', nbins=5)  # y轴最多显示5个刻度
        plt.xlabel("$t(ms)$")
        plt.ylabel("$I$")
        plt.savefig(f"{path}/0.png",dpi=self.dpi,bbox_inches='tight',pad_inches=0,transparent=True)
        plt.close()