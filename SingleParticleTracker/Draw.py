
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
class Draw:
    def __init__(self,dpi = 2400):
        """
        绘图类
        - dpi: 图像分辨率
        """
        self.dpi = dpi
    def draw_run(self,tracks,track_tags,order_special ,frames,special ,postions):
        """
        绘制轨迹，绘制速度，绘制概率，绘制密度
        - tracks: 轨迹
        - track_tags: 轨迹标签
        - order_special: 特殊轨迹的顺序
        - frames: 帧
        - special: 特殊轨迹
        - postions: 粒子位置
        """
        self.draw_track(tracks,frames)
        self.draw_scatter_norm_v(tracks)
        self.draw_scatter_special_norm_v(tracks,track_tags,order_special)
        self.draw_probability_norm_v(tracks)
        self.draw_density(frames,special,postions)
        
    def draw_track(self,tracks,frames):
        """
        绘制轨迹
        - tracks: 轨迹
        - frames: 帧
        """
        plt.imshow(frames[0],cmap='gray', vmin=0, vmax=255)
        for track in tqdm(tracks,desc = "绘制轨迹：",unit="条"):
            x = [point[2] for point in track]  
            y = [point[1] for point in track]  
            plt.plot(x,y,linewidth = 0.1)
        print("图像保存中...")
        plt.savefig("tracks.png",dpi=self.dpi)
        plt.close()
    def draw_scatter_norm_v(self,tracks):
        """
        绘制速度散点图
        - tracks: 轨迹
        """
        v_x = []
        v_y = []
        for i,track in enumerate(tqdm(tracks,desc = "绘制速率散点图：",unit="条")):    
            for j in range(len(track)):
                if j != 0 and track[j][0]!=track[j-1][0]:
                    v_y.append(track[j][1]-track[j-1][1])
                    v_x.append(track[j][2]-track[j-1][2])
        plt.scatter(v_x,v_y,marker = ".",s = 0.001)
        plt.plot([0,0],[-1,1],linewidth = 0.1)
        plt.plot([-1,1],[0,0],linewidth = 0.1)
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        print("图像保存中...")
        plt.savefig("scatter_norm.png", dpi=self.dpi)
        plt.close()

    def draw_scatter_special_norm_v(self,tracks,tracks_tags,order_special):
        """
        绘制特殊轨迹的速度散点图
        - tracks: 轨迹
        - track_tags: 轨迹标签
        - order_special: 特殊轨迹的顺序
        """
        v_x = []
        v_y = []
        v_x_init = []
        v_y_init = []
        v_x_end = []
        v_y_end = []
        v_x_s = []
        v_y_s = []
        for i,track in enumerate(tqdm(tracks,desc = "绘制特殊速率散点图：",unit="条")):
            for j in range(len(track)):
                if j != 0 and track[j][0]!=track[j-1][0]:
                    v_y.append(track[j][1]-track[j-1][1])
                    v_x.append(track[j][2]-track[j-1][2])
                    
                    if tracks_tags["gred_sum"][order_special["gred_sum"][i]][j] == 1 and tracks_tags["gred_sum"][order_special["gred_sum"][i]][j-1] == 0:
                        v_x_init.append(v_x[-1])
                        v_y_init.append(v_y[-1])
                    if tracks_tags["gred_sum"][order_special["gred_sum"][i]][j] == 0 and tracks_tags["gred_sum"][order_special["gred_sum"][i]][j-1] == 1:
                        v_x_end.append(v_x[-1])
                        v_y_end.append(v_y[-1])
                    if tracks_tags["gred_sum"][order_special["gred_sum"][i]][j] == 1 and tracks_tags["gred_sum"][order_special["gred_sum"][i]][j-1] == 1:
                        v_x_s.append(v_x[-1])
                        v_y_s.append(v_y[-1])
        plt.scatter(v_x,v_y,marker = ".",s = 0.001,c= "grey")
        plt.scatter(v_x_init,v_y_init,marker = ".",s = 10,c = "k",alpha = 0.5)
        plt.scatter(v_x_end,v_y_end,marker = ".",s = 10,c = "r",alpha = 0.5)
        plt.scatter(v_x_s,v_y_s,marker = ".",s = 10,c = "b",alpha = 0.5)
        plt.plot([0,0],[-1,1],linewidth = 0.1)
        plt.plot([-1,1],[0,0],linewidth = 0.1)
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        print("图像保存中...")
        plt.savefig("scatter_norm_special_v.png", dpi=self.dpi)
        plt.close()

    def draw_probability_norm_v(self,tracks):

        """
        绘制概率散点图
        - tracks: 轨迹
        """
        v_x = []
        v_y = []
        counter = [0 for i in range(1001)]
        for i,track in enumerate(tqdm(tracks,desc = "绘制概率图：",unit="条")):
            for j in range(len(track)):
                if j != 0 and track[j][0]!= track[j-1][0]:
                    v_y.append(track[j][1]-track[j-1][1])
                    v_x.append(track[j][2]-track[j-1][2])
                    if (v_x[-1]**2+v_y[-1]**2)**0.5 < 1 :
                        counter[round(1000*(v_x[-1]**2+v_y[-1]**2)**0.5)] += 1
        counter = np.array(counter)
        plt.plot([i/1000 for i in range(1001)],counter/np.sum(counter),alpha = 0.5,linewidth = 1)
        x = [i/1000 for i in range(1001)]
        y = counter/np.sum(counter)
        fft_y = np.fft.fft(y) 
        freqs = np.fft.fftfreq(len(x), d=0.001) 
        cutoff_freq =8 
        fft_y_filtered = fft_y.copy()  
        fft_y_filtered[np.abs(freqs) > cutoff_freq] = 0 
        y_filtered = np.fft.ifft(fft_y_filtered)  
        plt.plot(x, y_filtered.real, color='red', linewidth=0.5)
        plt.xlim(0,1)
        plt.ylim(0,0.004)
        print("图像保存中...")
        plt.savefig("probability_norm.png",dpi = self.dpi)
        plt.close()

    def draw_density(self,frames,special,postions):
        pass