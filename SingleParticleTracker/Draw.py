
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
class Draw:
    def __init__(self,dpi = 400,save_path = "./result/",sample_count = 100  ):
        """
        绘图类
        - dpi: 图像分辨率
        """
        self.dpi = dpi
        self.save_path = save_path
        self.sample_count = sample_count
        Path(self.save_path).mkdir(parents=True, exist_ok=True)

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
        self.draw_probability_norm_speed(tracks)
        self.draw_probability_special_speed(tracks,track_tags,order_special)
        self.draw_probability_special_v_x(tracks,track_tags,order_special)
        self.draw_probability_special_v_y(tracks,track_tags,order_special)
        self.long_track(frames,tracks)
        
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
        plt.axis('off')
        plt.savefig(f"{self.save_path}tracks.png",dpi=self.dpi,bbox_inches='tight',pad_inches=0,transparent=True)
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
        plt.tick_params(axis='both', direction='in', which='both')
        plt.locator_params(axis='x', nbins=5)  # x轴最多显示5个刻度
        plt.locator_params(axis='y', nbins=5)  # y轴最多显示5个刻度
        plt.xlabel("$v_x$")
        plt.ylabel("$v_y$")
        print("图像保存中...")
        plt.savefig(f"{self.save_path}scatter_norm.png",dpi=self.dpi,bbox_inches='tight',pad_inches=0,transparent=True)

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
        print("绘制特殊轨迹的速度散点图:")
        self.get_v(tracks,tracks_tags,order_special,v_x,v_y,v_x_init,v_y_init,v_x_end,v_y_end,v_x_s,v_y_s)
        plt.scatter(v_x,v_y,marker = ".",s = 0.001,c= "grey",label="All Points")
        plt.scatter(v_x_init,v_y_init,marker = ".",s = 1,c = "k",alpha = 0.5,label="Init")
        plt.scatter(v_x_end,v_y_end,marker = ".",s = 1,c = "r",alpha = 0.5,label="End")
        plt.scatter(v_x_s,v_y_s,marker = ".",s = 1,c = "b",alpha = 0.5,label="Process")
        plt.plot([0,0],[-1,1],linewidth = 0.1)
        plt.plot([-1,1],[0,0],linewidth = 0.1)
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        plt.tick_params(axis='both', direction='in', which='both')
        plt.locator_params(axis='x', nbins=5)  # x轴最多显示5个刻度
        plt.locator_params(axis='y', nbins=5)  # y轴最多显示5个刻度
        plt.xlabel("$v_x$")
        plt.ylabel("$v_y$")
        plt.legend(framealpha=0)
        print("图像保存中...")
        plt.savefig(f"{self.save_path}scatter_norm_special_v.png",dpi=self.dpi,bbox_inches='tight',pad_inches=0,transparent=True)
        plt.close()

    def draw_probability_norm_speed(self,tracks):

        """
        绘制概率散点图
        - tracks: 轨迹
        """
        v_x = []
        v_y = []
        counter = [0 for i in range(self.sample_count+1)]
        for i,track in enumerate(tqdm(tracks,desc = "绘制概率图：",unit="条")):
            for j in range(len(track)):
                if j != 0 and track[j][0]!= track[j-1][0]:
                    v_y.append(track[j][1]-track[j-1][1])
                    v_x.append(track[j][2]-track[j-1][2])
                    if (v_x[-1]**2+v_y[-1]**2)**0.5 < 1 :
                        counter[round((self.sample_count)*(v_x[-1]**2+v_y[-1]**2)**0.5)] += 1
        counter = np.array(counter)
        plt.plot([i/self.sample_count for i in range(self.sample_count + 1)],counter/np.sum(counter),alpha = 0.2,linewidth = 1)
        x = [i/self.sample_count for i in range(self.sample_count + 1)]
        y = counter/np.sum(counter)
        params =[]
        y = self.fit_xexpx2(x,y,params)
        plt.plot(x, y, color='red', linewidth=0.5)
        print("图像保存中...")
        plt.tick_params(axis='both', direction='in', which='both')
        plt.locator_params(axis='x', nbins=5)  # x轴最多显示5个刻度
        plt.locator_params(axis='y', nbins=5)  # y轴最多显示5个刻度
        plt.xlabel("$speed$")
        plt.ylabel("$probability$")
        equation = f"$f(x) = {params[0]:.2f}x e^{{-{params[1]:.2f}x^2}}$"
        plt.text(0.95, 0.95, equation, transform=plt.gca().transAxes,fontsize=12, verticalalignment='top', horizontalalignment='right',bbox=dict(facecolor='white', alpha=0, edgecolor='black', boxstyle='round,pad=0.5'))
        plt.savefig(f"{self.save_path}probability_norm.png",dpi=self.dpi,bbox_inches='tight',pad_inches=0,transparent=True)
        plt.close()

    def draw_probability_special_speed(self,tracks,tracks_tags,order_special):
        v_x = []
        v_y = []
        v_x_init = []
        v_y_init = []
        v_x_end = []
        v_y_end = []
        v_x_s = []
        v_y_s = []
        print("绘制特殊粒子速率分布图:")
        self.get_v(tracks,tracks_tags,order_special,v_x,v_y,v_x_init,v_y_init,v_x_end,v_y_end,v_x_s,v_y_s)
        counter_norm = [0 for i in range(self.sample_count+1)]
        counter_init = [0 for i in range(self.sample_count+1)]
        counter_end = [0 for i in range(self.sample_count+1)]
        counter_s = [0 for i in range(self.sample_count+1)]
        x = np.array([i for i in range(self.sample_count + 1)])/self.sample_count
        for i in range(len(v_x)):
            if (v_x[i]**2+v_y[i]**2)**0.5 < 1 :
                counter_norm[round(self.sample_count*(v_x[i]**2+v_y[i]**2)**0.5)] += 1
        counter_norm = np.array(counter_norm)/np.sum(counter_norm)
        params_norm = []
        y_norm = self.fit_xexpx2(x,counter_norm,params_norm)
        for i in range(len(v_x_init)):
            if (v_x_init[i]**2+v_y_init[i]**2)**0.5 < 1 :
                counter_init[round(self.sample_count*(v_x_init[i]**2+v_y_init[i]**2)**0.5)] += 1
        counter_init = np.array(counter_init)
        if np.sum(counter_init) != 0:
            counter_init = np.array(counter_init)/np.sum(counter_init)
        params_init = []
        y_init = self.fit_xexpx2(x,counter_init,params_init)
        for i in range(len(v_x_end)):
            if (v_x_end[i]**2+v_y_end[i]**2)**0.5 < 1 :
                counter_end[round(self.sample_count*(v_x_end[i]**2+v_y_end[i]**2)**0.5)] += 1
        counter_end = np.array(counter_end)
        if np.sum(counter_end) != 0:
            counter_end = np.array(counter_end)/np.sum(counter_end)
        params_end = []
        y_end = self.fit_xexpx2(x,counter_end,params_end)
        for i in range(len(v_x_s)):
            if (v_x_s[i]**2+v_y_s[i]**2)**0.5 < 1 :
                counter_s[round(self.sample_count*(v_x_s[i]**2+v_y_s[i]**2)**0.5)] += 1
        counter_s = np.array(counter_s)
        if np.sum(counter_s) != 0:
            counter_s = np.array(counter_s)/np.sum(counter_s)
        params_s = []
        y_s = self.fit_xexpx2(x,counter_s,params_s)
        plt.plot(x,counter_norm/np.sum(counter_norm),alpha = 0.2,color = "green")
        plt.plot(x,counter_init/np.sum(counter_init),alpha = 0.2,color = "k")
        plt.plot(x,counter_end/np.sum(counter_end),alpha = 0.2,color = "r")
        plt.plot(x,counter_s/np.sum(counter_s),alpha = 0.2, color = "b")
        plt.plot(x,y_norm,alpha = 1,linewidth = 1,label=f"All Points:$y = {params_norm[0]:.2f}x e^{{-{params_norm[1]:.2f}x^2}}$",color = "green")
        plt.plot(x,y_init,alpha = 1,linewidth = 1,label=f"Init:$y = {params_init[0]:.2f}x e^{{-{params_init[1]:.2f}x^2}}$",color = "k")
        plt.plot(x,y_end,alpha = 1,linewidth = 1,label=f"End:$y = {params_end[0]:.2f}x e^{{-{params_end[1]:.2f}x^2}}$",color = "r")
        plt.plot(x,y_s,alpha = 1,linewidth = 1,label=f"Process:$y = {params_s[0]:.2f}x e^{{-{params_s[1]:.2f}x^2}}$", color = "b")
        print("图像保存中...")
        plt.legend()
        plt.tick_params(axis='both', direction='in', which='both')
        plt.locator_params(axis='x', nbins=5)  # x轴最多显示5个刻度
        plt.locator_params(axis='y', nbins=5)  # y轴最多显示5个刻度
        plt.xlabel("$speed$")
        plt.ylabel("$probability$")
        plt.legend(framealpha=0)
        plt.savefig(f"{self.save_path}probability_special_speed.png",dpi=self.dpi,bbox_inches='tight',pad_inches=0,transparent=True)
        plt.close()
    def draw_probability_special_v_x(self,tracks,tracks_tags,order_special):
        """
        绘制特殊轨迹的速度x的概率分布
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
        n = int(2 * self.sample_count/(int(10)))#!!!!!
        n_norm = 2 * self.sample_count
        x_norm = np.array([i-1 for i in range(n_norm + 1)])/n_norm*2-1
        x = np.array([i-1 for i in range(n + 1)])/n*2-1
        print("绘制特殊粒子速度分布图（x）:")
        self.get_v(tracks,tracks_tags,order_special,v_x=v_x,v_y=v_y,v_x_init=v_x_init,v_y_init=v_y_init,v_x_end=v_x_end,v_y_end=v_y_end,v_x_s=v_x_s,v_y_s=v_y_s)
        counter_norm = [0 for i in range(n_norm+1)]
        counter_init = [0 for i in range(n+1)]
        counter_end = [0 for i in range(n+1)]
        counter_s = [0 for i in range(n+1)]
        for i in range(len(v_x)):
            if abs(v_x[i]) < 1 :
                counter_norm[round(n_norm/2*v_x[i] + n_norm/2)] += 1  
        counter_norm = np.array(counter_norm)/np.sum(counter_norm)
        params_norm = []
        y_norm = self.fit_expx2(x_norm,counter_norm,params_norm)
        for i in range(len(v_x_init)):
            if abs(v_x_init[i]) < 1 :
                counter_init[round(n/2*v_x_init[i] + n/2)] += 1
        counter_init = np.array(counter_init)
        if np.sum(counter_init) != 0:
            counter_init = np.array(counter_init)/np.sum(counter_init)
        params_init = []
        y_init = self.fit_expx2(x,counter_init,params_init)
        for i in range(len(v_x_end)):
            if abs(v_x_end[i]) < 1 :
                counter_end[round(n/2*v_x_end[i] + n/2)] += 1
        counter_end = np.array(counter_end)
        if np.sum(counter_end) != 0:
            counter_end = np.array(counter_end)/np.sum(counter_end)
        params_end = []
        y_end = self.fit_expx2(x,counter_end,params_end)
        for i in range(len(v_x_s)):
            if abs(v_x_s[i]) < 1 :
                counter_s[round(n/2*v_x_s[i] + n/2)] += 1
        counter_s = np.array(counter_s)
        if np.sum(counter_s) != 0:
            counter_s = np.array(counter_s)/np.sum(counter_s)
        params_s = []
        y_s = self.fit_expx2(x,counter_s,params_s)
        plt.plot(x_norm,10*counter_norm/np.sum(counter_norm),alpha = 0.2,color = "green")
        plt.plot(x,counter_init/np.sum(counter_init),alpha = 0.2,color = "k")
        plt.plot(x,counter_end/np.sum(counter_end),alpha = 0.2,color = "r")
        plt.plot(x,counter_s/np.sum(counter_s),alpha = 0.2, color = "b")
        plt.plot(x_norm,10*y_norm,alpha = 1,linewidth = 1,label=f"All Points:$y = {params_norm[0]:.2f} e^{{-({params_norm[1]:.2f}x-{params_norm[2]:.2f})^2}}$",color = "green")
        plt.plot(x_norm,y_init,alpha = 1,linewidth = 1,label=f"Init:$y = {params_init[0]:.2f} e^{{-({params_init[1]:.2f}x-{params_init[2]:.2f})^2}}$",color = "k")
        plt.plot(x_norm,y_end,alpha = 1,linewidth = 1,label=f"End:$y = {params_end[0]:.2f} e^{{-({params_end[1]:.2f}x-{params_end[2]:.2f})^2}}$",color = "r")
        plt.plot(x_norm,y_s,alpha = 1,linewidth = 1,label=f"Process:$y = {params_s[0]:.2f} e^{{-({params_s[1]:.2f}x-{params_s[2]:.2f})^2}}$", color = "b")
        print("图像保存中...")
        plt.legend()
        plt.tick_params(axis='both', direction='in', which='both')
        plt.locator_params(axis='x', nbins=5)  # x轴最多显示5个刻度
        plt.locator_params(axis='y', nbins=5)  # y轴最多显示5个刻度
        plt.xlabel("$v_x$")
        plt.ylabel("$probability$")
        plt.legend(framealpha=0)
        plt.savefig(f"{self.save_path}probability_special_v_x.png",dpi=self.dpi,bbox_inches='tight',pad_inches=0,transparent=True)
        plt.close()
        
    def draw_probability_special_v_y(self,tracks,tracks_tags,order_special):
        """
        绘制特殊轨迹的速度y的概率分布
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
        n_norm = 2 * self.sample_count
        n =  int(2 * self.sample_count/(int(10)))
        x_norm = np.array([i-1 for i in range(n_norm + 1)])/n_norm*2-1
        2 * self.sample_count/(int(10))#!!!!!
        x = np.array([i for i in range(n + 1)])/n*2-1
        print("绘制特殊粒子速度分布图（y）:")
        self.get_v(tracks,tracks_tags,order_special,v_x=v_x,v_y=v_y,v_x_init=v_x_init,v_y_init=v_y_init,v_x_end=v_x_end,v_y_end=v_y_end,v_x_s=v_x_s,v_y_s=v_y_s)
        counter_norm = [0 for i in range(n_norm+1)]
        counter_init = [0 for i in range(n+1)]
        counter_end = [0 for i in range(n+1)]
        counter_s = [0 for i in range(n+1)]
        for i in range(len(v_y)):
            if abs(v_y[i]) < 1 :
                counter_norm[round(n_norm/2*v_y[i] + n_norm/2)] += 1
        counter_norm = np.array(counter_norm)
        counter_norm = np.array(counter_norm)/np.sum(counter_norm)
        params_norm = []
        y_norm = self.fit_expx2(x_norm,counter_norm,params_norm)
        for i in range(len(v_y_init)):
            if abs(v_y_init[i]) < 1 :
                counter_init[round(n/2*v_y_init[i] + n/2)] += 1
        counter_init = np.array(counter_init)
        if np.sum(counter_init) != 0:
            counter_init = np.array(counter_init)/np.sum(counter_init)
        params_init = []
        y_init = self.fit_expx2(x,counter_init,params_init)
        for i in range(len(v_y_end)):
            if abs(v_y_end[i]) < 1 :
                counter_end[round(n/2*v_y_end[i] + n/2)] += 1
        counter_end = np.array(counter_end)
        if np.sum(counter_end) != 0:
            counter_end = np.array(counter_end)/np.sum(counter_end)
        params_end = []
        y_end = self.fit_expx2(x,counter_end,params_end)
        for i in range(len(v_y_s)):
            if abs(v_y_s[i]) < 1 :
                counter_s[round(n/2*v_y_s[i] + n/2)] += 1
        counter_s = np.array(counter_s)
        if np.sum(counter_s) != 0:
            counter_s = np.array(counter_s)/np.sum(counter_s)
        params_s = []
        y_s = self.fit_expx2(x,counter_s,params_s)
   
        plt.plot(x_norm,10*counter_norm/np.sum(counter_norm),alpha = 0.2,color = "green")
        plt.plot(x,counter_init/np.sum(counter_init),alpha = 0.2,color = "k")
        plt.plot(x,counter_end/np.sum(counter_end),alpha = 0.2,color = "r")
        plt.plot(x,counter_s/np.sum(counter_s),alpha = 0.2, color = "b")
        plt.plot(x_norm,10*y_norm,alpha = 1,linewidth = 1,label=f"All Points:$y = {params_norm[0]:.2f} e^{{-({params_norm[1]:.2f}x-{params_norm[2]:.2f})^2}}$",color = "green")
        plt.plot(x_norm,y_init,alpha = 1,linewidth = 1,label=f"Init:$y = {params_init[0]:.2f} e^{{-({params_init[1]:.2f}x-{params_init[2]:.2f})^2}}$",color = "k")
        plt.plot(x_norm,y_end,alpha = 1,linewidth = 1,label=f"End:$y = {params_end[0]:.2f} e^{{-({params_end[1]:.2f}x-{params_end[2]:.2f})^2}}$",color = "r")
        plt.plot(x_norm,y_s,alpha = 1,linewidth = 1,label=f"Process:$y = {params_s[0]:.2f} e^{{-({params_s[1]:.2f}x-{params_s[2]:.2f})^2}}$", color = "b")
        print("图像保存中...")
        plt.legend()
        plt.tick_params(axis='both', direction='in', which='both')
        plt.locator_params(axis='x', nbins=5)  # x轴最多显示5个刻度
        plt.locator_params(axis='y', nbins=5)  # y轴最多显示5个刻度
        plt.xlabel("$v_y$")
        plt.ylabel("$probability$")
        plt.legend(framealpha=0)
        plt.savefig(f"{self.save_path}probability_special_v_y.png",dpi=self.dpi,bbox_inches='tight',pad_inches=0,transparent=True)
        plt.close()

    def long_track(self,frames,tracks):
        """
        绘制最长单粒子轨迹
        - frames: 帧
        - tracks: 轨迹
        """
        max_len = 0
        load_max = 0
        for i,track in enumerate(tqdm(tracks,desc = "绘制最长单粒子轨迹：",unit="条")):
            if max_len<len(track) and track[0][1]>100 and track[0][1]<len(frames[0])-100 and track[0][2]>100 and track[0][2]<len(frames[0][0])-100:
                load_max = i
                max_len = len(track)
        x = [x for i,y,x in tracks[load_max]]
        y = [y for i,y,x in tracks[load_max]]
        size = 10
        #print(tracks[load_max][0])
        frame = frames[tracks[load_max][0][0]][round(tracks[load_max][0][1]-size):round(tracks[load_max][0][1]+size),round(tracks[load_max][0][2]-size):round(tracks[load_max][0][2]+size)]
        plt.imshow(frame,cmap='gray')
        plt.plot(np.array(x)-round(tracks[load_max][0][2]-size),np.array(y)-round(tracks[load_max][0][1]-size),linewidth = 0.1)
        plt.savefig("long_track.png",dpi = self.dpi)
        plt.close()
    def get_v(self,tracks,tracks_tags,order_special,v_x = [],v_y = [],v_x_init = [],v_y_init = [],v_x_end = [],v_y_end = [],v_x_s = [],v_y_s = []):
        """
        获取速度，去掉前后相同的点
        - tracks: 轨迹
        - track_tags: 轨迹标签
        - order_special: 特殊轨迹的顺序
        - v_x: 速度x
        - v_y: 速度y
        - v_x_init: 特殊轨迹的初始速度x
        - v_y_init: 特殊轨迹的初始速度y
        - v_x_end: 特殊轨迹的结束速度x
        - v_y_end: 特殊轨迹的结束速度y
        - v_x_s: 特殊轨迹的过程速度x
        - v_y_s: 特殊轨迹的过程速度y
        """
        for i,track in enumerate(tqdm(tracks,desc = "速度计算：",unit="条")):
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

    def fit_xexpx2(self,x,y,param = None):
        """
        用 x * exp(-x^2) 拟合数据
        - x: 自变量数据
        - y: 因变量数据
        """
        x = np.array(x)
        y = np.array(y)
        def model(x, a, b):
            return a * x * np.exp(-b * x**2)
        """
        使用模型 f(x) = a * x * exp(-b * x^2) 拟合数据 (x, y)
        - x: 自变量数据
        - y: 因变量数据
        """
        x = np.array(x)
        y = np.array(y)

        params, covariance = curve_fit(model, x, y)
        a, b = params
        param.append(a)
        param.append(b)
        y_fit = model(x, a, b)
        return y_fit
    def fit_expx2(self,x,y,param = None):
        """
        用exp(-x^2) 拟合数据
        - x: 自变量数据
        - y: 因变量数据
        """
        x = np.array(x)
        y = np.array(y)
        p0 = [0.5,20,0]
        def model(x, a , b,c):
            return a * np.exp(-b * (x-c)**2)
        """
        使用模型 f(x) = a  * exp(-b * x^2) 拟合数据 (x, y)
        - x: 自变量数据
        - y: 因变量数据
        """
        x = np.array(x)
        y = np.array(y)
        try:
            params, covariance = curve_fit(model, x, y,p0=p0)
            a,b,c = params
        except:
            a = b =c =0
        
        param.append(a)
        param.append(b)
        param.append(c)
        n_norm = 2 * self.sample_count
        x_norm = np.array([i-1 for i in range(n_norm + 1)])/n_norm*2-1
        y_fit = model(x_norm, a , b,c)
        return y_fit

    