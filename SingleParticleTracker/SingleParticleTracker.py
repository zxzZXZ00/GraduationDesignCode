
from matplotlib import pyplot as plt
import numpy as np
import tifffile
from scipy.ndimage import uniform_filter, gaussian_filter, maximum_filter
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from Draw import Draw

class SingleParticleTracker:
    def __init__(self, w=3, r_percent=0.1, sigma=1.0, Ts=2.0, max_step=100.0,r_particle=2.1):
        """
        初始化参数：
        - w: 邻域窗口半径（用于背景去除和最大值检测）
        - r_percent: 强度百分比阈值（保留前r%的候选点）
        - sigma: 高斯滤波的标准差
        - Ts: 非粒子判别的阈值（基于强度矩的聚类）
        - max_step: 轨迹链接的最大允许步长（像素）
        """
        self.w = w
        self.r_percent = r_percent
        self.sigma = sigma
        self.Ts = Ts
        self.max_step = max_step
        self.r_particle = r_particle
    def load_tiff(self, file_paths,limit=None):
        """
        加载TIFF文件并返回图像数据。()
        - file_paths: TIFF文件路径列表
        - limit: 可选的裁剪限制（y_start, y_end, x_start, x_end）
        """
        #print(limit)
        if limit is None:
            y_start = 125
            y_end = 450
            x_start = 10
            x_end = 700
        else:
            y_start = limit["y_start"]
            y_end = limit["y_end"]
            x_start = limit["x_start"]
            x_end = limit["x_end"]
        images = []
        for file_path in tqdm(file_paths, desc="加载TIFF文件", unit="帧"):
            image = tifffile.imread(file_path)[ y_start:y_end, x_start:x_end]
            images.append(image)
        return np.stack(images)
        
    def preprocess_image(self, frames):
        """
        预处理图像，包括高斯滤波和背景去除,返回处理后的图像
        - frames: 输入图像
        """
        fit_frames = []
        for frame in tqdm(frames,desc="滤波去噪：", unit="帧"):
            back_ground = uniform_filter(frame.astype(float), size=2*self.w + 1,mode="constant",cval=0)
            gaussian_result = gaussian_filter(frame - back_ground, sigma=self.sigma,mode="constant", cval=0)
            gaussian_result_0 = np.maximum(gaussian_result, 0)
            fit_frames.append(gaussian_result_0)
        fit_frames = np.array(fit_frames)
        return fit_frames
    def detect_particles(self, frames):
        """
        检测粒子并返回候选点的列表\n
        1.最大值滤波;
        2. 强度阈值筛选;
        3. 合并相邻点;
        - frames: 输入图像
        """
        postions = []
        for frame in tqdm(frames,desc="筛选粒子：", unit="帧"):
            # 最大值滤波;替换为范围内最大值
            max_filtered = maximum_filter(frame, size=2*self.w + 1,mode="constant",cval=0)
            mask = (frame == max_filtered)
            # 筛选强度足够的候选点
            threshold = np.percentile(frame, 100 - self.r_percent)
            candidates = np.argwhere(mask & (frame >= threshold))
            if len(candidates) == 0:
                merged = np.array([])
            # 合并相邻点
            else:
                candidates = candidates.astype(float)
                merged = []
                neighbors = []
                tags = np.zeros(len(candidates))

                for i in range(len(candidates)):
                    neighbors = []
                    neighbors.append(candidates[i])
                    if tags[i] != 0:
                        continue
                    for j in range(i+1,len(candidates)):
                        if tags[j]!= 0:
                            continue
                        else:
                            if (candidates[i][0] - candidates[j][0])**2 + (candidates[i][1] - candidates[j][1])**2 <= self.r_particle**2:
                                neighbors.append(candidates[j])
                                tags[j] = 1
                    merged.append(np.mean(neighbors, axis=0))
                merged = np.array(merged)
            postions.append(merged)
        #print(merged)
        return postions
    def refine_positions(self, frames, postions):
        """
        精确定位：亮度加权质心法
        - frame: 输入图像单帧
        - detections: 候选点列表(np.array)
        """
        #print(postions)
        re_postions = []
        for i in tqdm(range(len(frames)), desc="精确定位：", unit="帧"):
            postion = postions[i]
            frame = frames[i]
            refined = []
            postion = np.array(postion)
            frame = np.array(frame)
            if len(postion) == 0:
                refined = []
            else:
                for y,x in postion:
                    y_min = max(0, round(y - self.w))
                    y_max = min(frame.shape[0], round(y + self.w + 1))
                    x_min = max(0, round(x - self.w))
                    x_max = min(frame.shape[1], round(x + self.w + 1))
                    patch = frame[y_min:y_max, x_min:x_max]
                    # 计算质心偏移
                    y_rel, x_rel = np.indices(patch.shape)
                    y_rel = y_rel - (y - y_min)  # 相对中心坐标
                    x_rel = x_rel - (x - x_min)                    
                    m0 = np.sum(patch)
                    if m0 == 0:
                        return (np.nan, np.nan)  # 无效点                    
                    eps_y = np.sum(y_rel * patch) / m0
                    eps_x = np.sum(x_rel * patch) / m0
                    refined_y = y + eps_y
                    refined_x = x + eps_x
                    if not (np.isnan(refined_y) or np.isnan(refined_x)):
                            refined.append((refined_y, refined_x))
                refined = np.array(refined)
            re_postions.append(refined)
        return re_postions

    def detect_special(self, frames, postions,characteristics ={"gred_sum":1800,"r_special":10}):
        """
        谱聚类
        - frames: 输入图像帧
        - detections: 候选点列表(np.array)
        - characteristics: 特征
          - gred_sum: 灰度和阈值
          - r_spectral: 半径阈值          
        """
        special = {}
        gred_special = self.detect_special_gred_sum(frames,postions,characteristics["gred_sum"])
        if len(gred_special) == 0:
            return np.array([])
        special["gred_sum"] = gred_special
        return special
    
    def link_trajectories(self, all_detections,special = None,tracks_tags = None, order_special = None):
        """
        轨迹链接：基于匈牙利算法的最近邻关联
        - all_detections: 所有帧的检测结果列表np.array([[y,x],[],[]])
        - special: 特殊粒子标记矩阵{"gred_sum":[]}
        - tracks_tags: 特殊粒子标签{"gred_sum":[]}
        - order_special: 记录特殊粒子轨迹的索引{"gred_sum":[]}
        """
        trajectories = []# 存储所有轨迹[[(frame, y, x)]]
        cur_trajectories = []# 当前正在连接的轨迹[[(frame, y, x)]]
        tags = []#[已经未匹配帧数，与cur_trajectories对应]
        tracks_num = []# 轨迹在tracks_tags中的索引
        for frame_idx, detections in enumerate(tqdm(all_detections,desc="轨迹拟合：", unit="帧")):            
            if len(detections) == 0:
                for id,track in enumerate(cur_trajectories):
                    
                    tags[id] += 1
                    if tags[id] >= 5:
                        trajectories.append(track)
                        order_special["gred_sum"].append(tracks_num[id])
                        del cur_trajectories[id]
                        del tags[id]
                        del tracks_num[id]
                    else:
                        cur_trajectories[id].append((cur_trajectories[id][-1][0],cur_trajectories[id][-1][1],cur_trajectories[id][-1][2]))
                        if special is not None:
                            tracks_tags["gred_sum"][tracks_num[id]].append(tracks_tags["gred_sum"][tracks_num[id]][-1])
                        #print("1")

                continue
            if len(cur_trajectories) == 0:
                for idx, (y, x) in enumerate(detections):
                    ####
                    cur_trajectories.append([(frame_idx, y, x)])
                    tracks_num.append(len(cur_trajectories)+len(trajectories)-1)
                   
                    tracks_tags["gred_sum"].append([]) 
                                  
                    if special is not None :
                        if special["gred_sum"][frame_idx][idx] == 1:
                            tracks_tags["gred_sum"][-1].append(1)
                        else:
                            tracks_tags["gred_sum"][-1].append(0)
                    tags.append(0)
                continue
            # 构建成本矩阵（当前轨迹与当前检测点之间的距离）
            cost_matrix = np.full((len(cur_trajectories), len(detections)), np.inf)
            for track_id, track in enumerate(cur_trajectories):
                last_pos = track[-1][1:]  # 上一帧的(y, x)
                for det_idx, (y, x) in enumerate(detections):
                    if np.sqrt((last_pos[0] - y)**2 + (last_pos[1] - x)**2) <= self.max_step:
                        cost_matrix[track_id,det_idx] =np.sqrt((last_pos[0] - y)**2 + (last_pos[1] - x)**2)           
           
            row_ind = []
            col_ind = []
            try:
                large_value = 100
                cost_matrix_modified = np.where(cost_matrix == np.inf, large_value, cost_matrix)
                row_ind_pre, col_ind_pre = linear_sum_assignment(cost_matrix_modified)
                for i, j in zip(row_ind_pre, col_ind_pre):
                    if cost_matrix[i, j] != np.inf:
                        row_ind.append(i)
                        col_ind.append(j)
                for track_id, det_idx in zip(row_ind, col_ind):
                    cur_trajectories[track_id].append((frame_idx, detections[det_idx][0], detections[det_idx][1]))
                    if special is not None :
                        if special["gred_sum"][frame_idx][det_idx] == 1:
                            tracks_tags["gred_sum"][tracks_num[track_id]].append(1)
                        else:
                            tracks_tags["gred_sum"][tracks_num[track_id]].append(0)
                    tags[track_id]=0                    
            except:
                pass
            for id,track in enumerate(cur_trajectories):
                if id not in row_ind:
                    cur_trajectories[id].append((cur_trajectories[id][-1][0], cur_trajectories[id][-1][1], cur_trajectories[id][-1][2]))
                    if special is not None:                           
                            tracks_tags["gred_sum"][tracks_num[id]].append(tracks_tags["gred_sum"][tracks_num[id]][-1])
                    tags[id] += 1
                    if tags[id] >= 5:
                        trajectories.append(track)
                        order_special["gred_sum"].append(tracks_num[id])
                        del cur_trajectories[id]
                        del tags[id]
                        del tracks_num[id]
                        
            for idx, (y, x) in enumerate(detections):

                if idx not in col_ind:
                    cur_trajectories.append([(frame_idx, y, x)])
                    tracks_num.append(len(cur_trajectories)+len(trajectories)-1)
                    tracks_tags["gred_sum"].append([])
                    if special is not None :
                        if special["gred_sum"][frame_idx][idx] == 1:
                            tracks_tags["gred_sum"][-1].append(1)
                        else:
                            tracks_tags["gred_sum"][-1].append(0)
                    tags.append(0)
        trajectories.extend(cur_trajectories)
        order_special["gred_sum"].extend(tracks_num)
        return trajectories
    def run(self,file_paths,limit=None):
        """
        运行整个跟踪过程
        - file_paths: TIFF文件路径列表
        - limit: 可选的裁剪限制{"y_start":, "y_end":, "x_start":, "x_end":}
        """
        frames = []# 图像
        postions = []# 粒子点
        special ={} # 特殊粒子点矩阵
        tracks = []# 轨迹
        track_tags = {"gred_sum":[]}# 轨迹标签
        order_special = {"gred_sum":[]}# 轨迹标签顺序
        pre_frames = self.load_tiff(file_paths,limit)# 加载TIFF文件
        frames = self.preprocess_image(pre_frames)# 预处理图像
        pre_postions = self.detect_particles(frames)# 检测粒子
        postions = self.refine_positions(frames,pre_postions)# 精确定位
        special = self.detect_special(frames,postions)# 特殊粒子检测
        tracks = self.link_trajectories(postions,special,track_tags,order_special)# 轨迹链接
        draw = Draw(dpi=2400)# 轨迹绘制
        draw.draw_run(tracks = tracks,track_tags=track_tags,order_special = order_special,frames = frames,special = special ,postions = postions)# 轨迹绘制
        

        
    def detect_special_gred_sum(self,frames, postions,sum):
        """
        灰度和阈值
        - frame: 输入图像单帧
        - postions: 候选点列表(np.array)
          - gred_sum: 灰度和阈值
        """

        gred_special = []
        gred_specials = []
        for i,frame in enumerate(tqdm(frames,desc="筛选特殊粒子-gred_sum：", unit="帧")):
            gred_special=[]   
            for y,x in postions[i]:
                y_min = max(0, round(y - self.w))
                y_max = min(frame.shape[0], round(y + self.w + 1))
                x_min = max(0,round(x - self.w))
                x_max = min(frame.shape[1], round(x + self.w + 1))
                patch = frame[y_min:y_max, x_min:x_max]
                if np.sum(patch) > 800:
                    #print(np.sum(patch),i,y,x)
                    gred_special.append(1)
                else:
                    gred_special.append(0)
            gred_specials.append(gred_special)
        return gred_specials







            
tracker = SingleParticleTracker(w=3, r_percent=0.1, sigma=1.0, Ts=3.0, max_step=2)
path = []
for i in range(1,10001):
    path.append(f"VimbaImage_{i}.tiff")
tracker.run(path)


