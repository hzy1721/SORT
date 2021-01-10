"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
import glob
import time
import argparse
from filterpy.kalman import KalmanFilter

matplotlib.use('TkAgg')

np.random.seed(0)


def linear_assignment(cost_matrix):
    """用于求解指派问题的匈牙利算法"""
    try:
        # 尝试使用lap包
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        # 否则使用scipy.optimize
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    计算两个边界框的IoU。边界框格式：[x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    # 计算交集部分的两个顶点坐标
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    # 交集部分的宽高
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    # 交集的面积
    wh = w * h
    # 交集面积 / 并集面积
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return o


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    边界框格式转换：[x1,y1,x2,y2] -> [x,y,s,r]
    """
    # 宽高
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    # 中心点坐标
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    # 面积
    s = w * h  # scale is just area
    # 宽 / 高：宽高比
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    边界框格式转换：[x,y,s,r] -> [x1,y1,x2,y2]
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        # 未提供score，返回(1,4)形状的数组
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        # 如果提供了score，返回(1,5)形状的数组
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


class KalmanBoxTracker(object):
    """使用卡尔曼滤波器预测边界框的下一帧位置。
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        # 观测噪声协方差矩阵R
        self.kf.R[2:, 2:] *= 10.
        # 后验估计误差的协方差
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        # 过程激励噪声协方差矩阵Q
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        # 边界框
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        # 上次更新后经过的时间
        self.time_since_update = 0
        # 已跟踪目标的ID
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        # 历史
        self.history = []
        #
        self.hits = 0
        #
        self.hit_streak = 0
        # 持续帧数
        self.age = 0

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        用当前观测值计算当前估计值。
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        返回当前预测值。
        """
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        返回后验估计。
        """
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    将检测结果关联到目标。返回匹配成功的索引对、未匹配的检测结果索引、未匹配的目标列表。
    参数：
        detections: 检测结果列表
        trackers: 目标列表
        iou_threshold: 检测结果与已跟踪目标匹配的IoU阈值
    """
    # 没有已跟踪的目标
    if len(trackers) == 0:
        # (0,2) (len(detections)) (0,5)
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    # 两两计算检测结果和跟踪目标的IoU
    iou_matrix = iou_batch(detections, trackers)

    # IoU结果不为空，也就是至少有两个bbox的IoU>0
    if min(iou_matrix.shape) > 0:
        # 取出
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    # 未匹配的检测结果
    unmatched_detections = []
    # 遍历检测结果
    for d, det in enumerate(detections):
        # 如果不在匹配成功列表里，加入未匹配列表
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    # 未匹配的已跟踪目标
    unmatched_trackers = []
    # 遍历已跟踪目标
    for t, trk in enumerate(trackers):
        # 如果不在匹配成功列表里，加入未匹配列表
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    # 过滤后的匹配成功索引对
    matches = []
    # 遍历匹配成功的索引对
    for m in matched_indices:
        # 如果IoU小于阈值，取消匹配
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        # 加入返回列表
        else:
            matches.append(m.reshape(1, 2))
    # 转换为适当的形状，便于返回
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        # 已跟踪的目标列表
        self.trackers = []
        # 当前帧序号
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        # 已跟踪目标的当前帧预测值
        trks = np.zeros((len(self.trackers), 5))
        # 待删除列表
        to_del = []
        #
        ret = []
        for t, trk in enumerate(trks):
            # 预测位置
            pos = self.trackers[t].predict()[0]
            # 初始化速度为0
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            # 如果预测值中有NaN，则删除
            if np.any(np.isnan(pos)):
                to_del.append(t)
        # 去掉含无效值的预测值
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        # 删除已跟踪目标
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    # 是否实时演示结果
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',
                        action='store_true')
    # 存放检测结果的路径，默认是data
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    # 阶段，默认是train
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    # N_lost，默认是1
    parser.add_argument("--max_age",
                        help="Maximum number of frames to keep alive a track without associated detections.",
                        type=int, default=1)
    # "试用期"，默认是3
    parser.add_argument("--min_hits",
                        help="Minimum number of associated detections before track is initialised.",
                        type=int, default=3)
    # IoU阈值，默认是0.3
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # 获取命令行参数
    args = parse_args()
    # 是否演示
    display = args.display
    # 阶段
    phase = args.phase
    # 花费的总时间
    total_time = 0.0
    # 总帧数
    total_frames = 0
    # 生成32个随机颜色，用于bbox
    colours = np.random.rand(32, 3)  # used only for display
    if display:
        # 如果要展示必须有原始图片
        if not os.path.exists('mot_benchmark'):
            print('\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n    (https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n    $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
            exit()
        # 打开交互模式
        plt.ion()
        # 获取图像
        fig = plt.figure()
        # 添加子图，1行1列第1个
        ax1 = fig.add_subplot(111, aspect='equal')
    # 创建结果目录
    if not os.path.exists('output'):
        os.makedirs('output')
    # 匹配检测结果文件det.txt
    pattern = os.path.join(args.seq_path, phase, '*', 'det', 'det.txt')
    # 遍历每个任务
    for seq_dets_fn in glob.glob(pattern):
        # 实例化SORT tracker
        mot_tracker = Sort(max_age=args.max_age,
                           min_hits=args.min_hits,
                           iou_threshold=args.iou_threshold)  # create instance of the SORT tracker
        # 从txt文件中加载检测结果
        seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
        # 任务名称
        seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0]
        # 打开输出结果文件
        with open(os.path.join('output', '%s.txt' % (seq)), 'w') as out_file:
            print("Processing %s." % (seq))
            # 遍历每一帧
            for frame in range(int(seq_dets[:, 0].max())):
                # 帧数从1开始，而不是从0开始
                frame += 1
                # 取出每一帧的检测结果 [x1,y1,w,h]
                dets = seq_dets[seq_dets[:, 0] == frame, 2:7]
                # [x1,y1,w,h] -> [x1,y1,x2,y2]
                dets[:, 2:4] += dets[:, 0:2]
                # 总帧数+1
                total_frames += 1

                if display:
                    # 帧图片路径
                    fn = os.path.join('mot_benchmark', phase, seq, 'img1', '%06d.jpg' % (frame))
                    # 读取图片
                    im = io.imread(fn)
                    # 显示在子图中
                    ax1.imshow(im)
                    # 添加标题
                    plt.title(seq + ' Tracked Targets')

                # 开始处理的时间
                start_time = time.time()
                # 开始跟踪
                trackers = mot_tracker.update(dets)
                # 处理当前帧花费的时间
                cycle_time = time.time() - start_time
                # 累加到总时间中
                total_time += cycle_time

                # 输出当前帧的已跟踪目标
                for d in trackers:
                    # 打印结果到输出文件中：frame,-1,x1,y1,w,h,1,-1,-1,-1
                    print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (frame, d[4], d[0], d[1], d[2] - d[0], d[3] - d[1]),
                          file=out_file)
                    # 绘制边界框
                    if display:
                        d = d.astype(np.int32)
                        ax1.add_patch(patches.Rectangle((d[0], d[1]), d[2] - d[0], d[3] - d[1], fill=False, lw=3,
                                                        ec=colours[d[4] % 32, :]))
                if display:
                    # 更新图
                    fig.canvas.flush_events()
                    plt.draw()
                    ax1.cla()
    # 打印花费的运行时间、总帧数、FPS
    print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (
            total_time, total_frames, total_frames / total_time))
    # 提示：使用--display选项无法得到真实的运行时间
    if display:
        print("Note: to get real runtime results run without the option: --display")
