"""
CRL关键时间段检测 + 视频导出（优化版）
使用多线程预加载，提高GPU利用率
"""
import cv2
import numpy as np
import torch
from PIL import Image
import clip
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
from crl_config import *
import threading
from queue import Queue

# ============ 关键时间段配置 ============
KEY_MOMENT_WINDOW = 0.1  # 关键帧前后n秒
OUTPUT_VIDEO_PATH = "output_keymoments.mp4"
PREFETCH_BATCHES = 5  # 预加载的batch数量
USE_HARDWARE_CODEC = True  # 使用硬件编解码器

# 输出模式选择
EXPORT_MODE = "highlights_only"  # "full" - 完整视频带OSD, "highlights_only" - 仅关键时间段
OUTPUT_HIGHLIGHTS_PATH = "output_highlights.mp4"  # 仅关键时间段的输出路径

# 编码器说明：
# XVID - 快速，质量好，推荐
# MJPG - 最快，文件大
# mp4v - 兼容性最好
# ======================================

print_config()

# 加载CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nLoading CLIP model: {CLIP_MODEL} on {device}...")
model, preprocess = clip.load(CLIP_MODEL, device=device)

# === 基底构建 ===
print(f"\n=== Basis Construction ===")
basis_dict = BASIS_TEXTS[CRITERION]
basis_names = list(basis_dict.keys())

basis_vectors = []
with torch.no_grad():
    for name, texts in basis_dict.items():
        text_tokens = clip.tokenize(texts).to(device)
        text_features = model.encode_text(text_tokens).cpu().numpy()
        text_features = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)
        basis_vector = np.mean(text_features, axis=0)
        basis_vector = basis_vector / np.linalg.norm(basis_vector)
        basis_vectors.append(basis_vector)

B = np.array(basis_vectors).T
print(f"Basis B: {B.shape}")

# === 多线程数据加载器 ===
class VideoLoader:
    def __init__(self, video_path, sample_rate, batch_size, preprocess_fn):
        self.video_path = video_path
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.preprocess_fn = preprocess_fn
        self.queue = Queue(maxsize=PREFETCH_BATCHES)
        self.stop_flag = False
        
    def _load_worker(self):
        """后台线程：读取视频并预处理"""
        cap = cv2.VideoCapture(self.video_path, cv2.CAP_FFMPEG)
        
        # 优化视频解码
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 最小缓冲
        
        # 使用硬件加速
        if USE_HARDWARE_CODEC:
            try:
                cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
                cap.set(cv2.CAP_PROP_HW_DEVICE, 0)  # 使用第一个硬件设备
            except:
                pass
        
        frame_count = 0
        batch_buffer = []
        
        while not self.stop_flag:
            ret, frame = cap.read()
            if not ret:
                # 处理剩余batch
                if len(batch_buffer) > 0:
                    batch_tensor = torch.stack(batch_buffer)
                    self.queue.put(batch_tensor)
                self.queue.put(None)  # 结束信号
                break
            
            if frame_count % self.sample_rate == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensor = self.preprocess_fn(Image.fromarray(frame_rgb))
                batch_buffer.append(frame_tensor)
                
                if len(batch_buffer) >= self.batch_size:
                    batch_tensor = torch.stack(batch_buffer)
                    self.queue.put(batch_tensor)
                    batch_buffer = []
            
            frame_count += 1
        
        cap.release()
    
    def start(self):
        """启动后台加载线程"""
        self.thread = threading.Thread(target=self._load_worker, daemon=True)
        self.thread.start()
    
    def get_batch(self):
        """获取一个batch（阻塞）"""
        return self.queue.get()
    
    def stop(self):
        """停止加载"""
        self.stop_flag = True
        self.thread.join()

# === 优化OpenCV设置 ===
# 设置OpenCV线程数
cv2.setNumThreads(8)  # 增加解码线程数

# === 第一遍：编码所有帧（使用多线程加载）===
print(f"\n=== Pass 1: Encoding frames (with prefetching) ===")
cap = cv2.VideoCapture(VIDEO_PATH, cv2.CAP_FFMPEG)

# 优化视频读取
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if USE_HARDWARE_CODEC:
    try:
        cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
        cap.set(cv2.CAP_PROP_HW_DEVICE, 0)
    except:
        pass

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

print(f"Video: {fps:.2f} fps, {total_frames} frames, {width}x{height}")

# 启动数据加载器
loader = VideoLoader(VIDEO_PATH, SAMPLE_RATE, BATCH_SIZE, preprocess)
loader.start()

z_conds = []
torch.backends.cudnn.benchmark = True

num_sampled = (total_frames + SAMPLE_RATE - 1) // SAMPLE_RATE
pbar = tqdm(total=num_sampled, desc="Encoding", unit="batch", smoothing=0.1)

with torch.no_grad():
    while True:
        batch_images = loader.get_batch()
        if batch_images is None:  # 结束信号
            break
        
        # GPU推理
        batch_images = batch_images.to(device, non_blocking=True)
        z_uni_batch = model.encode_image(batch_images).cpu().numpy()
        z_uni_batch = z_uni_batch / np.linalg.norm(z_uni_batch, axis=1, keepdims=True)
        z_cond_batch = np.dot(z_uni_batch, B)
        z_conds.append(z_cond_batch)
        
        pbar.update(len(batch_images))

pbar.close()
loader.stop()

z_conds = np.vstack(z_conds).astype(np.float64)
print(f"Encoded {len(z_conds)} frames")

# === 滤波和差分 ===
print("\n=== Computing key moments ===")
z_conds_smooth = np.zeros_like(z_conds)
for i in range(z_conds.shape[1]):
    z_conds_smooth[:, i] = gaussian_filter1d(z_conds[:, i], sigma=SMOOTH_SIGMA)

z_conds_diff = np.zeros_like(z_conds_smooth)
for i in range(z_conds_smooth.shape[1]):
    z_conds_diff[1:, i] = np.diff(z_conds_smooth[:, i])
    z_conds_diff[0, i] = z_conds_diff[1, i]

z_conds_change = np.abs(z_conds_diff)

# === 检测关键时间段（加权算法）===
key_moments = {}

print(f"Using weighted detection: score={SCORE_WEIGHT:.1f}, change={CHANGE_WEIGHT:.1f}, min_score={MIN_SCORE_THRESHOLD:.2f}")

for dim_idx, name in enumerate(basis_names):
    scores = z_conds_smooth[:, dim_idx]
    change_rates = z_conds_change[:, dim_idx]
    
    # 归一化到[0, 1]
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    change_norm = (change_rates - change_rates.min()) / (change_rates.max() - change_rates.min() + 1e-8)
    
    # 加权综合得分
    combined_score = SCORE_WEIGHT * scores_norm + CHANGE_WEIGHT * change_norm
    
    # 阈值检测
    threshold = np.percentile(combined_score, CHANGE_THRESHOLD_PERCENTILE)
    
    # 同时满足：综合得分高 且 原始相似度不太低
    keyframe_candidates = np.where(
        (combined_score > threshold) & 
        (scores > MIN_SCORE_THRESHOLD)
    )[0]
    
    # 扩展到前后时间窗口
    window_frames = int(fps * KEY_MOMENT_WINDOW / SAMPLE_RATE)
    key_moment_frames = set()
    
    for kf_idx in keyframe_candidates:
        for offset in range(-window_frames, window_frames + 1):
            frame_idx = kf_idx + offset
            if 0 <= frame_idx < len(z_conds):
                key_moment_frames.add(frame_idx)
    
    key_moments[name] = key_moment_frames
    print(f"  {name}: {len(keyframe_candidates)} keyframes → {len(key_moment_frames)} key moment frames")
    print(f"    Score range: [{scores.min():.3f}, {scores.max():.3f}]")

# === 第二遍：生成视频 ===
if EXPORT_MODE == "highlights_only":
    print(f"\n=== Pass 2: Exporting highlights only ===")
    # 合并所有维度的关键时间段
    all_key_moments = set()
    for frames_set in key_moments.values():
        all_key_moments.update(frames_set)
    all_key_moments = sorted(all_key_moments)
    print(f"Total key moment frames: {len(all_key_moments)}")
else:
    print(f"\n=== Pass 2: Generating full video with OSD ===")

cap = cv2.VideoCapture(VIDEO_PATH, cv2.CAP_FFMPEG)

# 优化视频读取
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if USE_HARDWARE_CODEC:
    try:
        cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
        cap.set(cv2.CAP_PROP_HW_DEVICE, 0)
    except:
        pass

# 使用更快的编码器（按优先级尝试）
fourcc_options = [
    ('XVID', 'XVID (fast, good quality)'),  # Xvid - 通常可用
    ('MJPG', 'MJPEG (very fast, large file)'),  # Motion JPEG - 最快
    ('mp4v', 'MPEG-4 (default)'),  # MPEG-4 - 默认
]

# 选择输出路径
output_path = OUTPUT_HIGHLIGHTS_PATH if EXPORT_MODE == "highlights_only" else OUTPUT_VIDEO_PATH

# 尝试不同的编码器
out = None
print("Trying video codecs...")
for codec_str, desc in fourcc_options:
    try:
        fourcc = cv2.VideoWriter_fourcc(*codec_str)
        test_out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if test_out.isOpened():
            out = test_out
            print(f"✓ Using codec: {codec_str} - {desc}")
            break
        test_out.release()
    except Exception as e:
        print(f"✗ {codec_str} failed: {e}")

if not out or not out.isOpened():
    print("Warning: All codecs failed, using uncompressed AVI")
    output_path = output_path.replace('.mp4', '.avi')
    fourcc = 0
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

def draw_keymoment_osd(frame, sampled_idx, key_moments, basis_names):
    """绘制醒目的关键时间段OSD"""
    h, w = frame.shape[:2]
    
    # 检查是否有任何关键时刻
    any_key_moment = any(sampled_idx in key_moments[name] for name in basis_names)
    
    if any_key_moment:
        # === 醒目的全屏边框 ===
        border_thickness = 8
        border_color = (0, 255, 0)  # 绿色
        
        # 绘制边框
        cv2.rectangle(frame, (0, 0), (w-1, h-1), border_color, border_thickness)
        
        # === 左上角大标签 ===
        for i, name in enumerate(basis_names):
            if sampled_idx in key_moments[name]:
                # 半透明背景
                overlay = frame.copy()
                label_height = 60
                label_width = 250
                cv2.rectangle(overlay, (10, 10), (10 + label_width, 10 + label_height), 
                            (0, 200, 0), -1)
                frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
                
                # 文字
                cv2.putText(frame, "CHAIR DETECTED", (20, 40),
                           cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, "Key Moment", (20, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # 闪烁圆点
                cv2.circle(frame, (label_width - 20, 35), 10, (255, 255, 255), -1)
                break
    else:
        # === 普通状态：右上角小指示器 ===
        indicator_width = 150
        indicator_height = 35
        start_x = w - indicator_width - 10
        start_y = 10
        
        # 半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (start_x, start_y), 
                     (start_x + indicator_width, start_y + indicator_height),
                     (50, 50, 50), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # 文字
        cv2.putText(frame, "Scanning...", (start_x + 10, start_y + 22),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return frame

frame_count = 0
sampled_frame_idx = 0
written_frames = 0

if EXPORT_MODE == "highlights_only":
    pbar = tqdm(total=total_frames, desc="Exporting highlights", unit="frame", smoothing=0.1)
else:
    pbar = tqdm(total=total_frames, desc="Rendering", unit="frame", smoothing=0.1)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    if EXPORT_MODE == "highlights_only":
        # 仅输出关键时间段
        if frame_count % SAMPLE_RATE == 0:
            if sampled_frame_idx in all_key_moments:
                # 添加OSD标记
                frame = draw_keymoment_osd(frame, sampled_frame_idx, key_moments, basis_names)
                out.write(frame)
                written_frames += 1
            sampled_frame_idx += 1
    else:
        # 输出完整视频带OSD
        if frame_count % SAMPLE_RATE == 0:
            frame = draw_keymoment_osd(frame, sampled_frame_idx, key_moments, basis_names)
            sampled_frame_idx += 1
        out.write(frame)
        written_frames += 1
    
    frame_count += 1
    pbar.update(1)

pbar.close()
cap.release()
out.release()

print(f"\n=== Done! ===")
print(f"Output: {output_path}")
print(f"Written frames: {written_frames}/{total_frames} ({written_frames/total_frames*100:.1f}%)")
print(f"\nKey moment summary:")
for name, frames in key_moments.items():
    duration = len(frames) * SAMPLE_RATE / fps
    print(f"  {name}: {duration:.1f}s ({len(frames)} frames)")
