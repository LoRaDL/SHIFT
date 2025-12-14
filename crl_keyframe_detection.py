"""
CRL关键帧检测 - 基于变化率识别关键帧
使用时间差分检测CRL得分的显著变化
"""
import cv2
import numpy as np
import torch
from PIL import Image
import clip
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# ============ 导入配置 ============
from crl_config import *

print_config()

# 加载CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading CLIP model: {CLIP_MODEL} on {device}...")
model, preprocess = clip.load(CLIP_MODEL, device=device)

# === 基底构建 ===
print(f"\n=== Basis Construction: '{CRITERION}' ===")
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
        print(f"  {name}: {len(texts)} texts")

B = np.array(basis_vectors).T
print(f"Basis B: {B.shape}")

# === 流式读取和编码（不保存帧到内存）===
from tqdm import tqdm

print(f"\nProcessing video: {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)

# 获取视频信息
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
num_sampled_frames = (total_frames_in_video + SAMPLE_RATE - 1) // SAMPLE_RATE
print(f"Video: {fps:.2f} fps, {total_frames_in_video} frames")
print(f"Will sample ~{num_sampled_frames} frames (every {SAMPLE_RATE} frames)")

print(f"\nEncoding with CRL (batch_size={BATCH_SIZE})...")
z_conds = []
frame_count = 0
batch_buffer = []

# 使用pin_memory加速CPU到GPU传输
torch.backends.cudnn.benchmark = True

with torch.no_grad():
    pbar = tqdm(total=total_frames_in_video, desc="Encoding", unit="frame", smoothing=0.1)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 采样
        if frame_count % SAMPLE_RATE == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch_buffer.append(frame_rgb)
            
            # 当批次满了，进行编码
            if len(batch_buffer) >= BATCH_SIZE:
                # 预处理批次（在CPU上）
                batch_images = torch.stack([
                    preprocess(Image.fromarray(f)) 
                    for f in batch_buffer
                ])
                
                # 异步传输到GPU
                batch_images = batch_images.to(device, non_blocking=True)
                
                # 批量编码
                z_uni_batch = model.encode_image(batch_images)
                
                # 同步并转到CPU
                z_uni_batch = z_uni_batch.cpu().numpy()
                z_uni_batch = z_uni_batch / np.linalg.norm(z_uni_batch, axis=1, keepdims=True)
                
                # 批量投影
                z_cond_batch = np.dot(z_uni_batch, B)
                z_conds.append(z_cond_batch)
                
                # 清空buffer释放内存
                batch_buffer = []
        
        frame_count += 1
        pbar.update(1)
    
    pbar.close()
    
    # 处理剩余的帧
    if len(batch_buffer) > 0:
        print(f"Processing final {len(batch_buffer)} frames...")
        batch_images = torch.stack([
            preprocess(Image.fromarray(f)) 
            for f in batch_buffer
        ]).to(device, non_blocking=True)
        
        z_uni_batch = model.encode_image(batch_images).cpu().numpy()
        z_uni_batch = z_uni_batch / np.linalg.norm(z_uni_batch, axis=1, keepdims=True)
        z_cond_batch = np.dot(z_uni_batch, B)
        z_conds.append(z_cond_batch)

cap.release()
z_conds = np.vstack(z_conds).astype(np.float64)
num_encoded_frames = len(z_conds)

print(f"Encoded {num_encoded_frames} frames, shape: {z_conds.shape}")

# === 滤波 ===
print(f"\nApplying Gaussian filter (σ={SMOOTH_SIGMA})...")
z_conds_smooth = np.zeros_like(z_conds)
for i in range(z_conds.shape[1]):
    z_conds_smooth[:, i] = gaussian_filter1d(z_conds[:, i], sigma=SMOOTH_SIGMA)

# === 时间差分 ===
print("Computing temporal derivatives...")
z_conds_diff = np.zeros_like(z_conds_smooth)
for i in range(z_conds_smooth.shape[1]):
    # 计算一阶导数（变化率）
    z_conds_diff[1:, i] = np.diff(z_conds_smooth[:, i])
    z_conds_diff[0, i] = z_conds_diff[1, i]  # 第一帧用第二帧的值

# 计算变化率的绝对值
z_conds_change = np.abs(z_conds_diff)

print(f"Change rate computed")

# === 关键帧检测 ===
print(f"\nDetecting keyframes (threshold: {CHANGE_THRESHOLD_PERCENTILE}th percentile)...")
keyframes_per_dim = {}

for dim_idx, name in enumerate(basis_names):
    change_rates = z_conds_change[:, dim_idx]
    
    # 计算阈值
    threshold = np.percentile(change_rates, CHANGE_THRESHOLD_PERCENTILE)
    
    # 找到变化率超过阈值的帧
    keyframe_candidates = np.where(change_rates > threshold)[0]
    
    # 按变化率排序，取top-k
    sorted_indices = keyframe_candidates[np.argsort(change_rates[keyframe_candidates])[::-1]]
    keyframes = sorted_indices[:TOP_K_KEYFRAMES]
    
    # 按时间顺序排序（帧索引从小到大）
    keyframes_sorted = np.sort(keyframes)
    
    keyframes_per_dim[name] = {
        'indices': keyframes_sorted,
        'change_rates': change_rates[keyframes_sorted],
        'scores': z_conds_smooth[keyframes_sorted, dim_idx],
        'threshold': threshold
    }
    
    print(f"  {name}: {len(keyframes)} keyframes detected")

# === 可视化1: 时间线 + 变化率 + 关键帧标记 ===
print("\nGenerating timeline visualization...")
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
frame_indices = np.arange(num_encoded_frames)

fig, axes = plt.subplots(len(basis_names), 2, figsize=(16, 5*len(basis_names)))
if len(basis_names) == 1:
    axes = axes.reshape(1, -1)

for dim_idx, name in enumerate(basis_names):
    color = colors[dim_idx % len(colors)]
    
    # 左图：滤波后的CRL得分 + 关键帧标记
    ax1 = axes[dim_idx, 0]
    ax1.plot(frame_indices, z_conds_smooth[:, dim_idx], 
            color=color, linewidth=2, label='Smoothed CRL Score')
    
    # 标记关键帧
    kf_info = keyframes_per_dim[name]
    ax1.scatter(kf_info['indices'], kf_info['scores'], 
               color='red', s=100, marker='*', zorder=5, 
               label=f'Keyframes (n={len(kf_info["indices"])})')
    
    ax1.set_xlabel('Frame Index', fontsize=11)
    ax1.set_ylabel('CRL Score', fontsize=11)
    ax1.set_title(f'{name} - CRL Score with Keyframes', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 右图：变化率 + 阈值线
    ax2 = axes[dim_idx, 1]
    ax2.plot(frame_indices, z_conds_change[:, dim_idx], 
            color=color, linewidth=1.5, alpha=0.7, label='Change Rate')
    ax2.axhline(y=kf_info['threshold'], color='red', linestyle='--', 
               linewidth=2, label=f'Threshold ({CHANGE_THRESHOLD_PERCENTILE}th %ile)')
    
    # 标记关键帧
    ax2.scatter(kf_info['indices'], kf_info['change_rates'], 
               color='red', s=100, marker='*', zorder=5, label='Keyframes')
    
    ax2.set_xlabel('Frame Index', fontsize=11)
    ax2.set_ylabel('|Change Rate|', fontsize=11)
    ax2.set_title(f'{name} - Change Rate Detection', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('crl_keyframe_timeline.png', dpi=200, bbox_inches='tight')
print("Saved: crl_keyframe_timeline.png")
plt.show()

# === 重新读取关键帧用于显示 ===
print("\nLoading keyframes for display...")

# 收集所有需要的帧索引
all_keyframe_indices = set()
for kf_info in keyframes_per_dim.values():
    all_keyframe_indices.update(kf_info['indices'])

# 重新读取视频，只保存关键帧
keyframe_images = {}
cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = 0
sampled_frame_idx = 0

pbar = tqdm(total=total_frames_in_video, desc="Loading keyframes", unit="frame")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    if frame_count % SAMPLE_RATE == 0:
        if sampled_frame_idx in all_keyframe_indices:
            keyframe_images[sampled_frame_idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        sampled_frame_idx += 1
    
    frame_count += 1
    pbar.update(1)

pbar.close()
cap.release()
print(f"Loaded {len(keyframe_images)} keyframes")

# === 可视化2: 关键帧展示 ===
print("\nGenerating keyframe gallery...")
fig, axes = plt.subplots(len(basis_names), TOP_K_KEYFRAMES, 
                        figsize=(3*TOP_K_KEYFRAMES, 4*len(basis_names)))
if len(basis_names) == 1:
    axes = axes.reshape(1, -1)

for dim_idx, name in enumerate(basis_names):
    kf_info = keyframes_per_dim[name]
    
    for i, (idx, change_rate, score) in enumerate(zip(
        kf_info['indices'], kf_info['change_rates'], kf_info['scores'])):
        
        # 计算视频时间戳
        actual_frame_num = idx * SAMPLE_RATE
        time_seconds = actual_frame_num / fps
        time_str = f"{int(time_seconds//60):02d}:{int(time_seconds%60):02d}.{int((time_seconds%1)*10):01d}"
        
        ax = axes[dim_idx, i]
        ax.imshow(keyframe_images[idx])
        ax.set_title(f'#{i+1} @ {time_str}\nFrame {actual_frame_num}\nScore: {score:.3f} | Δ: {change_rate:.4f}', 
                    fontsize=9, fontweight='bold')
        ax.axis('off')
    
    # 如果关键帧不足，隐藏多余的子图
    for i in range(len(kf_info['indices']), TOP_K_KEYFRAMES):
        axes[dim_idx, i].axis('off')

plt.suptitle(f'Keyframes Detected by Change Rate - {CRITERION}', 
            fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('crl_keyframes.png', dpi=150, bbox_inches='tight')
print("Saved: crl_keyframes.png")
plt.show()

# === 统计信息 ===
print("\n=== Keyframe Detection Summary ===")
for name, kf_info in keyframes_per_dim.items():
    print(f"\n{name}:")
    print(f"  Threshold: {kf_info['threshold']:.4f}")
    print(f"  Keyframes detected: {len(kf_info['indices'])}")
    print(f"  Keyframes (in temporal order):")
    for i, (idx, cr, score) in enumerate(zip(
        kf_info['indices'], kf_info['change_rates'], kf_info['scores'])):
        actual_frame = idx * SAMPLE_RATE
        time_sec = actual_frame / fps
        time_str = f"{int(time_sec//60):02d}:{int(time_sec%60):02d}.{int((time_sec%1)*10):01d}"
        print(f"    #{i+1}: Frame {actual_frame:4d} @ {time_str} | Score: {score:.3f} | Δ: {cr:.4f}")

print("\n=== Done! ===")
print("Output files:")
print("  - crl_keyframe_timeline.png: Timeline with keyframes and change rates")
print("  - crl_keyframes.png: Gallery of detected keyframes")
