"""
CRL静态分析 - 生成时间线图和top帧展示
"""
import cv2
import numpy as np
import torch
from PIL import Image
import clip
import matplotlib.pyplot as plt
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

print(f"Dimensions: {basis_names}")
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

B = np.array(basis_vectors).T  # [512 x k]
print(f"Basis B: {B.shape}")

# === 流式读取和编码（不保存帧）===
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

# 优化GPU性能
torch.backends.cudnn.benchmark = True

with torch.no_grad():
    pbar = tqdm(total=total_frames_in_video, desc="Encoding", unit="frame", smoothing=0.1)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % SAMPLE_RATE == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch_buffer.append(frame_rgb)
            
            if len(batch_buffer) >= BATCH_SIZE:
                # CPU预处理
                batch_images = torch.stack([
                    preprocess(Image.fromarray(f)) 
                    for f in batch_buffer
                ])
                
                # 异步传输到GPU
                batch_images = batch_images.to(device, non_blocking=True)
                
                # GPU推理
                z_uni_batch = model.encode_image(batch_images)
                z_uni_batch = z_uni_batch.cpu().numpy()
                z_uni_batch = z_uni_batch / np.linalg.norm(z_uni_batch, axis=1, keepdims=True)
                z_cond_batch = np.dot(z_uni_batch, B)
                z_conds.append(z_cond_batch)
                
                batch_buffer = []
        
        frame_count += 1
        pbar.update(1)
    
    pbar.close()
    
    # 处理剩余帧
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
z_conds = np.vstack(z_conds)
num_encoded_frames = len(z_conds)

print(f"Encoded {num_encoded_frames} frames, shape: {z_conds.shape}")

# === 数据滤波 ===
print("\nApplying smoothing filter...")
from scipy.ndimage import gaussian_filter1d

# 保存原始数据并转换为float64
z_conds_raw = z_conds.copy()
z_conds = z_conds.astype(np.float64)

# 对每个维度应用高斯滤波
sigma = 0.1  # 滤波强度，越大越平滑
z_conds_smooth = np.zeros_like(z_conds, dtype=np.float64)
for i in range(z_conds.shape[1]):
    z_conds_smooth[:, i] = gaussian_filter1d(z_conds[:, i], sigma=sigma)

print(f"Smoothing applied with sigma={sigma}")

# === 图1: 相似度时间线 ===
print("\nGenerating timeline plot...")
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

frame_indices = np.arange(num_encoded_frames)
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']

# 第1图：原始数据
for i, name in enumerate(basis_names):
    axes[0].plot(frame_indices, z_conds_raw[:, i], 
                label=name, color=colors[i % len(colors)], 
                alpha=0.5, linewidth=1.5)

axes[0].set_xlabel('Frame Index', fontsize=11)
axes[0].set_ylabel('CRL Score', fontsize=11)
axes[0].set_title(f'Raw CRL Scores - {CRITERION}', 
                 fontsize=13, fontweight='bold')
axes[0].legend(loc='upper right', fontsize=10)
axes[0].grid(True, alpha=0.3)

# 第2图：滤波后数据
for i, name in enumerate(basis_names):
    axes[1].plot(frame_indices, z_conds_smooth[:, i], 
                label=name, color=colors[i % len(colors)], 
                alpha=0.8, linewidth=2)

axes[1].set_xlabel('Frame Index', fontsize=11)
axes[1].set_ylabel('CRL Score (Smoothed)', fontsize=11)
axes[1].set_title(f'Smoothed CRL Scores (σ={sigma}) - {CRITERION}', 
                 fontsize=13, fontweight='bold')
axes[1].legend(loc='upper right', fontsize=10)
axes[1].grid(True, alpha=0.3)

# 第3图：最大值和对应维度（使用滤波后数据）
max_scores = np.max(z_conds_smooth, axis=1)
best_dims = np.argmax(z_conds_smooth, axis=1)
colors_scatter = [colors[dim % len(colors)] for dim in best_dims]

axes[2].scatter(frame_indices, max_scores, c=colors_scatter, s=30, alpha=0.6)
axes[2].plot(frame_indices, max_scores, color='gray', alpha=0.3, linewidth=1)
axes[2].set_xlabel('Frame Index', fontsize=11)
axes[2].set_ylabel('Max CRL Score', fontsize=11)
axes[2].set_title('Maximum Score Over Time (smoothed, colored by dominant dimension)', 
                 fontsize=13, fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('crl_timeline.png', dpi=200, bbox_inches='tight')
print("Saved: crl_timeline.png")
plt.show()

# === 重新读取top帧用于显示 ===
print("\nLoading top frames for display...")

# 收集所有需要的帧索引
top_frame_indices = set()
for dim_idx, name in enumerate(basis_names):
    scores = z_conds_smooth[:, dim_idx]
    top_indices = np.argsort(scores)[-3:][::-1]
    top_frame_indices.update(top_indices)

# 重新读取视频，只保存top帧
top_frame_images = {}
cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = 0
sampled_frame_idx = 0

pbar = tqdm(total=total_frames_in_video, desc="Loading top frames", unit="frame")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    if frame_count % SAMPLE_RATE == 0:
        if sampled_frame_idx in top_frame_indices:
            top_frame_images[sampled_frame_idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        sampled_frame_idx += 1
    
    frame_count += 1
    pbar.update(1)

pbar.close()
cap.release()
print(f"Loaded {len(top_frame_images)} top frames")

# === 图2: Top帧展示 ===
print("\nGenerating top frames...")

fig, axes = plt.subplots(len(basis_names), 3, figsize=(12, 4*len(basis_names)))
if len(basis_names) == 1:
    axes = axes.reshape(1, -1)

for dim_idx, name in enumerate(basis_names):
    scores = z_conds_smooth[:, dim_idx]
    top_indices = np.argsort(scores)[-3:][::-1]
    
    for i, idx in enumerate(top_indices):
        ax = axes[dim_idx, i]
        ax.imshow(top_frame_images[idx])
        ax.set_title(f'{name}\nFrame {idx}, Score: {scores[idx]:.3f}', 
                    fontsize=10, fontweight='bold')
        ax.axis('off')

plt.suptitle(f'Top Frames per Dimension - {CRITERION}', 
            fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('crl_top_frames.png', dpi=150, bbox_inches='tight')
print("Saved: crl_top_frames.png")
plt.show()

# === 统计信息 ===
print("\n=== Statistics ===")
for i, name in enumerate(basis_names):
    scores = z_conds[:, i]
    print(f"\n{name}:")
    print(f"  Mean: {np.mean(scores):.4f}")
    print(f"  Std:  {np.std(scores):.4f}")
    print(f"  Max:  {np.max(scores):.4f}")
    print(f"  Min:  {np.min(scores):.4f}")

print("\nDone!")
