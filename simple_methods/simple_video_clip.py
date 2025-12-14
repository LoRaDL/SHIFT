"""
简单的视频CLIP分析脚本 - 单文件版本
"""
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import clip

# ============ 配置区 ============
VIDEO_PATH = "dataset/vsi-super-recall/10mins/00000003.mp4"
KEYWORDS = ["Hello Kitty"]
SAMPLE_RATE = 1  # 每30帧取1帧
THRESHOLD = 0.23  # 相似度阈值
# ================================

# 加载CLIP模型
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading CLIP on {device}...")
model, preprocess = clip.load("ViT-B/32", device=device)

# 提取视频帧
print(f"Extracting frames from {VIDEO_PATH}...")
cap = cv2.VideoCapture(VIDEO_PATH)
frames = []
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_count % SAMPLE_RATE == 0:
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frame_count += 1
cap.release()
print(f"Extracted {len(frames)} frames")

# 编码帧
print("Encoding frames...")
frame_features = []
with torch.no_grad():
    for frame in frames:
        image = preprocess(Image.fromarray(frame)).unsqueeze(0).to(device)
        features = model.encode_image(image)
        frame_features.append(features.cpu().numpy())
frame_features = np.vstack(frame_features)
frame_features = frame_features / np.linalg.norm(frame_features, axis=1, keepdims=True)

# 编码关键词
print("Encoding keywords...")
with torch.no_grad():
    text = clip.tokenize(KEYWORDS).to(device)
    text_features = model.encode_text(text).cpu().numpy()
text_features = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)

# 计算相似度
similarities = np.dot(frame_features, text_features.T)
max_sims = np.max(similarities, axis=1)
best_kws = np.argmax(similarities, axis=1)

# 过滤
filtered_idx = np.where(max_sims >= THRESHOLD)[0]
print(f"Filtered {len(filtered_idx)}/{len(frames)} frames")

# 降维可视化
print("Visualizing...")
if len(filtered_idx) < 5:
    print(f"Warning: Only {len(filtered_idx)} frames passed filter. Lowering threshold or using all frames.")
    filtered_idx = np.arange(len(frames))  # 使用所有帧
perplexity = min(30, len(filtered_idx) - 1)
reduced = TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(frame_features[filtered_idx])

# 绘图
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 按关键词着色
axes[0].scatter(reduced[:, 0], reduced[:, 1], c=best_kws[filtered_idx], cmap='tab10', s=50, alpha=0.6)
axes[0].set_title('Frames by Keyword')
for i, kw in enumerate(KEYWORDS):
    axes[0].text(0.02, 0.98-i*0.05, f"{i}: {kw}", transform=axes[0].transAxes, fontsize=9)

# 按相似度着色
scatter = axes[1].scatter(reduced[:, 0], reduced[:, 1], c=max_sims[filtered_idx], cmap='viridis', s=50, alpha=0.6)
axes[1].set_title('Frames by Similarity')
plt.colorbar(scatter, ax=axes[1])

plt.tight_layout()
plt.savefig('result.png', dpi=200)
print("Saved to result.png")
plt.show()

# 显示top帧
top_idx = np.argsort(max_sims)[-9:][::-1]
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
for i, idx in enumerate(top_idx):
    axes[i//3, i%3].imshow(frames[idx])
    axes[i//3, i%3].set_title(f"{KEYWORDS[best_kws[idx]]}\n{max_sims[idx]:.3f}", fontsize=10)
    axes[i//3, i%3].axis('off')
plt.tight_layout()
plt.savefig('top_frames.png', dpi=150)
print("Saved to top_frames.png")
plt.show()

# 相似度随时间变化
print("Plotting similarity over time...")
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# 上图：每个关键词的相似度曲线
frame_indices = np.arange(len(frames))
for i, keyword in enumerate(KEYWORDS):
    axes[0].plot(frame_indices, similarities[:, i], label=keyword, alpha=0.7, linewidth=1.5)
axes[0].set_xlabel('Frame Index')
axes[0].set_ylabel('Similarity Score')
axes[0].set_title('Similarity Score for Each Keyword Over Time')
axes[0].legend(loc='upper right', fontsize=9)
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=THRESHOLD, color='r', linestyle='--', label=f'Threshold={THRESHOLD}', linewidth=1)

# 下图：最大相似度和对应的关键词
colors = plt.cm.tab10(best_kws / len(KEYWORDS))
axes[1].scatter(frame_indices, max_sims, c=colors, s=20, alpha=0.6)
axes[1].plot(frame_indices, max_sims, color='gray', alpha=0.3, linewidth=0.5)
axes[1].axhline(y=THRESHOLD, color='r', linestyle='--', label=f'Threshold={THRESHOLD}', linewidth=1)
axes[1].set_xlabel('Frame Index')
axes[1].set_ylabel('Max Similarity Score')
axes[1].set_title('Maximum Similarity Score Over Time (colored by best matching keyword)')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.savefig('similarity_timeline.png', dpi=200)
print("Saved to similarity_timeline.png")
plt.show()

print("Done!")
