"""
Conditional Representation Learning (CRL) for Video Analysis
基于用户指定准则构建语义基底，并进行条件表征转换
"""
import cv2
import numpy as np
import torch
from PIL import Image
import clip
from collections import deque

# ============ 导入配置 ============
from crl_config import *

print_config()

# 加载CLIP (VLM)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading CLIP model: {CLIP_MODEL} on {device}...")
model, preprocess = clip.load(CLIP_MODEL, device=device)

# === 阶段1: 基底构建 (Basis Construction) ===
print(f"\n=== Basis Construction for Criterion: '{CRITERION}' ===")
basis_dict = BASIS_TEXTS[CRITERION]
basis_names = list(basis_dict.keys())

print(f"Semantic dimensions: {len(basis_names)}")
for i, (name, texts) in enumerate(basis_dict.items()):
    print(f"  {i+1}. {name}: {len(texts)} descriptions")

# 文本编码：为每个维度构建基向量（多个描述的平均）
print("\nEncoding basis texts and aggregating...")
basis_vectors = []

with torch.no_grad():
    for name, texts in basis_dict.items():
        # 编码该维度的所有描述
        text_tokens = clip.tokenize(texts).to(device)
        text_features = model.encode_text(text_tokens).cpu().numpy()
        # 归一化
        text_features = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)
        # 聚合：取平均作为该维度的基向量
        basis_vector = np.mean(text_features, axis=0)
        # 再次归一化
        basis_vector = basis_vector / np.linalg.norm(basis_vector)
        basis_vectors.append(basis_vector)
        print(f"  {name}: averaged {len(texts)} texts -> basis vector")

# B 矩阵: [d x k], d=512 (CLIP维度), k=维度数量
basis_features = np.array(basis_vectors)  # [k x d]
B = basis_features.T  # [d x k]
print(f"\nSemantic basis B shape: {B.shape} (512 x {len(basis_names)})")

# 颜色方案
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 0, 255), (255, 128, 0)
]

def encode_frame(frame):
    """图像编码：获取通用表征 z_uni"""
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    with torch.no_grad():
        image_input = preprocess(image).unsqueeze(0).to(device)
        z_uni = model.encode_image(image_input).cpu().numpy()
    # 归一化
    z_uni = z_uni / np.linalg.norm(z_uni)
    return z_uni

def crl_transform(z_uni, B):
    """
    表征转换：z_cond = B^T @ z_uni
    将通用表征投影到语义基底上，得到条件表征
    """
    z_cond = np.dot(B.T, z_uni.T).flatten()  # [k]
    return z_cond

def draw_osd(frame, z_cond, history):
    """绘制OSD - 显示条件表征"""
    h, w = frame.shape[:2]
    
    # 计算OSD区域
    bar_height = 15
    spacing = 5
    num_basis = len(basis_names)
    graph_height = 80
    osd_height = num_basis * (bar_height + spacing) + graph_height + 20
    
    # 半透明背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, h - osd_height - 10), (w - 10, h - 10), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    # === 折线图（min-max缩放）===
    graph_x = 15
    graph_y = h - osd_height
    graph_w = w - 30
    
    if len(history) > 1:
        cv2.rectangle(frame, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_height), (20, 20, 20), -1)
        
        # 网格线
        for i in range(5):
            grid_y = graph_y + int(i * graph_height / 4)
            cv2.line(frame, (graph_x, grid_y), (graph_x + graph_w, grid_y), (50, 50, 50), 1)
        
        history_array = np.array(history)
        num_points = len(history_array)
        
        # 减去0.2基线后进行min-max缩放
        adjusted_history = np.maximum(history_array - 0.2, 0)
        global_min = np.min(adjusted_history)
        global_max = np.max(adjusted_history)
        value_range = max(global_max - global_min, 0.001)
        
        for i in range(num_basis):
            color = COLORS[i % len(COLORS)]
            points = []
            
            for j in range(num_points):
                x = int(graph_x + (j / max(num_points - 1, 1)) * graph_w)
                normalized_val = (adjusted_history[j, i] - global_min) / value_range
                y = int(graph_y + graph_height - normalized_val * graph_height)
                points.append((x, y))
            
            for k in range(len(points) - 1):
                cv2.line(frame, points[k], points[k + 1], color, 2)
        
        # 显示范围
        cv2.putText(frame, f"[{global_min:.2f}-{global_max:.2f}]", 
                    (graph_x + graph_w - 80, graph_y + 12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    
    # === 进度条区域 ===
    bar_width = w - 200
    y_start = h - osd_height + graph_height + 10
    
    # 找到最大值用于归一化显示
    max_val = max(np.max(z_cond), 0.001)
    
    for i, (basis_name, val) in enumerate(zip(basis_names, z_cond)):
        y = y_start + i * (bar_height + spacing)
        color = COLORS[i % len(COLORS)]
        
        # 使用维度名称
        display_text = basis_name[:15]
        
        cv2.putText(frame, display_text, (15, y + 12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 进度条
        bar_x = 120
        cv2.rectangle(frame, (bar_x, y), (bar_x + bar_width, y + bar_height), (40, 40, 40), -1)
        
        # 减去0.2基线，然后归一化
        adjusted_val = max(val - 0.2, 0)
        normalized_val = adjusted_val / max(max_val - 0.2, 0.001)
        filled_width = int(bar_width * normalized_val)
        if filled_width > 0:
            cv2.rectangle(frame, (bar_x, y), (bar_x + filled_width, y + bar_height), color, -1)
        
        # 数值
        cv2.putText(frame, f"{val:.2f}", (bar_x + bar_width + 5, y + 12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    return frame

# === 处理视频 ===
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Cannot open video: {VIDEO_PATH}")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"\n=== Processing Video ===")
print(f"Input: {total_frames} frames @ {fps} fps, {width}x{height}")
print(f"Output: {OUTPUT_VIDEO}")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

frame_count = 0
z_cond = np.zeros(len(basis_names))
history = deque(maxlen=100)

print("\nProcessing...")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # === 阶段2: 表征转换 (Representation Transformation) ===
    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        # 2.1 图像编码：获取通用表征
        z_uni = encode_frame(frame)
        
        # 2.2 投影计算：条件表征 = B^T @ z_uni
        z_cond = crl_transform(z_uni, B)
        
        history.append(z_cond.copy())
        
        if frame_count % 30 == 0:
            progress = frame_count / total_frames * 100
            print(f"Frame {frame_count}/{total_frames} ({progress:.1f}%)")
            print(f"  z_cond: {z_cond}")
    
    # 绘制OSD
    display_frame = draw_osd(frame.copy(), z_cond, history)
    
    # 帧号
    cv2.putText(display_frame, f"{frame_count}/{total_frames}", 
                (display_frame.shape[1] - 150, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    out.write(display_frame)
    frame_count += 1

cap.release()
out.release()

print(f"\n=== Done! ===")
print(f"Output saved to: {OUTPUT_VIDEO}")
print(f"\nCRL Summary:")
print(f"  Criterion: {CRITERION}")
print(f"  Semantic dimensions: {len(basis_names)}")
print(f"  Descriptions per dimension: {[len(texts) for texts in basis_dict.values()]}")
print(f"  Conditional representation: z_cond ∈ R^{len(basis_names)}")
