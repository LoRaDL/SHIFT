"""
导出带相似度OSD的视频
"""
import cv2
import numpy as np
import torch
from PIL import Image
import clip
from collections import deque

# ============ 配置 ============
VIDEO_PATH = "dataset/procthor/000290/raw_navigation_camera__1.mp4"
KEYWORDS = ["Teddy"]
PROCESS_EVERY_N_FRAMES = 1  # 每N帧计算一次相似度
OUTPUT_VIDEO = "output_with_osd.mp4"
# ==============================

# 加载CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading CLIP on {device}...")
model, preprocess = clip.load("ViT-B/32", device=device)

# 编码关键词
print("Encoding keywords...")
with torch.no_grad():
    text = clip.tokenize(KEYWORDS).to(device)
    text_features = model.encode_text(text).cpu().numpy()
text_features = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)

# 颜色方案 (BGR格式)
COLORS = [
    (255, 0, 0),    # 蓝
    (0, 255, 0),    # 绿
    (0, 0, 255),    # 红
    (255, 255, 0),  # 青
    (255, 0, 255),  # 品红
    (0, 255, 255),  # 黄
]

def encode_frame(frame):
    """编码单帧"""
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    with torch.no_grad():
        image_input = preprocess(image).unsqueeze(0).to(device)
        features = model.encode_image(image_input).cpu().numpy()
    features = features / np.linalg.norm(features)
    return features

def draw_osd(frame, similarities, history):
    """在帧上绘制简化OSD - 进度条、数字和滚动折线图"""
    h, w = frame.shape[:2]
    
    # 计算OSD区域大小
    bar_height = 15
    spacing = 5
    num_keywords = len(KEYWORDS)
    graph_height = 80  # 折线图高度
    osd_height = num_keywords * (bar_height + spacing) + graph_height + 20
    
    # 半透明背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, h - osd_height - 10), (w - 10, h - 10), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    # === 折线图区域（上方） ===
    graph_x = 15
    graph_y = h - osd_height
    graph_w = w - 30
    
    if len(history) > 1:
        # 绘制折线图背景
        cv2.rectangle(frame, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_height), (20, 20, 20), -1)
        
        # 绘制网格线
        for i in range(5):
            grid_y = graph_y + int(i * graph_height / 4)
            cv2.line(frame, (graph_x, grid_y), (graph_x + graph_w, grid_y), (50, 50, 50), 1)
        
        # 绘制折线（使用min-max缩放）
        history_array = np.array(history)
        num_points = len(history_array)
        
        # 计算全局min和max
        global_min = np.min(history_array)
        global_max = np.max(history_array)
        value_range = global_max - global_min
        
        # 避免除零
        if value_range < 0.001:
            value_range = 0.001
        
        for i in range(len(KEYWORDS)):
            color = COLORS[i % len(COLORS)]
            points = []
            
            for j in range(num_points):
                x = int(graph_x + (j / max(num_points - 1, 1)) * graph_w)
                # Min-max归一化
                normalized_val = (history_array[j, i] - global_min) / value_range
                y = int(graph_y + graph_height - normalized_val * graph_height)
                points.append((x, y))
            
            # 绘制线条
            for k in range(len(points) - 1):
                cv2.line(frame, points[k], points[k + 1], color, 2)
        
        # 显示当前范围
        cv2.putText(frame, f"[{global_min:.2f}-{global_max:.2f}]", 
                    (graph_x + graph_w - 80, graph_y + 12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    
    # === 进度条区域（下方） ===
    bar_width = w - 200
    y_start = h - osd_height + graph_height + 10
    
    for i, (keyword, sim) in enumerate(zip(KEYWORDS, similarities)):
        y = y_start + i * (bar_height + spacing)
        color = COLORS[i % len(COLORS)]
        
        # 关键词文本
        text = keyword[:12]
        cv2.putText(frame, text, (15, y + 12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 进度条背景
        bar_x = 120
        cv2.rectangle(frame, (bar_x, y), (bar_x + bar_width, y + bar_height), (40, 40, 40), -1)
        
        # 进度条填充（上限0.4）
        normalized_sim = min(sim / 0.4, 1.0)
        filled_width = int(bar_width * normalized_sim)
        if filled_width > 0:
            cv2.rectangle(frame, (bar_x, y), (bar_x + filled_width, y + bar_height), color, -1)
        
        # 数值
        cv2.putText(frame, f"{sim:.2f}", (bar_x + bar_width + 5, y + 12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    return frame

# 打开输入视频
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Cannot open video: {VIDEO_PATH}")
    exit()

# 获取视频信息
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Input video: {total_frames} frames @ {fps} fps, {width}x{height}")
print(f"Output video: {OUTPUT_VIDEO}")

# 创建输出视频
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

frame_count = 0
similarities = np.zeros(len(KEYWORDS))
history = deque(maxlen=100)

print("Processing video...")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 每N帧计算一次相似度
    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        features = encode_frame(frame)
        similarities = np.dot(features, text_features.T)[0]
        history.append(similarities.copy())
        
        # 显示进度
        if frame_count % 30 == 0:
            progress = frame_count / total_frames * 100
            print(f"Progress: {frame_count}/{total_frames} ({progress:.1f}%)")
    
    # 绘制OSD
    display_frame = draw_osd(frame.copy(), similarities, history)
    
    # 显示帧号（右上角，不遮挡）
    progress = frame_count / total_frames * 100
    cv2.putText(display_frame, f"{frame_count}/{total_frames}", 
                (display_frame.shape[1] - 150, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 写入输出视频
    out.write(display_frame)
    frame_count += 1

cap.release()
out.release()
print(f"\nDone! Output saved to: {OUTPUT_VIDEO}")
print(f"You can play it with: vlc {OUTPUT_VIDEO}")
