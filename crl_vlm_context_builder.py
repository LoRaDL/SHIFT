"""
CRL VLM上下文构建器
为VLM构建基于关键帧的视频理解上下文
"""
import cv2
import numpy as np
import torch
from PIL import Image
import clip
import json
import os
from datetime import datetime
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
from crl_config import *

class VLMContextBuilder:
    def __init__(self, video_path=None, criterion=None):
        """
        初始化VLM上下文构建器
        
        Args:
            video_path: 视频路径，默认使用配置文件
            criterion: 检测准则，默认使用配置文件
        """
        self.video_path = video_path or VIDEO_PATH
        self.criterion = criterion or CRITERION
        
        # 加载CLIP模型
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading CLIP model: {CLIP_MODEL} on {self.device}...")
        self.model, self.preprocess = clip.load(CLIP_MODEL, device=self.device)
        
        # 构建语义基底
        self._build_semantic_basis()
        
        # 视频信息
        self.fps = None
        self.total_frames = None
        self.width = None
        self.height = None
        
    def _build_semantic_basis(self):
        """构建语义基底"""
        print(f"Building semantic basis for: {self.criterion}")
        
        basis_dict = BASIS_TEXTS[self.criterion]
        self.basis_names = list(basis_dict.keys())
        
        basis_vectors = []
        with torch.no_grad():
            for name, texts in basis_dict.items():
                text_tokens = clip.tokenize(texts).to(self.device)
                text_features = self.model.encode_text(text_tokens).cpu().numpy()
                text_features = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)
                basis_vector = np.mean(text_features, axis=0)
                basis_vector = basis_vector / np.linalg.norm(basis_vector)
                basis_vectors.append(basis_vector)
        
        self.B = np.array(basis_vectors).T
        print(f"Semantic basis shape: {self.B.shape}")
    
    def extract_keyframes_with_context(self, output_dir="vlm_context", max_keyframes=20):
        """
        提取关键帧并构建VLM上下文
        
        Args:
            output_dir: 输出目录
            max_keyframes: 最大关键帧数量
            
        Returns:
            dict: 包含关键帧信息和上下文的字典
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 第一遍：编码所有帧
        print("=== Step 1: Encoding all frames ===")
        z_conds = self._encode_video()
        
        # 第二遍：检测关键时刻
        print("=== Step 2: Detecting key moments ===")
        key_moments = self._detect_key_moments(z_conds)
        
        # 第三遍：提取关键帧图像和构建上下文
        print("=== Step 3: Building VLM context ===")
        context_data = self._build_vlm_context(key_moments, z_conds, output_dir, max_keyframes)
        
        return context_data
    
    def _encode_video(self):
        """编码视频的所有采样帧"""
        cap = cv2.VideoCapture(self.video_path)
        
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video: {self.fps:.2f} fps, {self.total_frames} frames, {self.width}x{self.height}")
        
        z_conds = []
        frame_count = 0
        
        pbar = tqdm(total=self.total_frames, desc="Encoding frames")
        
        with torch.no_grad():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % SAMPLE_RATE == 0:
                    # 编码帧
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame_rgb)
                    image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
                    
                    # 获取图像特征
                    z_uni = self.model.encode_image(image_tensor).cpu().numpy()
                    z_uni = z_uni / np.linalg.norm(z_uni, axis=1, keepdims=True)
                    
                    # 投影到语义基底
                    z_cond = np.dot(z_uni, self.B)[0]
                    z_conds.append(z_cond)
                
                frame_count += 1
                pbar.update(1)
        
        pbar.close()
        cap.release()
        
        return np.array(z_conds, dtype=np.float32)
    
    def _detect_key_moments(self, z_conds):
        """检测关键时刻"""
        # 确保数据类型为float32
        z_conds = z_conds.astype(np.float32)
        
        # 滤波
        z_conds_smooth = np.zeros_like(z_conds, dtype=np.float32)
        for i in range(z_conds.shape[1]):
            z_conds_smooth[:, i] = gaussian_filter1d(z_conds[:, i], sigma=SMOOTH_SIGMA)
        
        # 计算变化率
        z_conds_diff = np.zeros_like(z_conds_smooth, dtype=np.float32)
        for i in range(z_conds_smooth.shape[1]):
            z_conds_diff[1:, i] = np.diff(z_conds_smooth[:, i])
            z_conds_diff[0, i] = z_conds_diff[1, i]
        
        z_conds_change = np.abs(z_conds_diff)
        
        # 检测关键帧
        key_moments = {}
        
        for dim_idx, name in enumerate(self.basis_names):
            scores = z_conds_smooth[:, dim_idx]
            change_rates = z_conds_change[:, dim_idx]
            
            # 归一化
            scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            change_norm = (change_rates - change_rates.min()) / (change_rates.max() - change_rates.min() + 1e-8)
            
            # 加权综合得分
            combined_score = SCORE_WEIGHT * scores_norm + CHANGE_WEIGHT * change_norm
            
            # 阈值检测
            threshold = np.percentile(combined_score, CHANGE_THRESHOLD_PERCENTILE)
            
            keyframe_candidates = np.where(
                (combined_score > threshold) & 
                (scores > MIN_SCORE_THRESHOLD)
            )[0]
            
            key_moments[name] = {
                'keyframes': keyframe_candidates,
                'scores': scores[keyframe_candidates],
                'changes': change_rates[keyframe_candidates],
                'combined_scores': combined_score[keyframe_candidates]
            }
            
            print(f"  {name}: {len(keyframe_candidates)} keyframes detected")
        
        return key_moments
    
    def _build_vlm_context(self, key_moments, z_conds, output_dir, max_keyframes):
        """构建VLM上下文"""
        # 收集所有关键帧
        all_keyframes = []
        for name, data in key_moments.items():
            for i, frame_idx in enumerate(data['keyframes']):
                all_keyframes.append({
                    'frame_idx': int(frame_idx),
                    'timestamp': float(frame_idx * SAMPLE_RATE / self.fps),
                    'dimension': name,
                    'score': float(data['scores'][i]),
                    'change': float(data['changes'][i]),
                    'combined_score': float(data['combined_scores'][i])
                })
        
        # 按综合得分排序，取前N个
        all_keyframes.sort(key=lambda x: x['combined_score'], reverse=True)
        selected_keyframes = all_keyframes[:max_keyframes]
        
        # 按时间排序
        selected_keyframes.sort(key=lambda x: x['frame_idx'])
        
        print(f"Selected {len(selected_keyframes)} keyframes for VLM context")
        
        # 提取关键帧图像
        cap = cv2.VideoCapture(self.video_path)
        frame_images = {}
        
        for kf in tqdm(selected_keyframes, desc="Extracting keyframe images"):
            frame_idx = kf['frame_idx']
            actual_frame_idx = frame_idx * SAMPLE_RATE
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, actual_frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # 保存图像
                image_filename = f"keyframe_{frame_idx:06d}.jpg"
                image_path = os.path.join(output_dir, image_filename)
                cv2.imwrite(image_path, frame)
                
                kf['image_path'] = image_path
                kf['image_filename'] = image_filename
                
                # 添加帧的语义向量
                kf['semantic_vector'] = z_conds[frame_idx].tolist()
        
        cap.release()
        
        # 构建完整的上下文数据
        context_data = {
            'metadata': {
                'video_path': self.video_path,
                'criterion': self.criterion,
                'total_frames': int(self.total_frames),
                'fps': float(self.fps),
                'duration_seconds': float(self.total_frames / self.fps),
                'dimensions': self.basis_names,
                'extraction_time': datetime.now().isoformat(),
                'sample_rate': int(SAMPLE_RATE)
            },
            'keyframes': selected_keyframes,
            'vlm_prompt': self._generate_vlm_prompt(selected_keyframes),
            'temporal_context': self._generate_temporal_context(selected_keyframes),
            'statistics': self._generate_statistics(key_moments, z_conds)
        }
        
        # 保存上下文数据
        context_file = os.path.join(output_dir, 'vlm_context.json')
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump(context_data, f, indent=2, ensure_ascii=False)
        
        print(f"VLM context saved to: {context_file}")
        
        return context_data
    
    def _generate_vlm_prompt(self, keyframes):
        """生成VLM提示词"""
        prompt = f"""# Video Analysis Context

## Task
Analyze the following sequence of {len(keyframes)} key frames extracted from a video based on the criterion: "{self.criterion}".

## Video Information
- Total duration: {self.total_frames / self.fps:.1f} seconds
- Frame rate: {self.fps:.1f} fps
- Detection dimensions: {', '.join(self.basis_names)}

## Key Frames Sequence
The frames are ordered chronologically. Each frame has been identified as significant based on semantic content and temporal changes.

**Note**: Each image has a red number label in the top-left corner (1, 2, 3, etc.) to help you identify the frame sequence.

"""
        
        for i, kf in enumerate(keyframes):
            prompt += f"""
### Frame {i+1}: {kf['image_filename']}
- **Timestamp**: {kf['timestamp']:.2f}s
- **Detection dimension**: {kf['dimension']}
- **Confidence score**: {kf['score']:.3f}
- **Change intensity**: {kf['change']:.3f}
"""
        
        prompt += """
## Analysis Instructions
1. Examine each frame in sequence
2. Identify the detected objects/concepts
3. Describe temporal relationships between frames
4. Summarize the overall video content and key events
5. Note any patterns or trends in the detection results

Please provide a comprehensive analysis of this video sequence.
"""
        
        return prompt
    
    def _generate_temporal_context(self, keyframes):
        """生成时间上下文信息"""
        if len(keyframes) < 2:
            return {"message": "Insufficient keyframes for temporal analysis"}
        
        temporal_info = {
            'total_span_seconds': float(keyframes[-1]['timestamp'] - keyframes[0]['timestamp']),
            'average_interval_seconds': 0.0,
            'frame_intervals': [],
            'dimension_timeline': {}
        }
        
        # 计算帧间隔
        intervals = []
        for i in range(1, len(keyframes)):
            interval = keyframes[i]['timestamp'] - keyframes[i-1]['timestamp']
            intervals.append(interval)
            temporal_info['frame_intervals'].append({
                'from_frame': int(i-1),
                'to_frame': int(i),
                'interval_seconds': float(interval)
            })
        
        if intervals:
            temporal_info['average_interval_seconds'] = float(np.mean(intervals))
        
        # 按维度组织时间线
        for dim in self.basis_names:
            dim_frames = [kf for kf in keyframes if kf['dimension'] == dim]
            temporal_info['dimension_timeline'][dim] = {
                'count': int(len(dim_frames)),
                'timestamps': [float(kf['timestamp']) for kf in dim_frames],
                'avg_score': float(np.mean([kf['score'] for kf in dim_frames])) if dim_frames else 0.0
            }
        
        return temporal_info
    
    def _generate_statistics(self, key_moments, z_conds):
        """生成统计信息"""
        stats = {
            'total_analyzed_frames': int(len(z_conds)),
            'dimensions': {}
        }
        
        for dim_idx, name in enumerate(self.basis_names):
            dim_scores = z_conds[:, dim_idx]
            stats['dimensions'][name] = {
                'mean_score': float(np.mean(dim_scores)),
                'max_score': float(np.max(dim_scores)),
                'min_score': float(np.min(dim_scores)),
                'std_score': float(np.std(dim_scores)),
                'keyframes_detected': int(len(key_moments[name]['keyframes']))
            }
        
        return stats

def main():
    """主函数示例"""
    print("=== CRL VLM Context Builder ===")
    
    # 创建上下文构建器
    builder = VLMContextBuilder()
    
    # 提取关键帧并构建上下文
    context_data = builder.extract_keyframes_with_context(
        output_dir="vlm_context_output",
        max_keyframes=15
    )
    
    print("\n=== Context Summary ===")
    print(f"Video: {context_data['metadata']['video_path']}")
    print(f"Duration: {context_data['metadata']['duration_seconds']:.1f}s")
    print(f"Keyframes extracted: {len(context_data['keyframes'])}")
    print(f"Dimensions: {', '.join(context_data['metadata']['dimensions'])}")
    
    print("\n=== VLM Prompt Preview ===")
    print(context_data['vlm_prompt'][:500] + "...")
    
    print("\n=== Temporal Context ===")
    temporal = context_data['temporal_context']
    print(f"Time span: {temporal['total_span_seconds']:.1f}s")
    print(f"Average interval: {temporal['average_interval_seconds']:.1f}s")
    
    print("\nDone! Check 'vlm_context_output' directory for results.")

if __name__ == "__main__":
    main()