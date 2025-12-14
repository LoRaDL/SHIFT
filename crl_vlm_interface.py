"""
CRL VLM交互接口 - LMStudio本地版
"""
import json
import os
import requests
from PIL import Image
import base64
from io import BytesIO
from typing import List, Dict, Any, Optional

class LMStudioVLMInterface:
    """LMStudio本地VLM接口"""
    
    def __init__(self, base_url: str = "http://localhost:1234"):
        self.base_url = base_url
        self.context_data = None
    
    def load_context(self, context_file: str):
        """加载VLM上下文数据"""
        with open(context_file, 'r', encoding='utf-8') as f:
            self.context_data = json.load(f)
        print(f"Loaded context with {len(self.context_data['keyframes'])} keyframes")
    
    def prepare_images_for_vlm(self, max_size: tuple = (512, 512), add_labels: bool = True) -> List[str]:
        """
        准备图像用于VLM输入
        
        Args:
            max_size: 最大图像尺寸 (width, height)
            add_labels: 是否在图片上添加数字标签
            
        Returns:
            List[str]: Base64编码的图像列表
        """
        if not self.context_data:
            raise ValueError("No context data loaded. Call load_context() first.")
        
        encoded_images = []
        
        for i, kf in enumerate(self.context_data['keyframes']):
            image_path = kf['image_path']
            if os.path.exists(image_path):
                # 加载并调整图像大小
                image = Image.open(image_path)
                image.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # 添加数字标签
                if add_labels:
                    image = self._add_frame_label(image, i + 1)
                
                # 转换为base64
                buffer = BytesIO()
                image.save(buffer, format='JPEG', quality=85)
                image_b64 = base64.b64encode(buffer.getvalue()).decode()
                encoded_images.append(image_b64)
            else:
                print(f"Warning: Image not found: {image_path}")
        
        return encoded_images
    
    def _add_frame_label(self, image: Image.Image, frame_number: int) -> Image.Image:
        """
        在图片左上角添加帧编号标签
        
        Args:
            image: PIL图像
            frame_number: 帧编号
            
        Returns:
            Image.Image: 添加了标签的图像
        """
        from PIL import ImageDraw, ImageFont
        
        # 创建副本避免修改原图
        labeled_image = image.copy()
        draw = ImageDraw.Draw(labeled_image)
        
        # 标签文本
        label_text = str(frame_number)
        
        # 尝试使用系统字体，如果失败则使用默认字体
        try:
            # Windows系统字体
            font_size = max(20, min(image.width, image.height) // 20)
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            try:
                # 备用字体
                font_size = max(20, min(image.width, image.height) // 20)
                font = ImageFont.load_default()
            except:
                font = None
        
        # 获取文本尺寸
        if font:
            bbox = draw.textbbox((0, 0), label_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else:
            text_width, text_height = 20, 15
        
        # 标签位置和尺寸
        padding = 5
        label_width = text_width + 2 * padding
        label_height = text_height + 2 * padding
        
        # 绘制半透明背景
        overlay = Image.new('RGBA', labeled_image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        # 背景矩形 (红色半透明)
        overlay_draw.rectangle(
            [(0, 0), (label_width, label_height)],
            fill=(255, 0, 0, 180)  # 红色背景，180透明度
        )
        
        # 合并背景
        labeled_image = Image.alpha_composite(labeled_image.convert('RGBA'), overlay).convert('RGB')
        
        # 重新创建绘制对象
        draw = ImageDraw.Draw(labeled_image)
        
        # 绘制白色文字
        text_x = padding
        text_y = padding
        
        if font:
            draw.text((text_x, text_y), label_text, fill=(255, 255, 255), font=font)
        else:
            draw.text((text_x, text_y), label_text, fill=(255, 255, 255))
        
        return labeled_image
    
    def get_vlm_prompt(self) -> str:
        """获取VLM提示词"""
        if not self.context_data:
            raise ValueError("No context data loaded.")
        return self.context_data['vlm_prompt']
    
    def create_lmstudio_messages(self, custom_prompt: Optional[str] = None) -> List[Dict]:
        """
        创建LMStudio API消息格式
        
        Args:
            custom_prompt: 自定义提示词
            
        Returns:
            List[Dict]: LMStudio消息格式
        """
        if not self.context_data:
            raise ValueError("No context data loaded.")
        
        # 准备图像
        images_b64 = self.prepare_images_for_vlm()
        
        # 构建消息内容
        content = []
        
        # 添加文本提示
        prompt_text = custom_prompt or self.get_vlm_prompt()
        content.append({
            "type": "text",
            "text": prompt_text
        })
        
        # 添加图像
        for i, image_b64 in enumerate(images_b64):
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_b64}"
                }
            })
        
        return [
            {
                "role": "system",
                "content": "You are an expert video analyst. Analyze the provided sequence of key frames and their context to understand the video content."
            },
            {
                "role": "user",
                "content": content
            }
        ]
    
    def analyze_with_lmstudio(self, custom_prompt: Optional[str] = None, model: str = "llava") -> Dict:
        """
        使用LMStudio分析视频
        
        Args:
            custom_prompt: 自定义提示词
            model: 模型名称
            
        Returns:
            Dict: 分析结果
        """
        messages = self.create_lmstudio_messages(custom_prompt)
        
        print(f"Sending {len(self.context_data['keyframes'])} frames to LMStudio...")
        
        # LMStudio API调用
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 2000,
            "temperature": 0.1,
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=300  # 5分钟超时
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'analysis': result['choices'][0]['message']['content'],
                    'usage': result.get('usage', {}),
                    'model': model
                }
            else:
                raise Exception(f"LMStudio API error: {response.status_code} - {response.text}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to connect to LMStudio: {e}")
    
    def export_for_manual_upload(self, output_dir: str = "lmstudio_export"):
        """导出文件用于手动上传到LMStudio"""
        if not self.context_data:
            raise ValueError("No context data loaded.")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 处理图片并添加标签
        for i, kf in enumerate(self.context_data['keyframes']):
            if os.path.exists(kf['image_path']):
                # 加载原图
                image = Image.open(kf['image_path'])
                
                # 添加数字标签
                labeled_image = self._add_frame_label(image, i + 1)
                
                # 保存带标签的图片
                dest_path = os.path.join(output_dir, f"frame_{i+1:02d}_{kf['image_filename']}")
                labeled_image.save(dest_path, 'JPEG', quality=95)
        
        # 导出提示词
        prompt_file = os.path.join(output_dir, "prompt.txt")
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write("=== LMStudio VLM Analysis Prompt ===\n\n")
            f.write(self.get_vlm_prompt())
        
        print(f"✅ Files exported to: {output_dir}")
        print(f"📄 Prompt: {prompt_file}")
        print(f"🖼️  Images: {len(self.context_data['keyframes'])} frames")
        
        return output_dir

def demo_analysis():
    """演示如何使用LMStudio VLM接口"""
    print("=== LMStudio VLM Interface Demo ===")
    
    # 检查上下文文件是否存在
    context_file = "vlm_context_output/vlm_context.json"
    if not os.path.exists(context_file):
        print(f"Context file not found: {context_file}")
        print("Please run crl_vlm_context_builder.py first to generate the context.")
        return
    
    # LMStudio接口选项
    print("\nLMStudio VLM Analysis Options:")
    print("1. Auto API call (requires LMStudio server running)")
    print("2. Export for manual upload")
    
    choice = input("Select option (1-2): ").strip()
    
    interface = LMStudioVLMInterface()
    interface.load_context(context_file)
    
    if choice == "1":
        # 自动API调用
        try:
            # 检查LMStudio是否运行
            test_response = requests.get(f"{interface.base_url}/v1/models", timeout=5)
            if test_response.status_code != 200:
                raise Exception("LMStudio server not responding")
            
            print("✅ LMStudio server detected")
            
            # 获取可用模型
            models = test_response.json()
            if models.get('data'):
                print(f"Available models: {[m['id'] for m in models['data']]}")
                model_name = models['data'][0]['id']  # 使用第一个模型
            else:
                model_name = "llava"  # 默认模型名
            
            result = interface.analyze_with_lmstudio(model=model_name)
            
            print("\n=== LMStudio Analysis ===")
            print(result['analysis'])
            
            # 保存结果
            output_file = "vlm_context_output/lmstudio_analysis.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("=== LMStudio VLM Analysis ===\n\n")
                f.write(result['analysis'])
                f.write(f"\n\n=== Model Info ===\n")
                f.write(f"Model: {result['model']}\n")
                f.write(f"Usage: {result['usage']}\n")
            
            print(f"\n📄 Analysis saved to: {output_file}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            print("💡 Make sure LMStudio is running with a vision model loaded")
            print("💡 Default server: http://localhost:1234")
    
    elif choice == "2":
        # 导出用于手动上传
        export_dir = interface.export_for_manual_upload()
        print(f"\n✅ Files ready for manual upload to LMStudio")
        print(f"📁 Location: {export_dir}")
        print("\n📋 Next steps:")
        print("1. Open LMStudio and load a vision-language model")
        print("2. Start a chat session")
        print("3. Upload all frame images from the export folder")
        print("4. Copy and paste the prompt from prompt.txt")
        print("5. Submit for analysis")
    
    else:
        print("Invalid choice")

if __name__ == "__main__":
    demo_analysis()