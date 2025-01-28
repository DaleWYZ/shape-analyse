from flask import Flask, request, jsonify, render_template, send_file
import os
import numpy as np
import cv2
import io
import base64
from shape_analyzer import analyze_shape, plot_shape_analysis
import logging
import traceback
import sys

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 确保上传目录存在
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    logger.info("访问首页")
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_image():
    logger.info("开始处理图片分析请求")
    try:
        if 'image' not in request.files:
            logger.warning("未找到上传的图片")
            return jsonify({'error': '没有上传图片'}), 400
        
        file = request.files['image']
        if file.filename == '':
            logger.warning("未选择文件")
            return jsonify({'error': '没有选择文件'}), 400
        
        # 检查文件类型
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
        if not file.filename.lower().endswith(tuple(allowed_extensions)):
            logger.warning(f"不支持的文件类型: {file.filename}")
            return jsonify({'error': '不支持的文件类型'}), 400

        # 保存上传的图片
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        logger.info(f"保存图片到: {image_path}")
        
        try:
            file.save(image_path)
            logger.info("图片保存成功")
            
            # 检查文件是否成功保存和可读
            if not os.path.exists(image_path):
                raise IOError("文件保存失败")
            
            # 检查文件大小
            file_size = os.path.getsize(image_path)
            logger.info(f"文件大小: {file_size} bytes")
            if file_size == 0:
                raise ValueError("文件为空")
            
            # 尝试读取图片验证其完整性
            test_img = cv2.imread(image_path)
            if test_img is None:
                raise ValueError("无法读取图片文件")
            
            logger.info("开始形状分析")
            results = analyze_shape(image_path)
            
            if results:
                logger.info("形状分析完成，准备返回结果")
                # 准备返回数据
                response_data = {
                    'status': 'success',
                    'results': {
                        'contours': results['轮廓点'].tolist(),
                        'circularity': float(results['圆度']),
                        'symmetry': float(1 - results['对称性指数']),
                        'regularity': float(results['形状规则度']),
                        'inner_regions': results['内部区域数量'],
                        'shape_features': {
                            'major_axis': float(results['主轴长度']),
                            'minor_axis': float(results['次轴长度']),
                            'orientation': float(results['方向角度']),
                            'center': results['轮廓点'].mean(axis=0)[0].tolist()
                        },
                        'inner_structure': {
                            'labels': results['内部分析']['labels'].tolist(),
                            'texture': np.mean(results['内部分析']['texture_features'], axis=0).tolist()
                        }
                    }
                }
                logger.info("返回分析结果")
                return jsonify(response_data)
            else:
                logger.warning("无法识别形状")
                return jsonify({'error': '无法识别形状'}), 400
                
        except Exception as e:
            logger.error(f"处理图片时发生错误: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({'error': f'处理图片时发生错误: {str(e)}'}), 500
        finally:
            # 清理临时文件
            try:
                if os.path.exists(image_path):
                    os.remove(image_path)
                    logger.info(f"已清理临时文件: {image_path}")
            except Exception as e:
                logger.warning(f"清理临时文件失败: {str(e)}")
                
    except Exception as e:
        logger.error(f"请求处理过程中发生错误: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'服务器错误: {str(e)}'}), 500

if __name__ == '__main__':
    logger.info("启动应用服务器")
    app.run(host='0.0.0.0', port=5000, debug=True) 