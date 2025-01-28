from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import os
import numpy as np
import cv2
import io
import base64
from shape_analyzer import analyze_shape, plot_shape_analysis
import logging
import traceback
import sys
import signal
import gc
from werkzeug.utils import secure_filename

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
CORS(app)

# 配置
MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))  # 16MB
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# 确保上传目录存在
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_uploads():
    """清理上传目录"""
    try:
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                logger.error(f"清理文件失败 {file_path}: {str(e)}")
    except Exception as e:
        logger.error(f"清理目录失败: {str(e)}")

def signal_handler(signum, frame):
    """处理进程信号"""
    logger.info(f"收到信号 {signum}，开始清理...")
    cleanup_uploads()
    sys.exit(0)

# 注册信号处理器
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

@app.route('/health')
def health_check():
    """健康检查端点"""
    return jsonify({'status': 'healthy'}), 200

@app.route('/')
def index():
    logger.info("访问首页")
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_image():
    logger.info("开始处理图片分析请求")
    image_path = None
    
    try:
        if 'image' not in request.files:
            logger.warning("未找到上传的图片")
            return jsonify({'error': '没有上传图片'}), 400
        
        file = request.files['image']
        if file.filename == '':
            logger.warning("未选择文件")
            return jsonify({'error': '没有选择文件'}), 400
        
        if not allowed_file(file.filename):
            logger.warning(f"不支持的文件类型: {file.filename}")
            return jsonify({'error': '不支持的文件类型'}), 400

        filename = secure_filename(file.filename)
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        logger.info(f"保存图片到: {image_path}")
        
        try:
            file.save(image_path)
            logger.info("图片保存成功")
            
            if not os.path.exists(image_path):
                raise IOError("文件保存失败")
            
            file_size = os.path.getsize(image_path)
            logger.info(f"文件大小: {file_size} bytes")
            if file_size == 0:
                raise ValueError("文件为空")
            
            if file_size > MAX_CONTENT_LENGTH:
                raise ValueError("文件太大")
            
            test_img = cv2.imread(image_path)
            if test_img is None:
                raise ValueError("无法读取图片文件")
            
            logger.info("开始形状分析")
            results = analyze_shape(image_path)
            
            if results:
                logger.info("形状分析完成，准备返回结果")
                try:
                    response_data = {
                        'status': 'success',
                        'results': {
                            'contours': [point[0].tolist() for point in results['轮廓点']] if isinstance(results['轮廓点'], np.ndarray) else results['轮廓点'],
                            'circularity': float(results['圆度']),
                            'symmetry': float(1 - results['对称性指数']),
                            'regularity': float(results['形状规则度']),
                            'inner_regions': int(results['内部区域数量']),
                            'shape_features': {
                                'major_axis': float(results['主轴长度']),
                                'minor_axis': float(results['次轴长度']),
                                'orientation': float(results['方向角度']),
                                'center': results['轮廓点'].mean(axis=0)[0].tolist() if isinstance(results['轮廓点'], np.ndarray) else [
                                    float(results['轮廓点'][0][0][0]),
                                    float(results['轮廓点'][0][0][1])
                                ]
                            },
                            'inner_structure': {
                                'labels': results['内部分析']['labels'].tolist() if isinstance(results['内部分析']['labels'], np.ndarray) else results['内部分析']['labels'],
                                'texture': np.mean(results['内部分析']['texture_features'], axis=0).tolist() if isinstance(results['内部分析']['texture_features'], np.ndarray) else results['内部分析']['texture_features']
                            }
                        }
                    }
                    
                    # 添加详细的数据验证日志
                    logger.debug("轮廓点数据类型: %s", type(results['轮廓点']))
                    logger.debug("轮廓点示例: %s", str(results['轮廓点'][:5]) if len(results['轮廓点']) > 5 else str(results['轮廓点']))
                    logger.debug("形状特征数据: %s", str(response_data['results']['shape_features']))
                    logger.debug("返回数据结构: %s", str(response_data))
                    
                    return jsonify(response_data)
                    
                except Exception as e:
                    logger.error(f"处理返回数据时发生错误: {str(e)}")
                    logger.error(traceback.format_exc())
                    return jsonify({'error': f'数据处理错误: {str(e)}'}), 500
            else:
                logger.warning("无法识别形状")
                return jsonify({'error': '无法识别形状'}), 400
                
        except Exception as e:
            logger.error(f"处理图片时发生错误: {str(e)}")
            logger.error(traceback.format_exc())
            error_response = jsonify({'error': f'处理图片时发生错误: {str(e)}'})
            error_response.headers.add('Access-Control-Allow-Origin', '*')
            return error_response, 500
        finally:
            try:
                if image_path and os.path.exists(image_path):
                    os.remove(image_path)
                    logger.info(f"已清理临时文件: {image_path}")
            except Exception as e:
                logger.warning(f"清理临时文件失败: {str(e)}")
            
            # 强制垃圾回收
            gc.collect()
                
    except Exception as e:
        logger.error(f"请求处理过程中发生错误: {str(e)}")
        logger.error(traceback.format_exc())
        error_response = jsonify({'error': f'服务器错误: {str(e)}'})
        error_response.headers.add('Access-Control-Allow-Origin', '*')
        return error_response, 500

@app.errorhandler(413)
def request_entity_too_large(error):
    """处理文件过大的错误"""
    return jsonify({'error': '文件太大'}), 413

@app.errorhandler(500)
def internal_server_error(error):
    """处理服务器错误"""
    return jsonify({'error': '服务器内部错误'}), 500

if __name__ == '__main__':
    logger.info("启动应用服务器")
    try:
        app.run(host='0.0.0.0', port=5000)
    except Exception as e:
        logger.error(f"服务器启动失败: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        cleanup_uploads() 