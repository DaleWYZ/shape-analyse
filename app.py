from flask import Flask, request, jsonify, render_template, send_file
import os
import numpy as np
import cv2
import io
import base64
from shape_analyzer import analyze_shape, plot_shape_analysis

app = Flask(__name__)

# 确保上传目录存在
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': '没有上传图片'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    # 保存上传的图片
    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)
    
    try:
        # 获取分析结果
        results = analyze_shape(image_path)
        
        if results:
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
            
            return jsonify(response_data)
        else:
            return jsonify({'error': '无法识别形状'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # 清理临时文件
        if os.path.exists(image_path):
            os.remove(image_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 