import cv2
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from skimage import measure, feature, morphology
from skimage.measure import regionprops
from skimage.filters import threshold_otsu
from skimage.transform import rotate
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from skimage.filters import gabor
from skimage.color import rgb2gray
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.colors as mcolors
import logging
import traceback

logger = logging.getLogger(__name__)

def analyze_inner_structure(image_path):
    """分析图像的内部结构"""
    logger.info("开始分析内部结构")
    try:
        # 读取并预处理图像
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("无法读取图像文件")
        
        logger.debug("转换图像为灰度")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. 内部区域分割
        logger.debug("计算距离变换")
        distance = ndi.distance_transform_edt(gray)
        
        logger.debug("检测局部最大值")
        coordinates = peak_local_max(distance, min_distance=20)
        local_max = np.zeros_like(distance, dtype=bool)
        local_max[tuple(coordinates.T)] = True
        
        logger.debug("执行分水岭分割")
        markers, num_features = ndi.label(local_max)
        labels = watershed(-distance, markers, mask=gray)
        
        # 2. 纹理特征提取
        logger.debug("提取Gabor纹理特征")
        gabor_responses = []
        for theta in range(4):  # 不同方向的纹理
            filt_real, filt_imag = gabor(gray, frequency=0.6,
                                       theta=theta * np.pi/4)
            gabor_responses.append(filt_real)
        
        logger.info("内部结构分析完成")
        return {
            'labels': labels,
            'texture_features': np.array(gabor_responses)
        }
    except Exception as e:
        logger.error(f"分析内部结构时发生错误: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def analyze_shape(image_path):
    """分析图像中的形状特征"""
    logger.info(f"开始分析形状: {image_path}")
    try:
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("无法读取图像文件")
        
        logger.debug("转换为灰度图像")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 图像预处理
        logger.debug("执行高斯模糊")
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        logger.debug("执行阈值分割")
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 找到轮廓
        logger.debug("查找轮廓")
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # 获取最大轮廓
            main_contour = max(contours, key=cv2.contourArea)
            
            # 计算基本特征
            logger.debug("计算基本形状特征")
            area = cv2.contourArea(main_contour)
            perimeter = cv2.arcLength(main_contour, True)
            
            # 拟合椭圆
            ellipse = cv2.fitEllipse(main_contour)
            
            # 计算圆度
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # 计算长宽比
            _, (width, height), _ = ellipse
            aspect_ratio = max(width, height) / min(width, height)
            
            logger.debug("计算高级形状特征")
            # 转换为二值图像用于scikit-image分析
            binary = thresh > 0
            label_img = measure.label(binary)
            regions = regionprops(label_img)
            main_region = regions[0]
            
            # 添加内部结构分析
            logger.debug("分析内部结构")
            inner_analysis = analyze_inner_structure(image_path)
            
            # 计算内部子区域的特征
            unique_regions = len(np.unique(inner_analysis['labels'])) - 1
            
            # 计算纹理复杂度
            texture_complexity = np.mean([np.std(resp) for resp in inner_analysis['texture_features']])
            
            logger.info("形状分析完成")
            return {
                "面积": area,
                "周长": perimeter,
                "圆度": circularity,
                "长宽比": aspect_ratio,
                "轮廓点": main_contour,
                "方向角度": np.degrees(main_region.orientation),
                "偏心率": main_region.eccentricity,
                "等效直径": main_region.equivalent_diameter,
                "致密度": main_region.solidity,
                "主轴长度": main_region.major_axis_length,
                "次轴长度": main_region.minor_axis_length,
                "形状规则度": main_region.extent,
                "对称性指数": main_region.eccentricity,
                "内部区域数量": unique_regions,
                "纹理复杂度": texture_complexity,
                "内部分析": inner_analysis
            }
        
        logger.warning("未找到有效轮廓")
        return None
    
    except Exception as e:
        logger.error(f"分析形状时发生错误: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def create_radar_chart(results, ax):
    # 定义要展示的特征和对应的最大值
    features = {
        '圆度': (results['圆度'], 1.0),
        '对称性': (1 - results['对称性指数'], 1.0),
        '规则度': (results['形状规则度'], 1.0),
        '致密度': (results['致密度'], 1.0),
        '纹理': (1 - results['纹理复杂度']/np.max(results['纹理复杂度']), 1.0),
        '复杂度': (len(str(results['内部区域数量']))/10, 1.0)
    }
    
    # 计算角度
    angles = np.linspace(0, 2*np.pi, len(features), endpoint=False)
    
    # 闭合图形
    angles = np.concatenate((angles, [angles[0]]))
    
    # 获取特征值和最大值
    values = [v[0] for v in features.values()]
    max_values = [v[1] for v in features.values()]
    values = np.concatenate((values, [values[0]]))
    
    # 绘制雷达图
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    
    # 设置角度刻度
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features.keys())
    
    # 设置范围
    ax.set_ylim(0, 1)
    
    # 添加网格
    ax.grid(True)
    
    return ax

def create_hierarchical_view(results, ax):
    def add_node(center, radius, label, value, level):
        circle = plt.Circle(center, radius, color=plt.cm.viridis(value))
        ax.add_artist(circle)
        
        # 添加标签
        ax.text(center[0], center[1], f'{label}\n{value:.2f}', 
                ha='center', va='center', color='white')
        
        return circle
    
    # 标准化所有值到0-1之间
    normalized_values = {
        '基本形状': results['圆度'],
        '对称性': 1 - results['对称性指数'],
        '纹理': 1 - results['纹理复杂度']/np.max(results['纹理复杂度']),
        '内部结构': results['致密度']
    }
    
    # 创建层次结构
    center_x, center_y = 0.5, 0.5
    main_radius = 0.4
    sub_radius = 0.15
    
    # 主节点
    add_node((center_x, center_y), main_radius, '形状总体',
             np.mean(list(normalized_values.values())), 0)
    
    # 子节点
    angles = np.linspace(0, 2*np.pi, len(normalized_values), endpoint=False)
    for i, (label, value) in enumerate(normalized_values.items()):
        x = center_x + (main_radius - sub_radius) * np.cos(angles[i])
        y = center_y + (main_radius - sub_radius) * np.sin(angles[i])
        add_node((x, y), sub_radius, label, value, 1)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

def plot_shape_analysis(image_path):
    img = cv2.imread(image_path)
    results = analyze_shape(image_path)
    
    if results:
        # 创建更大的图形显示更多信息
        plt.figure(figsize=(20, 12))
        
        # 1. 原始图像和轮廓
        plt.subplot(231)
        cv2.drawContours(img, [results["轮廓点"]], -1, (0, 255, 0), 2)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("轮廓检测结果")
        
        # 2. 内部区域分割结果
        plt.subplot(232)
        plt.imshow(results['内部分析']['labels'], cmap='nipy_spectral')
        plt.title(f"内部区域分割 (发现 {results['内部区域数量']} 个区域)")
        
        # 3. 纹理分析结果
        plt.subplot(233)
        texture_img = np.mean(results['内部分析']['texture_features'], axis=0)
        plt.imshow(texture_img, cmap='gray')
        plt.title(f"纹理分析 (复杂度: {results['纹理复杂度']:.2f})")
        
        # 4. 雷达图展示
        ax_radar = plt.subplot(234, projection='polar')
        create_radar_chart(results, ax_radar)
        plt.title("形状特征雷达图")
        
        # 5. 层次化视图
        ax_hier = plt.subplot(235)
        create_hierarchical_view(results, ax_hier)
        plt.title("形状特征层次图")
        
        # 6. 数值结果
        plt.subplot(236)
        plt.axis('off')
        info = f"""
        形状分析结果：
        
        基本特征：
        面积: {results['面积']:.1f}
        周长: {results['周长']:.1f}
        圆度: {results['圆度']:.3f}
        长宽比: {results['长宽比']:.3f}
        
        高级特征：
        方向角度: {results['方向角度']:.1f}°
        偏心率: {results['偏心率']:.3f}
        等效直径: {results['等效直径']:.1f}
        致密度: {results['致密度']:.3f}
        形状规则度: {results['形状规则度']:.3f}
        对称性指数: {results['对称性指数']:.3f}
        
        内部结构：
        内部区域数: {results['内部区域数量']}
        纹理复杂度: {results['纹理复杂度']:.3f}
        """
        plt.text(0.1, 0.95, info, fontsize=10, va='top')
        plt.title("详细分析结果")
        
        plt.tight_layout()
        plt.show() 