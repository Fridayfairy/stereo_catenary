"""
测试配置文件是否正确加载
"""

import yaml
import numpy as np

def test_config_loading():
    """
    测试配置文件加载
    """
    print("测试配置文件加载...")
    
    try:
        with open('config/cameras.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print("✅ 配置文件加载成功")
        
        # 检查必要的键
        required_keys = ['left_camera', 'right_camera', 'gravity_direction']
        for key in required_keys:
            if key not in config:
                print(f"❌ 缺少必要的键: {key}")
                return False
            print(f"✅ 找到键: {key}")
        
        # 检查相机参数
        for camera in ['left_camera', 'right_camera']:
            cam_config = config[camera]
            required_cam_keys = ['intrinsic_matrix', 'rotation_matrix', 'translation_vector']
            
            for key in required_cam_keys:
                if key not in cam_config:
                    print(f"❌ 相机 {camera} 缺少键: {key}")
                    return False
                print(f"✅ 相机 {camera} 找到键: {key}")
        
        # 转换为numpy数组测试
        print("\n测试numpy数组转换...")
        
        config['left_camera']['intrinsic_matrix'] = np.array(config['left_camera']['intrinsic_matrix'])
        config['left_camera']['rotation_matrix'] = np.array(config['left_camera']['rotation_matrix'])
        config['left_camera']['translation_vector'] = np.array(config['left_camera']['translation_vector'])
        
        config['right_camera']['intrinsic_matrix'] = np.array(config['right_camera']['intrinsic_matrix'])
        config['right_camera']['rotation_matrix'] = np.array(config['right_camera']['rotation_matrix'])
        config['right_camera']['translation_vector'] = np.array(config['right_camera']['translation_vector'])
        
        config['gravity_direction'] = np.array(config['gravity_direction'])
        
        print("✅ numpy数组转换成功")
        
        # 打印配置信息
        print("\n配置信息:")
        print(f"左相机内参矩阵形状: {config['left_camera']['intrinsic_matrix'].shape}")
        print(f"右相机内参矩阵形状: {config['right_camera']['intrinsic_matrix'].shape}")
        print(f"重力方向: {config['gravity_direction']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_config_loading()
    if success:
        print("\n🎉 配置文件测试通过！")
    else:
        print("\n💥 配置文件测试失败！")
