'''
代码的作用是生成储存相机外参的exr.txt文件
'''

import os
import numpy as np
import gzip
import json

# 为每个物体定义相机内参（转化为3x3矩阵形式）
camera_params_dict = {
    'basketball': np.array([
        [882.35, 0, 354.5],
        [0, 746.00, 236],
        [0, 0, 1]
    ]),
    'bicycle': np.array([
        [873.215, 0, 354.0],
        [0, 746.144, 236.5],
        [0, 0, 1]
    ]),
    'bridge': np.array([
        [882.358, 0, 354.0],
        [0, 758.966, 236.5],
        [0, 0, 1]
    ]),
    'car': np.array([
        [898.054, 0, 352.5],
        [0, 791.367, 235.5],
        [0, 0, 1]
    ]),
    'chair': np.array([
        [884.607, 0, 353.5],
        [0, 764.502, 236.0],
        [0, 0, 1]
    ]),
    'merge': np.array([
        [861.904, 0, 355.0],
        [0, 744.521, 237.0],
        [0, 0, 1]
    ]),
    'sitting': np.array([
        [895.250, 0, 353.0],
        [0, 772.566, 235.0],
        [0, 0, 1]
    ]),
    'soccer_goal_1': np.array([
        [896.590, 0, 352.0],
        [0, 777.678, 235.5],
        [0, 0, 1]
    ]),
    'standing': np.array([
        [899.773, 0, 352.5],
        [0, 776.647, 235.5],
        [0, 0, 1]
    ]),
    'tall_building': np.array([
        [884.394, 0, 355.0],
        [0, 750.200, 236.5],
        [0, 0, 1]
    ]),
    'apples': np.array([
        [841.491, 0, 355.0],
        [0, 830.171, 236.5],
        [0, 0, 1]
    ]),
    'corridor': np.array([
        [445.734, 0, 417.5],
        [0, 957.046, 240.5],
        [0, 0, 1]
    ]),
    'drone1': np.array([
        [783.316, 0, 355.0],
        [0, 771.577, 236.5],
        [0, 0, 1]
    ]),
    'drone2': np.array([
        [859.508, 0, 353.5],
        [0, 757.709, 235.5],
        [0, 0, 1]
    ]),
    'drone3': np.array([
        [894.182, 0, 352.5],
        [0, 772.576, 235.5],
        [0, 0, 1]
    ]),
    'drone4': np.array([
        [849.295, 0, 354.0],
        [0, 726.664, 233.0],
        [0, 0, 1]
    ]),
    'drone5': np.array([
        [889.846, 0, 352.0],
        [0, 717.849, 235.5],
        [0, 0, 1]
    ]),
    'drone6': np.array([
        [2202.306, 0, 350.5],
        [0, 1418.106, 234.5],
        [0, 0, 1]
    ]),
    'heated': np.array([
        [714.089, 0, 357.5],
        [0, 714.316, 239.5],
        [0, 0, 1]
    ]),
    'building': np.array([
        [833.234, 0, 320.0],
        [0, 821.382, 240.0],
        [0, 0, 1]
    ]),
    'DailyStuff': np.array([
        [810.178, 0, 320.0],
        [0, 813.324, 240.0],
        [0, 0, 1]
    ]),
    'Dimsum': np.array([
        [811.748, 0, 320.0],
        [0, 818.202, 240.0],
        [0, 0, 1]
    ]),
    'Ebike': np.array([
        [817.738, 0, 320.0],
        [0, 793.624, 240.0],
        [0, 0, 1]
    ]),
    # 第二组
    'IronIngot': np.array([
        [829.023, 0, 320.0],
        [0, 860.195, 240.0],
        [0, 0, 1]
    ]),
    'LandScape': np.array([
        [838.271, 0, 320.0],
        [0, 841.362, 240.0],
        [0, 0, 1]
    ]),
    'Parterre': np.array([
        [806.507, 0, 320.0],
        [0, 813.945, 240.0],
        [0, 0, 1]
    ]),
    'RoadBlock': np.array([
        [807.118, 0, 320.0],
        [0, 797.505, 240.0],
        [0, 0, 1]
    ]),
    'RotaryKiln': np.array([
        [833.568, 0, 320.0],
        [0, 825.741, 240.0],
        [0, 0, 1]
    ]),
    'Truck': np.array([
        [834.352, 0, 320.0],
        [0, 839.950, 240.0],
        [0, 0, 1]
    ]),
    'alley': np.array([
        [4024.485, 0, 352.0],
        [0, 323.771, 234.5],
        [0, 0, 1]
    ]),
    'ancient_building1': np.array([
        [790.528, 0, 343.5],
        [0, 248.835, 226.0],
        [0, 0, 1]
    ]),
    'ancient_building2': np.array([
        [1080.946, 0, 341.5],
        [0, 732.676, 229.0],
        [0, 0, 1]
    ]),
    'community1': np.array([
        [875.033, 0, 355.0],
        [0, 792.984, 236.5],
        [0, 0, 1]
    ]),
    'community2': np.array([
        [1049.192, 0, 333.0],
        [0, 1344.041, 222.0],
        [0, 0, 1]
    ]),
    'drone7': np.array([
        [900.259, 0, 353.5],
        [0, 771.725, 236.0],
        [0, 0, 1]
    ]),
    'drone8': np.array([
        [887.053, 0, 352.5],
        [0, 765.161, 235.0],
        [0, 0, 1]
    ]),
    'drone9': np.array([
        [906.067, 0, 353.0],
        [0, 781.275, 235.0],
        [0, 0, 1]
    ]),
    'drone10': np.array([
        [889.853, 0, 353.0],
        [0, 780.230, 236.0],
        [0, 0, 1]
    ]),
    'drone11': np.array([
        [1196.430, 0, 353.5],
        [0, 1671.079, 236.5],
        [0, 0, 1]
    ]),
    'drone12': np.array([
        [950.806, 0, 324.0],
        [0, 811.707, 223.0],
        [0, 0, 1]
    ]),
    'drone13': np.array([
        [1054.366, 0, 348.5],
        [0, 1064.448, 224.5],
        [0, 0, 1]
    ]),
    'drone14': np.array([
        [770.414, 0, 373.0],
        [0, 1234.053, 240.5],
        [0, 0, 1]
    ]),
    'drone15': np.array([
        [4693.829, 0, 353.0],
        [0, 920.299, 235.5],
        [0, 0, 1]
    ]),
    'excavator': np.array([
        [893.471, 0, 352.5],
        [0, 763.960, 235.5],
        [0, 0, 1]
    ]),
    'garden': np.array([
        [889.358, 0, 352.5],
        [0, 777.847, 235.5],
        [0, 0, 1]
    ]),
    'ice_box': np.array([
        [985.961, 0, 360.0],
        [0, 1011.968, 238.5],
        [0, 0, 1]
    ]),
    'ice_cream': np.array([
        [1083.750, 0, 339.0],
        [0, 838.800, 223.0],
        [0, 0, 1]
    ]),
    'kettle': np.array([
        [842.012, 0, 359.5],
        [0, 720.294, 239.5],
        [0, 0, 1]
    ]),
    'laptop': np.array([
        [908.381, 0, 353.0],
        [0, 775.092, 235.5],
        [0, 0, 1]
    ]),
    'meeting_room': np.array([
        [813.604, 0, 352.0],
        [0, 896.324, 234.5],
        [0, 0, 1]
    ]),
    'mountain_road1': np.array([
        [752.279, 0, 365.0],
        [0, 802.522, 243.0],
        [0, 0, 1]
    ]),
    # 第三部分
    'mountain_road2': np.array([
        [720.321, 0, 364.5],
        [0, 718.039, 242.0],
        [0, 0, 1]
    ]),
    'sidewalk': np.array([
        [894.036, 0, 351.0],
        [0, 722.250, 235.0],
        [0, 0, 1]
    ]),
    'stone': np.array([
        [891.852, 0, 353.5],
        [0, 780.689, 236.0],
        [0, 0, 1]
    ]),
    'traffic_scene1': np.array([
        [808.684, 0, 359.5],
        [0, 699.957, 240.0],
        [0, 0, 1]
    ]),
    'traffic_scene2': np.array([
        [937.055, 0, 352.5],
        [0, 623.682, 234.5],
        [0, 0, 1]
    ]),
    'trunk': np.array([
        [886.879, 0, 353.5],
        [0, 771.776, 236.0],
        [0, 0, 1]
    ]),
    'village_road1': np.array([
        [369.849, 0, 324.5],
        [0, 519.213, 224.0],
        [0, 0, 1]
    ]),
    'village_road2': np.array([
        [925.521, 0, 326.0],
        [0, 2126.952, 218.5],
        [0, 0, 1]
    ]),

    # 第四部分
    'SingleTime_AM00_origin': np.array([
        [1215.147, 0, 320.0],
        [0, 1216.453, 256.0],
        [0, 0, 1]
    ]),
    'SingleTime_AM00_part': np.array([
        [1184.313, 0, 320.0],
        [0, 1185.254, 256.0],
        [0, 0, 1]
    ]),
    'SingleTime_AM10_origin': np.array([
        [1134.058, 0, 320.0],
        [0, 1134.591, 256.0],
        [0, 0, 1]
    ]),
    'SingleTime_AM10_part': np.array([
        [1166.151, 0, 320.0],
        [0, 1168.725, 256.0],
        [0, 0, 1]
    ]),
    'SingleTime_PM14_origin': np.array([
        [1131.770, 0, 320.0],
        [0, 1132.554, 256.0],
        [0, 0, 1]
    ]),
    'SingleTime_PM14_part': np.array([
        [1161.010, 0, 320.0],
        [0, 1162.561, 256.0],
        [0, 0, 1]
    ]),
    'SingleTime_PM18_origin': np.array([
        [1135.217, 0, 320.0],
        [0, 1136.254, 256.0],
        [0, 0, 1]
    ]),
    'SingleTime_PM22_origin': np.array([
        [1139.990, 0, 320.0],
        [0, 1141.075, 256.0],
        [0, 0, 1]
    ]),
    'SingleTime_PM18_part': np.array([
        [1175.956, 0, 320.0],
        [0, 1178.965, 256.0],
        [0, 0, 1]
    ]),
    'SingleTime_PM22_part': np.array([
        [1179.487, 0, 320.0],
        [0, 1181.742, 256.0],
        [0, 0, 1]
    ]),

    # 第五部分
    'desk1-halfsphere': np.array([
        [1996.31, 0, 320.0],
        [0, 1999.87, 240.0],
        [0, 0, 1]
    ]),
    'desk1-halfsphere-ic': np.array([
        [1474.64, 0, 320.0],
        [0, 1473.89, 240.0],
        [0, 0, 1]
    ]),
    'desk1-halfsphere-person': np.array([
        [1420.29, 0, 320.0],
        [0, 1419.69, 240.0],
        [0, 0, 1]
    ]),
    'desk1-xyz': np.array([
        [1849.89, 0, 320.0],
        [0, 1787.97, 240.0],
        [0, 0, 1]
    ]),
    'desk2-halfsphere': np.array([
        [1698.92, 0, 320.0],
        [0, 1688.9, 240.0],
        [0, 0, 1]
    ]),
    'trees': np.array([
        [855.314, 0, 240.0],
        [0, 856.989, 320.0],
        [0, 0, 1]
    ]),

    # 第六部分
    'local_robust': np.array([
        [2365.03, 0, 320.0],
        [0, 3395.32, 256.0],
        [0, 0, 1]
    ]),
    'outdoor_robust_day1_part1': np.array([
        [1048.53, 0, 320.0],
        [0, 979.695, 256.0],
        [0, 0, 1]
    ]),
    'outdoor_robust_day1_part2': np.array([
        [180.856, 0, 320.0],
        [0, 1392.72, 256.0],
        [0, 0, 1]
    ]),
    'outdoor_robust_night1': np.array([
        [1536.53, 0, 320.0],
        [0, 1604.57, 256.0],
        [0, 0, 1]
    ]),
    'outdoor_robust_night2': np.array([
        [3359.65, 0, 320.0],
        [0, 3321.77, 256.0],
        [0, 0, 1]
    ]),
    'scence1': np.array([
        [1117.57, 0, 320.0],
        [0, 260.213, 240.0],
        [0, 0, 1]
    ]),
    'scence2': np.array([
        [1466.51, 0, 320.0],
        [0, 772.04, 240.0],
        [0, 0, 1]
    ]),
    'scence3': np.array([
        [1416.6, 0, 320.0],
        [0, 502.349, 240.0],
        [0, 0, 1]
    ]),
    'scence4': np.array([
        [1172.78, 0, 320.0],
        [0, 1060.45, 240.0],
        [0, 0, 1]
    ]),
    'scence5': np.array([
        [1184.28, 0, 320.0],
        [0, 1252.36, 240.0],
        [0, 0, 1]
    ]),
    'scence6': np.array([
        [1141.89, 0, 320.0],
        [0, 898.286, 240.0],
        [0, 0, 1]
    ]),
    'scence7': np.array([
        [861.942, 0, 320.0],
        [0, 467.332, 240.0],
        [0, 0, 1]
    ]),
    'scence8': np.array([
        [1248.98, 0, 320.0],
        [0, 1270.49, 240.0],
        [0, 0, 1]
    ]),
    'scence9': np.array([
        [845.337, 0, 320.0],
        [0, 422.29, 240.0],
        [0, 0, 1]
    ]),
    'scence10': np.array([
        [1025.42, 0, 320.0],
        [0, 1252.74, 240.0],
        [0, 0, 1]
    ]),
    'scence11': np.array([
        [1803.02, 0, 320.0],
        [0, 396.235, 240.0],
        [0, 0, 1]
    ]),
    'scence12': np.array([
        [1763.97, 0, 320.0],
        [0, 361.021, 240.0],
        [0, 0, 1]
    ]),
    'scence13': np.array([
        [1223.69, 0, 320.0],
        [0, 1239.19, 240.0],
        [0, 0, 1]
    ]),
    'scence14': np.array([
        [1153.02, 0, 320.0],
        [0, 1201.71, 240.0],
        [0, 0, 1]
    ]),
    'scence15': np.array([
        [931.653, 0, 320.0],
        [0, 897.366, 240.0],
        [0, 0, 1]
    ]),
    'scence19': np.array([
        [1200.27, 0, 320.0],
        [0, 315.419, 240.0],
        [0, 0, 1]
    ]),

}

# 创建一个字典来存储每个物体的image_list
objects_image_lists = {}

# 假设物体文件夹的名称
'''
object_names = ['basketball', 'bridge', 'car', 'chair', 'merge', 'sitting','standing','tall_building',  #8
                'corridor', 'drone3', 'drone4', 'drone5', 'building', 'DailyStuff', 'Dimsum', 'excavator',#8
                'Ebike','IronIngot', 'LandScape', 'Parterre', 'RoadBlock', 'RotaryKiln', 'ancient_building1', 'ancient_building2', #8
                 'community1', 'community2', 'drone7', 'drone9', 'drone10', 'drone11', 'drone12','bicycle', # 8
                'drone13', 'drone14', 'drone8','ice_box', 'ice_cream','mountain_road1', 'mountain_road2', # 7
                'sidewalk', 'traffic_scene1', 'trunk','village_road1', 'heated','drone6','kettle',  # 7
                 'SingleTime_AM00_part', 'SingleTime_AM10_origin', 'SingleTime_AM10_part', # 3
                'SingleTime_PM14_origin','SingleTime_PM14_part', 'SingleTime_PM18_origin', 'SingleTime_PM22_origin','SingleTime_PM18_part','SingleTime_PM22_part', # 6
                'local_robust', 'meeting_room','soccer_goal_1','traffic_scene2',# 4
                'scence3','scence4','scence5','scence6','scence7','scence8','scence9','scence10', # 8
                'scence13','scence14','scence15', 'desk1-halfsphere', 'desk1-halfsphere-ic', 'desk1-halfsphere-person', 'desk1-xyz', 'desk2-halfsphere','trees' # 9
                ]
'''
object_names = ['apples', 'drone1','drone2','drone15','garden','laptop','SingleTime_AM00_origin',
                'scence1','scence2','scence11','scence12','scence19','stone','Truck'] #9



# base_folder_path = '/media/siu/512G/thermal'
base_folder_path = 'thermal'

num = 0

for object_name in object_names:
    # 设置图像和相应的文本文件路径
    image_folder = os.path.join(base_folder_path, 'thermal_images', object_name, 'images')
    exr_file_path = os.path.join(base_folder_path, 'colmap', object_name, 'sparse', 'exr.txt')

    # 创建列表来存储字典
    image_list = []

    with open(exr_file_path, 'r') as file:
        lines = file.readlines()
        for line in lines[4:]:
            data = line.split()
            if len(data) < 10:
                continue  # 确保有足够的数据
            image_id = data[0]
            qw, qx, qy, qz = map(float, data[1:5])
            tx, ty, tz = map(float, data[5:8])
            camera_id = data[8]
            image_name = data[9]

            # 生成外参数的4x4矩阵
            R = np.array([[1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
                          [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
                          [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]])

            extrinsics = np.zeros((3, 4))
            extrinsics[:3, :3] = R
            extrinsics[:3, 3] = [tx, ty, tz]

            # 获取对应物体的内参
            camera_params = camera_params_dict[object_name]

            # 生成对应的字典
            image_data = {
                'filepath': os.path.join(image_folder, image_name),
                'extri': extrinsics.tolist(),  # 外参数为4x4矩阵
                'intri': camera_params.tolist(),  # 内参数为3x3矩阵
            }

            # 将字典加入列表
            image_list.append(image_data)

    # 将每个物体的image_list存储到字典中
    objects_image_lists[object_name] = image_list
    num = num+1
    print(num)

output_file_path = 'thermal_test.jgz'
with gzip.open(output_file_path, 'wt', encoding='utf-8') as gz_file:
    json.dump(objects_image_lists, gz_file)



