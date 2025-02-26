'''统计所有文件数量'''
import os
import argparse
from collections import defaultdict

def count_files_in_folders(base_path):
    # 创建一个默认字典来存储每个文件夹的文件计数
    folder_counts = defaultdict(int)
    total_files = 0
    
    try:
        # 遍历基础路径下的所有文件夹
        for folder_name in os.listdir(base_path):
            folder_path = os.path.join(base_path, folder_name)
            
            # 确保这是一个文件夹且名称是15-24之间的数字
            if os.path.isdir(folder_path) and folder_name.isdigit():
                if 15 <= int(folder_name) <= 24:
                    # 计算该文件夹中的文件数量
                    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
                    file_count = len(files)
                    folder_counts[folder_name] = file_count
                    total_files += file_count
        
        # 打印统计结果
        print("\n=== 文件夹统计结果 ===")
        print("文件夹\t文件数量")
        print("-------------------")
        
        # 按文件夹号码排序输出
        for folder_num in range(15, 25):
            folder_name = str(folder_num)
            count = folder_counts[folder_name]
            print(f"{folder_name}\t{count}")
            
        print("-------------------")
        print(f"总计:\t{total_files} 个文件")
        
    except Exception as e:
        print(f"统计过程中出现错误: {str(e)}")

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='统计文件夹中的文件数量')
    parser.add_argument('path', type=str, help='要统计的基础文件夹路径')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 运行统计函数
    count_files_in_folders(args.path)