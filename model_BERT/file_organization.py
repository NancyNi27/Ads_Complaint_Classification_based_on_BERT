import os
import shutil
from pathlib import Path

def organize_files(source_dir='output'):
    # 确保源目录存在
    if not os.path.exists(source_dir):
        print(f"源目录 {source_dir} 不存在")
        return
    
    # 创建要处理的文件夹范围（15-24）
    folder_numbers = range(15, 25)
    
    # 为每个数字创建对应的文件夹
    for num in folder_numbers:
        folder_name = str(num)
        folder_path = os.path.join(source_dir, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"创建文件夹: {folder_name}")
    
    # 遍历源目录中的所有文件
    for filename in os.listdir(source_dir):
        if filename.endswith('.txt'):
            try:
                # 获取文件名前两位数字
                prefix = filename[:2]
                if prefix.isdigit():
                    prefix_num = int(prefix)
                    
                    # 检查数字是否在15-24范围内
                    if 15 <= prefix_num <= 24:
                        # 构建源文件和目标文件的完整路径
                        source_file = os.path.join(source_dir, filename)
                        target_folder = os.path.join(source_dir, str(prefix_num))
                        target_file = os.path.join(target_folder, filename)
                        
                        # 移动文件
                        if os.path.isfile(source_file):
                            shutil.move(source_file, target_file)
                            print(f"移动文件 {filename} 到文件夹 {prefix_num}")
                
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")

if __name__ == "__main__":
    organize_files('/Users/niwenyu/Desktop/OCR_PDF_EXTRACT/work/output')