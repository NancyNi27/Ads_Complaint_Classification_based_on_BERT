import os
import pandas as pd
import time
from pathlib import Path

def process_txt_files(input_dir):
    """
    处理文件夹中的所有txt文件，提取并组织数据
    
    Parameters:
    input_dir (str): 输入文件夹路径
    
    Returns:
    pandas.DataFrame: 包含所有处理后数据的DataFrame
    """
    # 初始化结果列表
    results = []
    
    # 定义列名
    columns = [
        "文件名", "complaint_number", "advertiser", "advertisement",
        "date_of_meeting", "outcome", "complaints", "istr", "iend",
        "len(complaints)", "ttarr(i)", "adver_", "iadstr", "iadend",
        "len(adver)", "ttarr(i)_adver"
    ]
    
    start_time = time.time()
    
    # 遍历文件夹中的所有txt文件
    for file_name in os.listdir(input_dir):
        if not file_name.endswith('.txt'):
            continue
            
        file_path = os.path.join(input_dir, file_name)
        
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 移除空行和处理换行符
        lines = [line.strip() for line in lines if line.strip()]
        
        # 初始化数据字典
        data = {col: '' for col in columns}
        data['文件名'] = file_name
        
        # 提取基本信息
        if len(lines) >= 5:
            data['complaint_number'] = lines[0]
            data['advertiser'] = lines[1]
            data['advertisement'] = lines[2]
            data['date_of_meeting'] = lines[3]
            data['outcome'] = lines[4]
        
        # 查找complaints部分
        istr = 0
        iend = 0
        complaints_text = ""
        
        # 查找起始位置
        for i, line in enumerate(lines[6:], 6):
            lower_line = line.lower()
            if ("summary of the complaints board decision" in lower_line or 
                "complaint:" in lower_line or 
                "complain t:" in lower_line):
                istr = i
                data['ttarr(i)'] = line
                break
        
        # 查找结束位置
        if istr > 0:
            for i, line in enumerate(lines[istr:], istr):
                if len(line.strip()) < 3:
                    iend = i
                    break
            
            # 如果没找到结束位置，使用最后一行
            if iend == 0:
                iend = len(lines)
            
            # 提取complaints文本
            complaints_text = ",".join(lines[istr:iend])
            data['complaints'] = complaints_text
            data['istr'] = istr
            data['iend'] = iend
            data['len(complaints)'] = len(complaints_text)
        
        # 查找advertisement部分
        iadstr = 0
        iadend = 0
        adver_text = ""
        
        # 查找起始位置
        for i, line in enumerate(lines[6:], 6):
            if line.startswith("Advertisement"):
                iadstr = i
                data['ttarr(i)_adver'] = line
                break
        
        # 查找结束位置
        if iadstr > 0:
            for i, line in enumerate(lines[iadstr:], iadstr):
                if len(line.strip()) < 3:
                    iadend = i
                    break
            
            # 如果没找到结束位置，使用最后一行
            if iadend == 0:
                iadend = len(lines)
            
            # 提取advertisement文本
            adver_text = ",".join(lines[iadstr:iadend])
            data['adver_'] = adver_text
            data['iadstr'] = iadstr
            data['iadend'] = iadend
            data['len(adver)'] = len(adver_text)
        
        results.append(data)
    
    # 创建DataFrame
    df = pd.DataFrame(results, columns=columns)
    
    end_time = time.time()
    print(f"处理完毕，共用时：{end_time - start_time:.2f}秒！")
    
    return df

def main():
    # 设置输入文件夹路径
    input_dir = "./output"  # 替换为你的输入文件夹路径
    
    # 确保输入文件夹存在
    if not os.path.exists(input_dir):
        print(f"错误：文件夹 {input_dir} 不存在！")
        return
    
    # 处理文件
    df = process_txt_files(input_dir)
    
    # 保存结果到Excel
    output_file = "results.xlsx"
    df.to_excel(output_file, index=False)
    print(f"结果已保存到 {output_file}")

if __name__ == "__main__":
    main()