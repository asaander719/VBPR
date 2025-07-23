import pandas as pd
import ast
import csv

def convert_string_to_list_csv(input_data, output_file='converted_data.csv'):
    """
    将包含字符串格式列表的数据转换为实际列表，并保存为CSV
    
    Args:
        input_data: 输入数据（可以是字符串、列表或文件路径）
        output_file: 输出CSV文件路径
    """
    
    # 如果输入是字符串数据，先解析为行
    if isinstance(input_data, str):
        lines = input_data.strip().split('\n')
    elif isinstance(input_data, list):
        lines = input_data
    else:
        # 假设是文件路径
        with open(input_data, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    
    # 解析数据
    processed_data = []
    
    for line in lines:
        if isinstance(line, str):
            line = line.strip()
        if not line:
            continue
            
        # 分割每行数据
        parts = line.split(',', 2)  # 只分割前两个逗号
        
        if len(parts) >= 3:
            col1 = parts[0].strip()
            col2 = parts[1].strip()
            
            # 处理第三列的字符串列表
            list_str = parts[2].strip()
            
            # 移除最后的,0如果存在
            if list_str.endswith(',0'):
                list_str = list_str[:-2]
            
            # 去掉引号
            if list_str.startswith('"') and list_str.endswith('"'):
                list_str = list_str[1:-1]
            
            # 将字符串转换为实际的列表
            try:
                actual_list = ast.literal_eval(list_str)
                processed_data.append([col1, col2, actual_list])
            except (ValueError, SyntaxError) as e:
                print(f"解析错误: {line}")
                print(f"错误详情: {e}")
                continue
    
    # 保存为CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # 写入表头
        writer.writerow(['Column1', 'Column2', 'List_Column'])
        
        # 写入数据
        for row in processed_data:
            # 将列表转换为没有引号的字符串格式
            list_str = str(row[2]).replace("'", "")
            writer.writerow([row[0], row[1], list_str])
    
    print(f"数据转换完成！已保存到: {output_file}")
    print(f"处理了 {len(processed_data)} 行数据")
    
    # 显示前几行作为示例
    print("\n转换后的数据示例:")
    with open(output_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            print(line.strip())
            if i >= 4:  # 只显示前5行
                break
    
    return processed_data

def main():
    # 您提供的示例数据
    sample_data = '''766,59,"[42, 43, 41, 44, 40, 154, 9, 10, 193, 192, 191, 189, 190, 13, 12, 11, 157, 151, 150, 145, 153, 152, 147, 149, 146, 148, 144, 187, 188, 155, 156, 185, 173, 182, 180, 176, 184, 179, 178, 175, 181, 177, 183, 174, 186]",0
766,120,"[42, 43, 41, 44, 40, 154, 9, 10, 193, 192, 191, 189, 190, 13, 12, 11, 157, 151, 150, 145, 153, 152, 147, 149, 146, 148, 144, 187, 188, 155, 156, 185, 173, 182, 180, 176, 184, 179, 178, 175, 181, 177, 183, 174, 186]",0
766,91,"[42, 43, 41, 44, 40, 154, 9, 10, 193, 192, 191, 189, 190, 13, 12, 11, 157, 151, 150, 145, 153, 152, 147, 149, 146, 148, 144, 187, 188, 155, 156, 185, 173, 182, 180, 176, 184, 179, 178, 175, 181, 177, 183, 174, 186]",0
766,52,"[42, 43, 41, 44, 40, 154, 9, 10, 193, 192, 191, 189, 190, 13, 12, 11, 157, 151, 150, 145, 153, 152, 147, 149, 146, 148, 144, 187, 188, 155, 156, 185, 173, 182, 180, 176, 184, 179, 178, 175, 181, 177, 183, 174, 186]",0'''
    
    # 转换数据
    convert_string_to_list_csv(sample_data, 'converted_data.csv')
    
    # 如果您有一个输入文件，可以这样使用：
    # convert_string_to_list_csv('input_file.txt', 'output_file.csv')

if __name__ == "__main__":
    main()