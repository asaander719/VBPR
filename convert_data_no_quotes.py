import ast
import csv

def convert_to_csv_no_quotes(input_data, output_file='converted_data_no_quotes.csv'):
    """
    将包含字符串格式列表的数据转换为没有引号的列表格式，并保存为CSV
    
    Args:
        input_data: 输入数据（字符串）
        output_file: 输出CSV文件路径
    """
    
    # 解析输入数据
    lines = input_data.strip().split('\n')
    processed_data = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 分割每行数据 - 更仔细地处理
        # 找到第二个逗号的位置
        first_comma = line.find(',')
        second_comma = line.find(',', first_comma + 1)
        
        if first_comma != -1 and second_comma != -1:
            col1 = line[:first_comma].strip()
            col2 = line[first_comma+1:second_comma].strip()
            
            # 获取第三列（列表部分）
            rest = line[second_comma+1:].strip()
            
            # 移除最后的,0如果存在
            if rest.endswith(',0'):
                rest = rest[:-2]
            
            # 去掉外层引号
            if rest.startswith('"') and rest.endswith('"'):
                rest = rest[1:-1]
            
            # 解析列表
            try:
                actual_list = ast.literal_eval(rest)
                processed_data.append([col1, col2, actual_list])
            except (ValueError, SyntaxError) as e:
                print(f"解析错误: {line}")
                print(f"错误详情: {e}")
                continue
    
    # 保存为CSV，使用自定义格式确保列表没有引号
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        # 写入表头
        csvfile.write('Column1,Column2,List_Column\n')
        
        # 写入数据
        for row in processed_data:
            col1, col2, list_data = row
            
            # 将列表转换为没有引号的字符串格式
            list_str = str(list_data)  # 这会产生 [1, 2, 3] 格式
            
            # 写入行，只对包含逗号的字段加引号
            if ',' in col1:
                col1 = f'"{col1}"'
            if ',' in col2:
                col2 = f'"{col2}"'
            
            # 列表部分需要用引号包围，因为它包含逗号
            csvfile.write(f'{col1},{col2},"{list_str}"\n')
    
    print(f"数据转换完成！已保存到: {output_file}")
    print(f"处理了 {len(processed_data)} 行数据")
    
    # 显示转换后的数据
    print("\n转换后的数据:")
    with open(output_file, 'r', encoding='utf-8') as f:
        content = f.read()
        print(content)
    
    return processed_data

def main():
    # 您提供的示例数据
    sample_data = '''766,59,"[42, 43, 41, 44, 40, 154, 9, 10, 193, 192, 191, 189, 190, 13, 12, 11, 157, 151, 150, 145, 153, 152, 147, 149, 146, 148, 144, 187, 188, 155, 156, 185, 173, 182, 180, 176, 184, 179, 178, 175, 181, 177, 183, 174, 186]",0
766,120,"[42, 43, 41, 44, 40, 154, 9, 10, 193, 192, 191, 189, 190, 13, 12, 11, 157, 151, 150, 145, 153, 152, 147, 149, 146, 148, 144, 187, 188, 155, 156, 185, 173, 182, 180, 176, 184, 179, 178, 175, 181, 177, 183, 174, 186]",0
766,91,"[42, 43, 41, 44, 40, 154, 9, 10, 193, 192, 191, 189, 190, 13, 12, 11, 157, 151, 150, 145, 153, 152, 147, 149, 146, 148, 144, 187, 188, 155, 156, 185, 173, 182, 180, 176, 184, 179, 178, 175, 181, 177, 183, 174, 186]",0
766,52,"[42, 43, 41, 44, 40, 154, 9, 10, 193, 192, 191, 189, 190, 13, 12, 11, 157, 151, 150, 145, 153, 152, 147, 149, 146, 148, 144, 187, 188, 155, 156, 185, 173, 182, 180, 176, 184, 179, 178, 175, 181, 177, 183, 174, 186]",0'''
    
    # 转换数据
    convert_to_csv_no_quotes(sample_data)
    
    print("\n" + "="*50)
    print("如果您有其他数据文件，可以这样使用：")
    print("from convert_data_no_quotes import convert_to_csv_no_quotes")
    print("convert_to_csv_no_quotes(your_data, 'your_output.csv')")

if __name__ == "__main__":
    main()