import ast

def convert_to_clean_csv(input_data, output_file='final_converted_data.csv'):
    """
    将数据转换为您要求的格式：第三列是没有引号的列表格式
    
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
            
        # 找到前两个逗号的位置
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
    
    # 手动写入CSV文件，确保列表格式正确
    with open(output_file, 'w', encoding='utf-8') as f:
        # 写入表头
        f.write('Column1,Column2,List_Column\n')
        
        # 写入数据行
        for row in processed_data:
            col1, col2, list_data = row
            
            # 将列表转换为字符串，保持 [1, 2, 3] 格式
            list_str = str(list_data)
            
            # 直接写入，不给列表加引号
            f.write(f'{col1},{col2},{list_str}\n')
    
    print(f"数据转换完成！已保存到: {output_file}")
    print(f"处理了 {len(processed_data)} 行数据")
    
    # 显示转换结果
    print("\n转换后的数据:")
    with open(output_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            print(f"第{i+1}行: {line.strip()}")
    
    return processed_data

def main():
    # 您提供的示例数据
    sample_data = '''766,59,"[42, 43, 41, 44, 40, 154, 9, 10, 193, 192, 191, 189, 190, 13, 12, 11, 157, 151, 150, 145, 153, 152, 147, 149, 146, 148, 144, 187, 188, 155, 156, 185, 173, 182, 180, 176, 184, 179, 178, 175, 181, 177, 183, 174, 186]",0
766,120,"[42, 43, 41, 44, 40, 154, 9, 10, 193, 192, 191, 189, 190, 13, 12, 11, 157, 151, 150, 145, 153, 152, 147, 149, 146, 148, 144, 187, 188, 155, 156, 185, 173, 182, 180, 176, 184, 179, 178, 175, 181, 177, 183, 174, 186]",0
766,91,"[42, 43, 41, 44, 40, 154, 9, 10, 193, 192, 191, 189, 190, 13, 12, 11, 157, 151, 150, 145, 153, 152, 147, 149, 146, 148, 144, 187, 188, 155, 156, 185, 173, 182, 180, 176, 184, 179, 178, 175, 181, 177, 183, 174, 186]",0
766,52,"[42, 43, 41, 44, 40, 154, 9, 10, 193, 192, 191, 189, 190, 13, 12, 11, 157, 151, 150, 145, 153, 152, 147, 149, 146, 148, 144, 187, 188, 155, 156, 185, 173, 182, 180, 176, 184, 179, 178, 175, 181, 177, 183, 174, 186]",0'''
    
    # 转换数据
    result = convert_to_clean_csv(sample_data)
    
    print("\n" + "="*60)
    print("转换完成！现在第三列是没有引号的列表格式。")
    print("如果您需要处理其他数据，可以调用：")
    print("convert_to_clean_csv(your_data, 'your_output_file.csv')")
    
    # 验证结果
    print("\n验证：读取生成的CSV文件")
    with open('final_converted_data.csv', 'r', encoding='utf-8') as f:
        content = f.read()
        print("文件内容预览（前500字符）:")
        print(content[:500] + "..." if len(content) > 500 else content)

if __name__ == "__main__":
    main()