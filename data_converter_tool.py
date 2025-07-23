#!/usr/bin/env python3
"""
数据转换工具
将包含字符串格式列表的数据转换为CSV格式，第三列为没有引号的列表格式

使用方法:
1. 从文件转换: python data_converter_tool.py input.txt output.csv
2. 交互式输入: python data_converter_tool.py
3. 从标准输入: cat input.txt | python data_converter_tool.py - output.csv
"""

import ast
import sys
import argparse

def convert_data_to_csv(input_data, output_file):
    """
    将数据转换为CSV格式
    
    Args:
        input_data: 输入数据（字符串或行列表）
        output_file: 输出文件路径
    
    Returns:
        处理的行数
    """
    if isinstance(input_data, str):
        lines = input_data.strip().split('\n')
    else:
        lines = input_data
    
    processed_data = []
    error_count = 0
    
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
            
        try:
            # 找到前两个逗号的位置
            first_comma = line.find(',')
            second_comma = line.find(',', first_comma + 1)
            
            if first_comma == -1 or second_comma == -1:
                print(f"警告: 第{line_num}行格式不正确，跳过: {line[:50]}...")
                error_count += 1
                continue
            
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
            elif rest.startswith("'") and rest.endswith("'"):
                rest = rest[1:-1]
            
            # 解析列表
            actual_list = ast.literal_eval(rest)
            processed_data.append([col1, col2, actual_list])
            
        except (ValueError, SyntaxError) as e:
            print(f"错误: 第{line_num}行解析失败: {e}")
            print(f"问题行: {line}")
            error_count += 1
            continue
    
    # 写入CSV文件
    with open(output_file, 'w', encoding='utf-8') as f:
        # 写入表头
        f.write('Column1,Column2,List_Column\n')
        
        # 写入数据行
        for row in processed_data:
            col1, col2, list_data = row
            list_str = str(list_data)
            f.write(f'{col1},{col2},{list_str}\n')
    
    print(f"转换完成！")
    print(f"- 输出文件: {output_file}")
    print(f"- 成功处理: {len(processed_data)} 行")
    if error_count > 0:
        print(f"- 错误/跳过: {error_count} 行")
    
    return len(processed_data)

def main():
    parser = argparse.ArgumentParser(description='将字符串列表数据转换为CSV格式')
    parser.add_argument('input_file', nargs='?', default=None, 
                       help='输入文件路径（使用 - 表示标准输入，不指定则交互输入）')
    parser.add_argument('output_file', nargs='?', default='converted_output.csv',
                       help='输出CSV文件路径（默认: converted_output.csv）')
    
    args = parser.parse_args()
    
    # 获取输入数据
    if args.input_file == '-':
        # 从标准输入读取
        print("从标准输入读取数据...")
        input_data = sys.stdin.read()
    elif args.input_file:
        # 从文件读取
        try:
            with open(args.input_file, 'r', encoding='utf-8') as f:
                input_data = f.read()
            print(f"从文件读取数据: {args.input_file}")
        except FileNotFoundError:
            print(f"错误: 找不到文件 {args.input_file}")
            return 1
        except Exception as e:
            print(f"错误: 读取文件失败 {e}")
            return 1
    else:
        # 交互式输入
        print("请输入数据（输入完成后按Ctrl+D结束）:")
        print("格式示例: 766,59,\"[42, 43, 41, 44]\",0")
        print("-" * 50)
        try:
            input_data = sys.stdin.read()
        except KeyboardInterrupt:
            print("\n操作被取消")
            return 1
    
    if not input_data.strip():
        print("错误: 没有输入数据")
        return 1
    
    # 转换数据
    try:
        processed_count = convert_data_to_csv(input_data, args.output_file)
        
        # 显示结果预览
        print("\n结果预览:")
        with open(args.output_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i < 5:  # 只显示前5行
                    print(f"  {line.strip()}")
                else:
                    print("  ...")
                    break
        
        return 0
        
    except Exception as e:
        print(f"转换失败: {e}")
        return 1

if __name__ == "__main__":
    exit(main())