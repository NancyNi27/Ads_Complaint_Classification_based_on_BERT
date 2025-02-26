#!/usr/bin/env python
# -*- coding:utf-8 -*-
# url: https://www.asa.co.nz/decisions/search-browse-decisions/
import os
import re
import csv
import time
import random
import requests


def get_response(url, method, data=None):
    # 延时采集
    time.sleep(random.uniform(1, 3))
    try:
        # 发送请求
        response = requests.request(method, url, headers=headers, data=data)
        if response.status_code == 200:
            return response  # 返回响应对象
        # 返回错误信息
        error_info = f"请求出错:\n响应码:{response.status_code}\nurl:{url}"
        if data:
            error_info += f"请求载荷:\n{data}"
        return error_info
    except Exception as e:
        print(f"error: {e}")
        return f"error: {e}"


def parse_data(data_list, data_path):
    # 遍历数据
    for i, data in enumerate(data_list, start=1):
        try:
            # 提取数据
            pdf_title = data.get("title").strip()   # PDF文件名
            pdf_format = "." + data.get("filename").split(".")[-1]  # PDF文件格式
            pdf_name = re.sub(r'[\\/:*?"<>|]', '_', pdf_title + pdf_format)  # PDF文件名拼接
            pdf_url = data.get("url")   # PDF文件下载地址

            # 获取PDF二进制数据 并保存
            response = get_response(pdf_url, "GET")
            if "请求出错" in response or "error" in response:
                save_error_info(data_path, pdf_name, pdf_url, response)  # 记录错误信息
                print(response)
                continue
            binary_data = response.content
            save_pdf_data(pdf_name, binary_data, data_path)

            print(f"第{i}个文件: {pdf_name} 已保存")
        except Exception as e:
            print(f'第{i}个文件: {data.get("title")} 采集出错：{e}')
            save_error_info(data_path, data.get("title"), data.get("url"), e)   # 记录错误信息
        time.sleep(0.15)
    print()


def save_pdf_data(pdf_name, bin_data, data_path):
    # 保存PDF数据到本地
    try:
        file_path = data_path + pdf_name
        with open(file_path, "wb") as f:
            f.write(bin_data)
    except Exception as e:
        print(f"保存出错：{e}")


def save_error_info(data_path, pdf_name, pdf_url, error_info):
    # 错误信息记录
    # 新建表头
    table_headers = ["所属文件夹", "PDF文件名称", "PDF文件链接", "产生的错误信息"]
    # 判断表头是否存在
    write_header = not os.path.exists('error_info.csv') or os.stat("error_info.csv").st_size == 0
    # 错误信息写入
    with open("error_info.csv", 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(table_headers)
        writer.writerow([data_path, pdf_name, pdf_url, error_info])


def main():
    # 循环采集所有页码的数据
    for page in range(29):
        print(f"正在采集第{page + 1}页的数据...\n")  # 输出提示信息
        # 创建文件夹
        data_path = f"./data{page + 1}/"
        if not os.path.exists(data_path):
            os.mkdir(data_path)
            print(f"创建文件夹 data{page + 1} 成功，第{page + 1}页PDF文件将保存至该文件夹\n")

        # 更新请求载荷
        body["page"] = str(page)
        # 发送请求 获取响应数据
        response = get_response(index_url, "POST", data=body)
        if "请求出错" in response or "error" in response:
            save_error_info(data_path, "", f"page = {page}", response)
            print(response)
            continue
        response_data = response.json()
        # 解析数据
        parse_data(response_data, data_path)


if __name__ == '__main__':
    # 准备请求前置条件
    # 搜索页url
    index_url = 'https://www.asa.co.nz/backend/documentsearch.php'
    # 请求载荷
    body = {
        "keyword": "",
        "year": "",
        "media[]": [
            "1", "3", "5", "7", "2", "4", "6", "8", "9", "11", "14", "10", "12", "15", "16", "18",
            "20", "22", "24", "17", "19", "21", "23", "25", "27", "29", "31", "26", "28", "30", "35",
            "37", "39", "41", "36", "38", "40", "33", "43", "32", "34", "42"
        ],
        "page": "0"
    }
    # 请求头headers
    headers = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    }

    # 爬虫主方法
    main()
