import os
import zipfile
import shutil

# 设置源文件夹和目标压缩文件路径
source_folder = os.path.abspath('.')
destination_zip = os.path.join(os.path.dirname(source_folder), 'hand-gesture-recognition.zip')

# 确保目标文件不存在
if os.path.exists(destination_zip):
    os.remove(destination_zip)

print(f"开始创建压缩包：{destination_zip}")
print(f"源文件夹：{source_folder}")

# 创建压缩包
with zipfile.ZipFile(destination_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
    # 遍历源文件夹中的所有文件和子文件夹
    for root, dirs, files in os.walk(source_folder):
        # 排除 .trae 目录
        if '.trae' in dirs:
            dirs.remove('.trae')
        
        # 排除 Python 安装包文件（如果存在）
        if 'python-3.10.11-amd64-installer.exe' in files:
            files.remove('python-3.10.11-amd64-installer.exe')
        
        # 排除当前脚本文件
        if 'create_zip.py' in files:
            files.remove('create_zip.py')
        
        # 遍历文件并添加到压缩包
        for file in files:
            # 获取文件的完整路径
            file_path = os.path.join(root, file)
            # 获取文件在压缩包中的相对路径
            arcname = os.path.relpath(file_path, os.path.dirname(source_folder))
            # 添加文件到压缩包
            zipf.write(file_path, arcname)
            print(f"已添加: {arcname}")

print(f"压缩包创建完成：{destination_zip}")
print(f"文件大小: {os.path.getsize(destination_zip) / 1024:.2f} KB")
