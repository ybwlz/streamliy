import os

# ========== 获取路径信息 ==========
# 1. 获取当前工作目录（运行命令的位置）
current_dir = os.getcwd()  # 示例: "C:/Users/name/桌面"

# 2. 获取脚本文件的绝对路径
script_path = os.path.abspath(__file__)  # 示例: "C:/项目/app.py"

# 3. 获取脚本所在目录（去掉文件名）
script_dir = os.path.dirname(script_path)  # 示例: "C:/项目"

# 4. 获取文件名（去掉路径）
file_name = os.path.basename(script_path)  # 示例: "app.py"

# ========== 路径组合 ==========
# 5. 智能拼接路径（自动处理不同系统的斜杠）
path1 = os.path.join("文件夹", "子文件夹", "文件.txt")
# Windows: "文件夹\子文件夹\文件.txt"
# Mac/Linux: "文件夹/子文件夹/文件.txt"

# 6. 分离文件名和扩展名
name, ext = os.path.splitext("data.csv")  # name="data", ext=".csv"

# ========== 文件和目录操作 ==========
# 7. 检查文件/目录是否存在
if os.path.exists("某个文件.txt"):
    print("文件存在")
    
# 8. 检查是否是文件
if os.path.isfile("路径"):
    print("这是文件")
    
# 9. 检查是否是目录
if os.path.isdir("路径"):
    print("这是文件夹")

# 10. 列出目录下的所有文件
files = os.listdir("目录路径")  # 返回文件名列表，比如['app.py', 'jiaoyi.csv', 'data.csv', 'image.png', 'readme.txt']

# 11. 创建目录
os.makedirs("新文件夹/子文件夹", exist_ok=True)  # 创建多级目录