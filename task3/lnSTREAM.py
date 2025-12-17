import streamlit as st
import pandas as pd
import time



# ========== 页面配置（必须在最前面）==========
st.set_page_config(
    page_title="我的应用",  # 浏览器标签页标题
    page_icon="📊",         # 浏览器标签页图标
    layout="wide",         # 页面布局："centered"窄版/"wide"宽版
    initial_sidebar_state="expanded"  # 侧边栏初始状态
)

# ========== 文本显示 ==========
st.title("大标题")          # 最大标题
st.header("中标题")         # 中等标题
st.subheader("小标题")      # 小标题
st.write("普通文本")        # 通用显示，可显示任何内容
st.markdown("**Markdown文本**")  # 支持Markdown语法
st.text("纯文本")           # 固定宽度的文本
st.latex(r"E = mc^2")      # 显示数学公式

# ========== 输入控件 ==========
# 1. 按钮
if st.button("点击我"):
    st.write("按钮被点击了！")

# 2. 输入框
name = st.text_input("请输入姓名", "张三")  # 默认值"张三"

# 3. 下拉选择框
option = st.selectbox("请选择", ["选项1", "选项2", "选项3"])

# 4. 滑动条
value = st.slider("选择一个数值", 0, 100, 50)  # 最小值0,最大值100,默认50

# 5. 复选框
if st.checkbox("显示详情"):
    st.write("详情内容...")

# 6. 文件上传
uploaded_file = st.file_uploader("选择CSV文件", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)

# ========== 数据显示 ==========
# 显示表格
st.dataframe(df)           # 交互式表格
st.table(df.head())        # 静态表格

# 显示JSON/字典
st.json({"key": "value"})  # 格式化的JSON

# ========== 图表显示 ==========
# 支持多种图表库
st.line_chart(df)          # 折线图
st.bar_chart(df)           # 柱状图
st.area_chart(df)          # 面积图

# ========== 布局组件 ==========
# 1. 侧边栏（所有st.sidebar.xxx会显示在侧边）
st.sidebar.title("侧边栏标题")
selection = st.sidebar.selectbox("侧边选择", ["A", "B"])

# 2. 列布局
col1, col2, col3 = st.columns(3)  # 创建3列
with col1:
    st.write("第一列内容")
with col2:
    st.write("第二列内容")

# 3. 选项卡
tab1, tab2 = st.tabs(["选项卡1", "选项卡2"])
with tab1:
    st.write("选项卡1内容")
with tab2:
    st.write("选项卡2内容")

# 4. 容器
with st.container():
    st.write("容器内的内容")

# 5. 展开折叠
with st.expander("点击展开详情"):
    st.write("这里是详细信息...")

# ========== 状态消息 ==========
st.success("成功消息！")    # 绿色成功提示
st.info("信息提示")        # 蓝色信息提示，infomation,信息
st.warning("警告信息")      # 黄色警告提示
st.error("错误信息")       # 红色错误提示
st.exception(e)            # 显示异常信息# type: ignore 

# ========== 进度和状态 ==========
# 进度条
progress_bar = st.progress(0)
for i in range(100):
    progress_bar.progress(i + 1)

# 加载动画
with st.spinner("正在加载..."):
    time.sleep(2)  # 模拟耗时操作
st.success("加载完成！")

# ========== 缓存（性能优化） ==========
@st.cache_data  # 缓存数据（不会变的数据）
def load_data():
    return pd.read_csv("大文件.csv")

@st.cache_resource  # 缓存资源（模型、连接等）
def load_model():
    return expensive_model()# type: ignore