
import streamlit as st
import time 

def main():
    st.title("欢迎使用我的Streamlit应用")
    st.write("这是一个简单的Streamlit应用示例。")
    if st.button("点击这里显示当前时间"):
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) #time.strftime()是Python中的一个函数，用于将时间元组转换为指定格式的字符串表示,
        #localtime()是time模块中的一个函数，用于获取当前的本地时间，并返回一个时间元组
        st.write(f"当前时间是：{current_time}")

if __name__ == "__main__": #__name__是Python中的一个内置变量，用于判断当前模块是否是主程序入口点,这段代码的作用是确保当脚本被直接运行时，才会调用main()函数
    #而main()函数包含了Streamlit应用的主要逻辑，在上面的代码中定义了，对比falsk中的app.run()，只不过run没有在程序中定义，是flask框架自带的，而main是程序员自己定义的，flask使用了run（）来启动应用
    #再对比Django中的manage.py文件，里面有execute_from_command_line()函数，也是类似的作用，都是用来启动应用的
    #还有一点需要注意的是，Flask和Django都是Web框架，而Streamlit更侧重于数据应用的快速开发，所以它们的启动方式有所不同
    #在Flask和Django中，通常需要配置路由、视图等，而Streamlit则更注重于直接展示数据和交互界面
    #因此，虽然它们的启动方式不同，但目的都是为了运行应用程序
    #总结来说，这段代码的作用是确保当脚本被直接运行时，才会调用main()函数，启动Streamlit应用
    #在Django中，每一个模块有一个__init__.py文件，这个文件的作用是告诉Python这个目录是一个包，可以包含比如models.py, views.py等模块，
    # 相当于把这些模块组织在一起，形成一个完整的应用，就像Flask中的蓝图（Blueprint）一样
    main()