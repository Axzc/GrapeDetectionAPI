# GrapeDetectionAPI

### 安装opencv

**系统环境为ubuntu18.04**

1.下载

安装版本为OpenCV – 3.4.8[下载链接](http://opencv.org/releases.html)
然后选择 源码安装 sources

2.安装
``` shell script
# 解压zip
unzip opencv-3.4.8.zip; cd opencv-3.4.8
# 安装工具和依赖
sudo apt-get install cmake  
sudo apt-get install build-essential libgtk2.0-dev libavcodec-dev libavformat-dev libjpeg.dev libtiff4.dev libswscale-dev libjasper-dev
# cmake && make
mkdir build; cd build; cmake ..; sudo make
sudo make install
```

3.编译环境

- 将opencv的库添加到系统路径
```shel script
sudo vim /etc/ld.so.conf.d/opencv.conf
# 在文件中添加
/usr/local/lib  
sudo ldconfig  #使路径生效
```

- 配置bash
```shell srcipt
sudo vim /etc/bash.bashrc 
# 在末尾加入
PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig  
export PKG_CONFIG_PATH
# 使设置生效
source /etc/bash.bashrc
# 更新
sudo updatedb
```

4.测试

```shell script
进入目录
/opencv-3.4.8/samples/cpp/example_cmake
cmake .
make
./opencv_example
```
执行结束后可以看到opencv,就是成功了


### 将.cpp文件生成.so动态链接库

- 创建CMakeLists.txt

```cpp
创建项目目录
mkdir get3dfixed
创建CMakeLists.txt
```
CMakeLists.txt 内容为
```cpp
# CMake 的版本
cmake_minimum_required(VERSION 2.8)

# 指定项目名称
project(get3dfixeds1)

# 调用 opencv 模块
find_package(OpenCV REQUIRED)
 
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

if(CMAKE_VERSION VERSION_LESS "2.8.11")
    include_directories(${OpenCV_INCLUDE_DIRS})  # 增加opencv路径
endif()

# add_executable(get3dfixeds1 get3DFixed.cpp)
# 生成动态库或共享库 .so
add_library(get3dfixeds1 SHARED get3DFixed.cpp)
# 为get3dfixeds1 添加openv的动态链接库
target_link_libraries(get3dfixeds1 ${OpenCV_LIBS})
```

- 生成 .so文件
```shell script
mkdir build; cd build; cmake ..; make
如果修改了cpp代码可以在build中重新生成.so
cmake clean; cmake ..; make
```

### 安装依赖
```
pip install -r requirements.txt
```
