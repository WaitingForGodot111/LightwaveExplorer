enable_language(C)

macro(conditionally_fetch_dependencies)
    if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/dlib)
        message("Using existing dlib clone")
    else()
        execute_process(COMMAND wget -q https://github.com/davisking/dlib/archive/refs/tags/v19.24.6.zip)
        execute_process(COMMAND unzip -qq -o v19.24.6.zip)
        execute_process(COMMAND mv dlib-19.24.6 dlib)
    endif()

    if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/gcem)
        message("Using existing gcem clone")
    else()
        execute_process(COMMAND wget -q https://github.com/kthohr/gcem/archive/refs/tags/v1.18.0.zip)
        execute_process(COMMAND unzip -qq -o v1.18.0.zip)
        execute_process(COMMAND mv gcem-1.18.0 gcem)
    endif()

    if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/miniz)
        message("Using existing miniz download")
    else()
        execute_process(COMMAND wget -q https://github.com/richgel999/miniz/releases/download/3.1.0/miniz-3.1.0.zip)
        execute_process(COMMAND unzip -qq -o miniz-3.1.0 -d miniz)
    endif()

    include_directories(SYSTEM ${CMAKE_CURRENT_BINARY_DIR}/dlib)
    include_directories(SYSTEM ${CMAKE_CURRENT_BINARY_DIR}/gcem/include)
    include_directories(SYSTEM ${CMAKE_CURRENT_BINARY_DIR}/miniz)
endmacro()

conditionally_fetch_dependencies()

find_package(OpenMP)

add_definitions(-DCPUONLY -DNOSYCL)
add_library(miniz STATIC miniz/miniz.c)

add_executable(${EXECUTABLE_NAME}
    Source/LightwaveExplorerCommandLineMain.cpp
    Source/LightwaveExplorerUtilities.cpp
    Source/Devices/LightwaveExplorerCoreCPU.cpp
    Source/Devices/DlibLibraryComponents.cpp)

target_link_libraries(${EXECUTABLE_NAME} miniz)

if(OpenMP_FOUND)
    target_compile_options(${EXECUTABLE_NAME} PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(${EXECUTABLE_NAME} ${OpenMP_CXX_LIBRARIES})
endif()
