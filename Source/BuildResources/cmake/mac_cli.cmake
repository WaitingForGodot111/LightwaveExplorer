enable_language(C)
conditionally_fetch_dependencies()

find_package(OpenMP)
add_definitions(-DCPUONLY -DNOSYCL -DNDEBUG)
add_compile_options(-O3 -Wall -mmacosx-version-min=13.3 -march=native)

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
