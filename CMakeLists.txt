cmake_minimum_required(VERSION 3.21)

# Normal C++ host + HIP device kernels (AMD ROCm).
# Override GPU arch: -DCMAKE_HIP_ARCHITECTURES=gfx906;gfx1030;gfx1100
if(NOT DEFINED CMAKE_HIP_ARCHITECTURES OR CMAKE_HIP_ARCHITECTURES STREQUAL "")
  set(CMAKE_HIP_ARCHITECTURES "gfx906;gfx1030;gfx1100")
endif()

project(welvet_rocm VERSION 0.0 LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(WELVET_BUILD_SHARED "Build shared library (C ABI / NAC)" ON)
option(WELVET_BUILD_TOOLS "Build CLI tools for w2a go_v_rocm" ON)
option(WELVET_BUILD_EXAMPLES "Build standalone ROCm smoke examples" ON)

find_package(hip REQUIRED CONFIG)

include(GNUInstallDirs)

# Device kernels only — HIP language (do not mix into host OBJECT with hip::device).
add_library(welvet_hip_kernels OBJECT src/rocm/kernels.hip)
set_target_properties(welvet_hip_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(welvet_hip_kernels
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_definitions(welvet_hip_kernels PRIVATE WELVET_ROCM=1)
target_link_libraries(welvet_hip_kernels PUBLIC hip::device)

# Host C++ — hip::host for hip_runtime.h without -x hip.
set(WELVET_HOST_SOURCES
  src/core/status.cpp
  src/rocm/device.cpp
  src/rocm/buffer.cpp
  src/gaps/registry.cpp
  src/interop/c_abi.cpp
)
add_library(welvet_host_obj OBJECT ${WELVET_HOST_SOURCES})
set_target_properties(welvet_host_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(welvet_host_obj
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_definitions(welvet_host_obj PRIVATE WELVET_ROCM=1)
if(WELVET_BUILD_SHARED)
  target_compile_definitions(welvet_host_obj PRIVATE WELVET_BUILD_SHARED WELVET_EXPORTS)
endif()
target_link_libraries(welvet_host_obj PUBLIC hip::host)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(welvet_host_obj PRIVATE -ffp-contract=off)
endif()

if(WELVET_BUILD_SHARED)
  add_library(welvet SHARED
    $<TARGET_OBJECTS:welvet_host_obj>
    $<TARGET_OBJECTS:welvet_hip_kernels>
  )
else()
  add_library(welvet STATIC
    $<TARGET_OBJECTS:welvet_host_obj>
    $<TARGET_OBJECTS:welvet_hip_kernels>
  )
endif()
add_library(welvet::welvet ALIAS welvet)
target_include_directories(welvet
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
# Host dependents must not inherit hip::device's `-x hip` compile flag.
# Device runtime is PRIVATE so tools/examples stay plain CXX against the .so.
target_link_libraries(welvet PUBLIC hip::host)
target_link_libraries(welvet PRIVATE hip::device)
set_target_properties(welvet PROPERTIES
  OUTPUT_NAME welvet_rocm
  POSITION_INDEPENDENT_CODE ON
)

function(welvet_link_engine target)
  target_link_libraries(${target} PRIVATE welvet)
endfunction()

if(WELVET_BUILD_TOOLS)
  add_executable(welvet_dense_fwd tools/dense_fwd/main.cpp)
  welvet_link_engine(welvet_dense_fwd)
endif()

if(WELVET_BUILD_EXAMPLES)
  add_executable(example_rocm_smoke examples/rocm_smoke/main.cpp)
  welvet_link_engine(example_rocm_smoke)
endif()

install(TARGETS welvet
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY kernels/ DESTINATION ${CMAKE_INSTALL_DATADIR}/welvet/kernels)
if(WELVET_BUILD_TOOLS)
  install(TARGETS welvet_dense_fwd RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

message(STATUS "welvet.rocm platform: ${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "HIP: ${hip_VERSION}")
message(STATUS "HIP archs: ${CMAKE_HIP_ARCHITECTURES}")
message(STATUS "CXX: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
