cmake_minimum_required(VERSION 3.21)

# C++ host + HIP device kernels (AMD ROCm).
# Override: -DCMAKE_HIP_ARCHITECTURES=gfx1100
if(NOT DEFINED CMAKE_HIP_ARCHITECTURES OR CMAKE_HIP_ARCHITECTURES STREQUAL "")
  set(CMAKE_HIP_ARCHITECTURES "gfx906;gfx1030;gfx1100")
endif()

project(welvet_rocm VERSION 0.52 LANGUAGES CXX HIP)

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

# Device kernels only (HIP language).
add_library(welvet_hip_kernels OBJECT src/rocm/kernels.hip)
set_target_properties(welvet_hip_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(welvet_hip_kernels
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_definitions(welvet_hip_kernels PRIVATE WELVET_ROCM=1)
target_link_libraries(welvet_hip_kernels PUBLIC hip::device)

set(WELVET_HOST_SOURCES
  src/core/status.cpp
  src/core/dtype.cpp
  src/core/activate.cpp
  src/core/tensor.cpp
  src/core/float_codecs.cpp
  src/rocm/device.cpp
  src/rocm/buffer.cpp
  src/weights/store.cpp
  src/weights/native_codec.cpp
  src/layers/dense/dense.cpp
  src/layers/dense/session.cpp
  src/layers/rmsnorm/rmsnorm.cpp
  src/layers/layernorm/layernorm.cpp
  src/layers/softmax/softmax.cpp
  src/layers/swiglu/swiglu.cpp
  src/layers/embedding/embedding.cpp
  src/layers/mha/mha.cpp
  src/layers/cnn1/cnn1.cpp
  src/layers/cnn2/cnn2.cpp
  src/layers/cnn3/cnn3.cpp
  src/layers/convt1/convt1.cpp
  src/layers/convt2/convt2.cpp
  src/layers/convt3/convt3.cpp
  src/layers/rnn/rnn.cpp
  src/layers/lstm/lstm.cpp
  src/layers/sequential/sequential.cpp
  src/layers/residual/residual.cpp
  src/layers/stack/stack.cpp
  src/layers/mamba/mamba.cpp
  src/layers/metacognition/metacognition.cpp
  src/layers/gdn/gdn.cpp
  src/layers/kmeans/kmeans.cpp
  src/layers/parallel/parallel.cpp
  src/architecture/grid.cpp
  src/runtime/forward.cpp
  src/runtime/backward.cpp
  src/runtime/training.cpp
  src/runtime/line.cpp
  src/runtime/step.cpp
  src/runtime/tween_train.cpp
  src/runtime/tween_chain_train.cpp
  src/runtime/tween_split_train.cpp
  src/runtime/tween_alt_train.cpp
  src/runtime/cameral_train.cpp
  src/systems/tween.cpp
  src/systems/tween_chain.cpp
  src/systems/tween_split.cpp
  src/systems/tween_alt.cpp
  src/systems/cam_sync.cpp
  src/gaps/registry.cpp
  src/interop/c_abi.cpp
  src/interop/model_session.cpp
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
elseif(MSVC)
  target_compile_options(welvet_host_obj PRIVATE /fp:precise)
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
# Do not PUBLIC-link hip::device (propagates -x hip to dependents).
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
  add_executable(welvet_dense_fwd
    tools/dense_fwd/main.cpp
    tools/dense_fwd/volume_fwd.cpp
    tools/dense_fwd/stepbp_wires.cpp
  )
  target_include_directories(welvet_dense_fwd PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tools/dense_fwd)
  welvet_link_engine(welvet_dense_fwd)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(welvet_dense_fwd PRIVATE -ffp-contract=off)
  elseif(MSVC)
    target_compile_options(welvet_dense_fwd PRIVATE /fp:precise)
  endif()
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
