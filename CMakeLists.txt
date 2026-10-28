cmake_minimum_required(VERSION 3.10)
project(RayTracingRealTime)

# C++の標準規格を設定（チュートリアルはモダンなC++を使います）
set ( CMAKE_CXX_STANDARD          17 )
set ( CMAKE_CXX_STANDARD_REQUIRED ON )
set ( CMAKE_CXX_EXTENSIONS        OFF )

set ( SOURCE_CPP
 src/main.cc
 src/camera.h
 src/vec3.h
 src/color.h
 src/material.h
 src/texture.h
 src/rtw_stb_image.h
 src/ray.h
 src/hittable.h
 src/hittable_list.h
 src/sphere.h
 src/quad.h
 src/interval.h
 src/aabb.h
 src/perlin.h
 src/bvh.h
 src/constant_medium.h
 src/rtweekend.h
 src/external/stb_image.h
)

set ( SOURCE_SHADER
 src/shader/RayTracingCS.hlsl
 src/shader/DisplayCS.hlsl
 src/shader/DenoiseCS.hlsl 
)

# シェーダー群をVSの事前コンパイルから一括除外
set_source_files_properties(${SOURCE_SHADER} PROPERTIES HEADER_FILE_ONLY TRUE)

# 実行フォルダ「RealTime」と、コンパイルするソースコードを指定
add_executable(RealTime ${SOURCE_CPP} ${SOURCE_SHADER})

# プロジェクト全体ではなく、RealTimeターゲットに対してインクルードパスを設定
target_include_directories(RealTime PRIVATE src)

# DirectX用のライブラリを追加
target_link_libraries(RealTime PRIVATE d3d11 d3dcompiler dxgi)

# --- シェーダーファイルの自動コピー設定 ---
add_custom_command(TARGET RealTime POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_CURRENT_SOURCE_DIR}/src/shader"
    "$<TARGET_FILE_DIR:RealTime>/shader"
    COMMENT "Copying shader directory to output directory"
)