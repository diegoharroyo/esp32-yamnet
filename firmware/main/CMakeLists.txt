idf_component_register(SRCS main.cc frontend.c blobs.S
                       INCLUDE_DIRS .
                       EMBED_TXTFILES ../../tools/yamnet_ref/yamnet_class_map.csv)

# blobs.S .incbin's paths are relative to the repo root.
set(root ${CMAKE_CURRENT_LIST_DIR}/../..)
target_compile_options(${COMPONENT_LIB} PRIVATE "$<$<COMPILE_LANGUAGE:ASM>:-Wa,-I${root}>")
set_property(SOURCE blobs.S APPEND PROPERTY OBJECT_DEPENDS
             ${root}/build_py/yamnet_core_int8.tflite ${CMAKE_CURRENT_LIST_DIR}/clips.bin)
