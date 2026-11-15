/* The model and the clips straight from their files. Embedded with .incbin rather than
 * EMBED_FILES because flatbuffers and int16 samples need alignment that one doesn't give. */
    .section .rodata.yamnet_blobs, "a"
    .balign 16
    .global yamnet_model
yamnet_model:
    .incbin "build_py/yamnet_core_int8.tflite"
    .balign 16
    .global yamnet_clips
yamnet_clips:
    .incbin "firmware/main/clips.bin"
