# RootConstantsPerformance
Benchmark for root constants
<pre>
Command Lines Options :
-h, --help:                              print command line options;
--debug:                                 enable debug mode, enable d3d debug layer, set input buffer to all 0, print out results.
--use_root_constants:                    use root constants to upload uniform contents.
--use_uniform_buffer:                    use uniform buffer to upload uiniform contents.
--use_dbo:                               upload all uniforms in a single pass and access them with provided index per dispatch.
--batch_mode:                            use batching mode, batch all dispatch in single command list and submit them once.
                                         Default behaviour is record dispatch in different command lists and submit them every time.
--dispatch_block_num NUM:                setting numbers of dispatched blocks, default value is 512.
--thread_block_size NUM:                 setting 1-dimension block size, default value is 512.
--dispatch_times_per_frame NUM:          setting dispatch numbers per frame, default value is 512.
--frame_num NUM:                         setting running frame numbers, default value is 10.
--constant_upload_bytes NUM:             setting upload constant bytes, default value is 64, min: 16, max: 512, must divide by 16.
--cbuffer_use_scalars:                   compute shader cbuffer uses float4 scalars. This is the default behaviour
--cbuffer_use_array:                     compute shader cbuffer uses float4 array. Only for not root constants uploading mode.
--handle_oob_with_clamp                  using clamp when access cbuffer constants. This affects performance when cbuffer constants is array type.
</pre>

## Batch Mode
This mode tries to simulate a case we observed in TFJS webgpu backend. In TFJS, webgpu backend batch all dispatches in a single submit to achieve a
better performance than dispatch and submit immediately.

## Use Uniform Buffer
In TFJS, Op knows the uniform data (e.g. shape) when it is executing. So TFJS WebGPU backend employees `WriteBuffer` to update such uniforms. The
"use uniform buffer" uploading mode tries to simulate this behaviour by uploading cpu data to upload buffer first and then copy to default buffer.

## Use DBO
DBO is another common use case that developer knows all uniforms and choose which one to use in runtime.
