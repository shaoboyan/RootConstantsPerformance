# RootConstantsPerformance
Benchmark for root constants
<pre>
Command Lines Options :
-h, --help:                              print command line options;
--debug:                                 enable debug mode, enable d3d debug layer, set input buffer to all 0, print out results.
--use_root_constants:                    use root constants to upload uniform contents. Default behaviour is using uniform buffer.
--use_uniform_buffer:                    use uniform buffer to upload uiniform contents. Default behaviour is using uniform buffer.
--batch_mode:                            use batching mode, batch all dispatch in single command list and submit them once.
                                         Default behaviour is record dispatch in different command lists and submit them every time.
--dispatch_block_num NUM:                setting numbers of dispatched blocks, default value is 128.
--thread_block_size NUM:                 setting 1-dimension block size, default value is 32.
--dispatch_times_per_frame NUM:          setting dispatch numbers per frame, default value is 256.
--frame_num NUM:                         setting running frame numbers, default value is 3.
--constant_upload_bytes NUM:             setting upload constant bytes, default value is 64, min: 16, max: 512, must divide by 16.
</pre>
