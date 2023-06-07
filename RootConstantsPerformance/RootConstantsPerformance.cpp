// ConsoleApplication1.cpp : This file contains the 'main' function. Program execution begins and
// ends there.
//

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include <D3dcompiler.h>
#include <wrl.h>
#include "d3dx12.h"
#include "dxgi1_4.h"

using Microsoft::WRL::ComPtr;

inline std::string HrToString(HRESULT hr) {
    char s_str[64] = {};
    sprintf_s(s_str, "HRESULT of 0x%08X", static_cast<UINT>(hr));
    return std::string(s_str);
}

class HrException : public std::runtime_error {
  public:
    HrException(HRESULT hr) : std::runtime_error(HrToString(hr)), m_hr(hr) {}
    HRESULT Error() const { return m_hr; }

  private:
    const HRESULT m_hr;
};

#define SAFE_RELEASE(p) \
    if (p)              \
    (p)->Release()

inline void ThrowIfFailed(HRESULT hr) {
    if (FAILED(hr)) {
        throw HrException(hr);
    }
}

// Helper function to download results from buffers.
// It download the content based on the dispatch_thread_num.
// It tries to download content from the head, middle and tail
// part of the buffer if thread number is large enough.
void DownloadResultBufferContents(ComPtr<ID3D12Device> device,
                                  ComPtr<ID3D12CommandQueue> command_queue,
                                  ComPtr<ID3D12Resource> result_buffer,
                                  size_t result_buffer_size,
                                  uint32_t result_buffer_index,
                                  uint32_t dispatch_thread_num) {
    ComPtr<ID3D12Resource> download_buffer;
    const D3D12_HEAP_PROPERTIES download_buffer_heap =
        CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
    const D3D12_RESOURCE_DESC download_buffer_resource =
        CD3DX12_RESOURCE_DESC::Buffer(result_buffer_size);
    ThrowIfFailed(device->CreateCommittedResource(
        &download_buffer_heap, D3D12_HEAP_FLAG_NONE, &download_buffer_resource,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&download_buffer)));

    ComPtr<ID3D12CommandAllocator> download_command_allocator;
    ThrowIfFailed(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                 IID_PPV_ARGS(&download_command_allocator)));

    ComPtr<ID3D12GraphicsCommandList> download_command_list;
    ThrowIfFailed(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                            download_command_allocator.Get(), nullptr,
                                            IID_PPV_ARGS(&download_command_list)));

    D3D12_RESOURCE_BARRIER uav_to_copy_src_barrier;
    uav_to_copy_src_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    uav_to_copy_src_barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    uav_to_copy_src_barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    uav_to_copy_src_barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    uav_to_copy_src_barrier.Transition.pResource = result_buffer.Get();
    uav_to_copy_src_barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

    download_command_list->ResourceBarrier(1, &uav_to_copy_src_barrier);
    download_command_list->CopyBufferRegion(download_buffer.Get(), 0, result_buffer.Get(), 0,
                                            result_buffer_size);
    ThrowIfFailed(download_command_list->Close());

    ID3D12CommandList* download_command_lists[] = {download_command_list.Get()};
    command_queue->ExecuteCommandLists(1, download_command_lists);

    HANDLE fence_event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    ComPtr<ID3D12Fence> fence;
    UINT64 signal_fence_value = 1;
    ThrowIfFailed(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));

    ThrowIfFailed(command_queue->Signal(fence.Get(), signal_fence_value));
    if (fence->GetCompletedValue() < signal_fence_value) {
        ThrowIfFailed(fence->SetEventOnCompletion(signal_fence_value, fence_event));
        WaitForSingleObject(fence_event, INFINITE);
    }

    // Print the start, middle and end contents
    float* p_download_data_begin;
    ThrowIfFailed(
        download_buffer->Map(0, nullptr, reinterpret_cast<void**>(&p_download_data_begin)));

    // print the first 8 results
    uint32_t start = 0;
    uint32_t end = dispatch_thread_num > 2 ? 2 : dispatch_thread_num;
    for (uint32_t i = start; i < end; ++i) {
        std::cout << "Results in uav[" << result_buffer_index << "] start position : " << std::endl;
        std::cout << "thread[" << i << "] results: " << std::endl;
        for (uint32_t j = 0; j < 4; ++j) {
            std::cout << static_cast<float>(p_download_data_begin[i * 4 + j]) << " ";
        }
        std::cout << std::endl;
    }

    // print the middle 8 results
    start = dispatch_thread_num / 2;
    end = start + 2;
    if (end < dispatch_thread_num) {
        for (uint32_t i = start; i < end; ++i) {
            std::cout << "Results in uav[" << result_buffer_index
                      << "] middle position : " << std::endl;
            std::cout << "thread[" << i << "] results: " << std::endl;
            for (uint32_t j = 0; j < 4; ++j) {
                std::cout << static_cast<float>(p_download_data_begin[i * 4 + j]) << " ";
            }
            std::cout << std::endl;
        }
    }

    if (dispatch_thread_num > 4) {
        start = dispatch_thread_num - 2;
        end = dispatch_thread_num;
        for (uint32_t i = start; i < end; ++i) {
            std::cout << "Results in uav[" << result_buffer_index
                      << "] end position : " << std::endl;
            std::cout << "thread[" << i << "] results: " << std::endl;
            for (uint32_t j = 0; j < 4; ++j) {
                std::cout << static_cast<float>(p_download_data_begin[i * 4 + j]) << " ";
            }
            std::cout << std::endl;
        }
    }
    download_buffer->Unmap(0, nullptr);

    download_command_allocator->Reset();
}

int main(int argc, char* argv[]) {
    enum UploadMode {
        use_root_constants,
        use_uniform_buffer,
        use_dbo,
    };

    enum CbufferContentType {
        scalar,
        array,
    };

    bool use_batch_mode = false;
    UploadMode upload_mode = use_uniform_buffer;
    CbufferContentType cbuffer_content_type = scalar;
    bool debug_mode = false;
    bool handle_oob_with_clamp = false;
    uint32_t dispatch_block_num = 512;
    uint32_t dispatch_times_per_frame = 512;
    uint32_t frame_num = 10;
    uint32_t thread_block_size = 512;

    // The content size that use root constants or uniform buffer to upload.
    size_t constant_upload_bytes = 64;
    const size_t bytes_per_element = 4;

    // The number of float each thread load and out to result buffer.
    const size_t element_per_thread = 4;
    const size_t buffer_alignment_bytes = 256;
    const uint32_t dispatch_thread_num = thread_block_size * dispatch_block_num;

    for (int i = 0; i < argc; ++i) {
        if (std::string(argv[i]) == "-h" || std::string(argv[i]) == "--help") {
            std::cout << "Command Lines Options :" << std::endl;
            std::cout << "-h, --help:                              print command line options;"
                      << std::endl;
            std::cout << "--debug:                                 enable debug mode, enable d3d "
                         "debug layer, set input buffer to all 0, print out results."
                      << std::endl;
            std::cout << "--use_root_constants:                    use root constants to upload "
                         "uniform contents."
                      << std::endl;
            std::cout << "--use_uniform_buffer:                    use uniform buffer to upload "
                         "uiniform contents."
                      << std::endl;
            std::cout << "--use_dbo:                               upload all uniforms in a single "
                         "pass and access them with provided index per dispatch."
                      << std::endl;
            std::cout << "--batch_mode:                            use batching mode, batch all "
                         "dispatch in single command list and submit them once."
                      << std::endl;
            std::cout << "                                         Default behaviour is record "
                         "dispatch in different command lists and submit them every time."
                      << std::endl;
            std::cout << "--dispatch_block_num NUM:                setting numbers of dispatched "
                         "blocks, default value is 128."
                      << std::endl;
            std::cout << "--thread_block_size NUM:                 setting 1-dimension block size, "
                         "default value is 32."
                      << std::endl;
            std::cout << "--dispatch_times_per_frame NUM:          setting dispatch numbers per "
                         "frame, default value is 256."
                      << std::endl;
            std::cout << "--frame_num NUM:                         setting running frame numbers, "
                         "default value is 3."
                      << std::endl;
            std::cout << "--constant_upload_bytes NUM:             setting upload constant bytes, "
                         "default value is 64, min: 16, max: 512, must divide by 16."
                      << std::endl;
            std::cout
                << "--cbuffer_use_scalars:                    compute shader cbuffer uses float4 "
                   "scalars. This is the default behaviour"
                << std::endl;
            std::cout
                << "--cbuffer_use_array:                      compute shader cbuffer uses float4 "
                   "array. Only for not root constants uploading mode."
                << std::endl;
            std::cout << "--handle_oob_with_clamp                   using clamp when access "
                         "cbuffer constants. "
                         "This affects performance when cbuffer constants is array type. "
                      << std::endl;
            return 1;
        }

        if (std::string(argv[i]) == "--debug") {
            debug_mode = true;
            continue;
        }

        if (std::string(argv[i]) == "--use_root_constants") {
            upload_mode = use_root_constants;
            continue;
        }

        if (std::string(argv[i]) == "--use_uniform_buffer") {
            upload_mode = use_uniform_buffer;
            continue;
        }

        if (std::string(argv[i]) == "--use_dbo") {
            upload_mode = use_dbo;
            continue;
        }

        if (std::string(argv[i]) == "--batch_mode") {
            use_batch_mode = true;
            continue;
        }

        if (std::string(argv[i]) == "--dispatch_block_num") {
            ++i;
            dispatch_block_num = atoi(argv[i]);
            continue;
        }

        if (std::string(argv[i]) == "--thread_block_size") {
            ++i;
            thread_block_size = atoi(argv[i]);
            continue;
        }

        if (std::string(argv[i]) == "--dispatch_times_per_frame") {
            ++i;
            dispatch_times_per_frame = atoi(argv[i]);
            continue;
        }

        if (std::string(argv[i]) == "--frame_num") {
            ++i;
            frame_num = atoi(argv[i]);
            continue;
        }

        if (std::string(argv[i]) == "--constant_upload_bytes") {
            ++i;
            constant_upload_bytes = atoi(argv[i]);
            continue;
        }

        if (std::string(argv[i]) == "--cbuffer_use_scalars") {
            cbuffer_content_type = scalar;
            continue;
        }

        if (std::string(argv[i]) == "--cbuffer_use_array") {
            cbuffer_content_type = array;
            continue;
        }

        if (std::string(argv[i]) == "--handle_oob_with_clamp") {
            handle_oob_with_clamp = true;
            continue;
        }
    }

    std::cout << "Start running .... " << std::endl;
    std::cout << "Parameters:" << std::endl;
    std::cout << "debug_mode: " << debug_mode << std::endl;
    std::cout << "use_root_constants: " << (upload_mode == use_root_constants) << std::endl;
    std::cout << "use_uniform_buffer: " << (upload_mode == use_uniform_buffer) << std::endl;
    std::cout << "use_dbo: " << (upload_mode == use_dbo) << std::endl;
    std::cout << "batch_mode: " << use_batch_mode << std::endl;
    std::cout << "dispatch_block_num: " << dispatch_block_num << std::endl;
    std::cout << "thread_block_size: " << thread_block_size << std::endl;
    std::cout << "dispatch_times_per_frame: " << dispatch_times_per_frame << std::endl;
    std::cout << "frame_num: " << frame_num << std::endl;
    std::cout << "constant_upload_bytes: " << constant_upload_bytes << std::endl;
    std::cout << "cbuffer_use_scalars: " << (cbuffer_content_type == scalar) << std::endl;
    std::cout << "cbuffer_use_array: " << (cbuffer_content_type == array) << std::endl;
    std::cout << "handle_oob_with_clamp: " << handle_oob_with_clamp << std::endl;

    const uint32_t constant_upload_elements_num =
        static_cast<uint32_t>(constant_upload_bytes / bytes_per_element);

    if (debug_mode) {
        Microsoft::WRL::ComPtr<ID3D12Debug> debug_controller;
        ThrowIfFailed(D3D12GetDebugInterface(IID_PPV_ARGS(&debug_controller)));
        debug_controller->EnableDebugLayer();
    }

    // Acquire adapter and create device
    ComPtr<IDXGIFactory4> factory;
    if (debug_mode) {
        ThrowIfFailed(CreateDXGIFactory2(DXGI_CREATE_FACTORY_DEBUG, IID_PPV_ARGS(&factory)));
    } else {
        ThrowIfFailed(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)));
    }

    ComPtr<IDXGIAdapter1> hardware_adapter;
    ComPtr<ID3D12Device> device;
    for (UINT adapter_index = 0;
         DXGI_ERROR_NOT_FOUND != factory->EnumAdapters1(adapter_index, &hardware_adapter);
         ++adapter_index) {
        DXGI_ADAPTER_DESC1 desc;
        hardware_adapter->GetDesc1(&desc);

        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
            continue;
        }

        if (D3D12CreateDevice(hardware_adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                              IID_PPV_ARGS(&device)) > 0) {
            break;
        }
    }

    // Create compute root signature.
    D3D12_FEATURE_DATA_ROOT_SIGNATURE feature_data = {};

    // This is the highest version the sample supports. If CheckFeatureSupport succeeds, the
    // HighestVersion returned will not be greater than this.
    feature_data.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_1;

    if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &feature_data,
                                           sizeof(feature_data)))) {
        feature_data.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_0;
    }

    // Create compute signature.
    CD3DX12_ROOT_PARAMETER1 compute_root_parameters[3];

    if (upload_mode == use_root_constants) {
        compute_root_parameters[0].InitAsConstants(constant_upload_elements_num, 0);
    }

    if (upload_mode == use_uniform_buffer || upload_mode == use_dbo) {
        compute_root_parameters[0].InitAsConstantBufferView(0, 0);
    }

    CD3DX12_DESCRIPTOR_RANGE1 srv_ranges[1];
    srv_ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0,
                       D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC);

    CD3DX12_DESCRIPTOR_RANGE1 uav_ranges[1];
    uav_ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0,
                       D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE);

    compute_root_parameters[1].InitAsDescriptorTable(1, srv_ranges);
    compute_root_parameters[2].InitAsDescriptorTable(1, uav_ranges);

    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC compute_root_signature_desc;
    compute_root_signature_desc.Init_1_1(_countof(compute_root_parameters),
                                         compute_root_parameters);

    ComPtr<ID3DBlob> signature;
    ComPtr<ID3DBlob> error;
    ThrowIfFailed(D3DX12SerializeVersionedRootSignature(
        &compute_root_signature_desc, feature_data.HighestVersion, &signature, &error));

    ComPtr<ID3D12RootSignature> compute_root_signature;

    ThrowIfFailed(device->CreateRootSignature(0, signature->GetBufferPointer(),
                                              signature->GetBufferSize(),
                                              IID_PPV_ARGS(&compute_root_signature)));

    // Construct shader with string, the shader source code looks like (if cbuffer_use_scalar):
    // #define threadBlockSize NUM
    // cbuffer ConstantMasks : register(b0)
    // {
    //   float4 mask0;
    //   float4 mask1;
    //   float4 mask2;
    //   float4 mask3;
    //   ...
    //  };
    // Buffer<float4> input         : register(t0);    // SRV: input
    // RWBuffer<float4> output    : register(u0);    // UAV: output

    // [numthreads(threadBlockSize, 1, 1)]
    // void CSMain(uint3 groupId : SV_GroupID, uint groupIndex : SV_GroupIndex)
    // {
    //
    //   uint index = (groupId.x * threadBlockSize) + groupIndex;
    //
    //   output[index] = input[index] + mask0 + mask1 + mask2 + mask3 + ...
    // }
    // or (if cbuffer_use_array):
    // #define threadBlockSize NUM
    // cbuffer ConstantMasks : register(b0)
    // {
    //   float4 mask[n];
    //  };
    // Buffer<float4> input         : register(t0);    // SRV: input
    // RWBuffer<float4> output    : register(u0);    // UAV: output

    // [numthreads(threadBlockSize, 1, 1)]
    // void CSMain(uint3 groupId : SV_GroupID, uint groupIndex : SV_GroupIndex)
    // {
    //
    //   uint index = (groupId.x * threadBlockSize) + groupIndex;
    //
    //   // if not handle_oob_with_clamp
    //   output[index] = input[index] + mask[0] + mask[1] + ...
    //   // if handle_oob_with_clamp
    //   output[index] = input[index] + mask[clamp(0, 0, n)] + mask[clamp(1, 0, n)]
    // }

    std::string compute_shader_source_code =
        "#define threadBlockSize " + std::to_string(thread_block_size) + std::string("\n");
    compute_shader_source_code += std::string("cbuffer ConstantMasks : register(b0) {\n");

    // The mask is float4 vector type in shader.
    uint32_t mask_num = constant_upload_elements_num / 4;

    if (cbuffer_content_type == scalar) {
        for (uint32_t i = 0; i < mask_num; ++i) {
            compute_shader_source_code +=
                std::string("    float4 mask") + std::to_string(i) + std::string(";\n");
        }
    }

    if (cbuffer_content_type == array) {
        compute_shader_source_code +=
            std::string("    float4 mask[") + std::to_string(mask_num) + std::string("];\n");
    }
    compute_shader_source_code += std::string("};\n");

    std::string shader_body = R"(

Buffer<float4> input         : register(t0);    // SRV: input 
RWBuffer<float4> output    : register(u0);    // UAV: output

[numthreads(threadBlockSize, 1, 1)]
void CSMain(uint3 groupId : SV_GroupID, uint groupIndex : SV_GroupIndex)
{
    uint index = (groupId.x * threadBlockSize) + groupIndex;

)";
    compute_shader_source_code += shader_body;
    compute_shader_source_code += std::string("    output[index] = input[index]");

    if (cbuffer_content_type == scalar) {
        for (uint32_t i = 0; i < mask_num; ++i) {
            compute_shader_source_code += std::string(" + mask") + std::to_string(i);
        }
    }

    if (cbuffer_content_type == array) {
        for (uint32_t i = 0; i < mask_num; ++i) {
            if (handle_oob_with_clamp) {
                compute_shader_source_code += std::string(" + mask[clamp(") + std::to_string(i) +
                                              std::string(", 0, ") + std::to_string(mask_num) +
                                              std::string(")]");
            } else {
                compute_shader_source_code +=
                    std::string(" + mask[") + std::to_string(i) + std::string("]");
            }
        }
    }

    compute_shader_source_code += std::string(";\n}");

    // Create compute shader object
    ComPtr<ID3DBlob> compute_shader;
    ComPtr<ID3DBlob> error_signature;

    UINT compileFlags;
    if (debug_mode) {
        compileFlags = D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
    } else {
        compileFlags = 0;
    }

    ThrowIfFailed(D3DCompile(
        compute_shader_source_code.c_str(), compute_shader_source_code.length(), nullptr, nullptr,
        nullptr, "CSMain", "cs_5_0", compileFlags, 0, &compute_shader, &error_signature));

    // Describe and create the compute pipeline state object (PSO).
    uint32_t compute_pipline_state_object_num = use_batch_mode ? dispatch_times_per_frame : 1;
    std::vector<ComPtr<ID3D12PipelineState>> compute_state(compute_pipline_state_object_num);

    for (uint32_t i = 0; i < compute_pipline_state_object_num; ++i) {
        D3D12_COMPUTE_PIPELINE_STATE_DESC compute_pso_desc = {};
        compute_pso_desc.pRootSignature = compute_root_signature.Get();
        compute_pso_desc.CS = CD3DX12_SHADER_BYTECODE(compute_shader.Get());

        ThrowIfFailed(
            device->CreateComputePipelineState(&compute_pso_desc, IID_PPV_ARGS(&compute_state[i])));
    }

    // Create upload buffer and input buffers. Upload contents to the input buffer.
    size_t content_size = bytes_per_element * element_per_thread * dispatch_thread_num;
    size_t align = content_size / buffer_alignment_bytes;
    if (content_size % buffer_alignment_bytes > 0) {
        align += 1;
    }
    const size_t input_output_buffer_size = align * buffer_alignment_bytes;
    const D3D12_HEAP_PROPERTIES upload_heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
    const D3D12_RESOURCE_DESC srv_upload_resource_desc =
        CD3DX12_RESOURCE_DESC::Buffer(input_output_buffer_size);

    ComPtr<ID3D12Resource> srv_upload_buffer;
    ThrowIfFailed(device->CreateCommittedResource(
        &upload_heap, D3D12_HEAP_FLAG_NONE, &srv_upload_resource_desc,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&srv_upload_buffer)));

    std::default_random_engine engine;
    engine.seed(static_cast<uint32_t>(std::chrono::system_clock::now().time_since_epoch().count()));
    std::uniform_real_distribution<float> distribution(0.0, 1.0);

    if (!debug_mode) {
        float* p_upload_data_begin;
        CD3DX12_RANGE read_range(0, 0);  // We do not intend to read from this resource on the CPU.
        ThrowIfFailed(
            srv_upload_buffer->Map(0, &read_range, reinterpret_cast<void**>(&p_upload_data_begin)));

        for (uint32_t i = 0; i < dispatch_thread_num; ++i) {
            p_upload_data_begin[i] = distribution(engine);
        }

        srv_upload_buffer->Unmap(0, nullptr);
    }

    const D3D12_HEAP_PROPERTIES srv_Heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    const D3D12_RESOURCE_DESC srv_resource_desc =
        CD3DX12_RESOURCE_DESC::Buffer(input_output_buffer_size);
    ComPtr<ID3D12Resource> srv_buffer;
    ThrowIfFailed(device->CreateCommittedResource(
        &srv_Heap, D3D12_HEAP_FLAG_NONE, &srv_resource_desc, D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr, IID_PPV_ARGS(&srv_buffer)));

    ComPtr<ID3D12CommandQueue> command_queue;
    D3D12_COMMAND_QUEUE_DESC queue_desc = {};
    queue_desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ThrowIfFailed(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&command_queue)));

    D3D12_RESOURCE_BARRIER copy_dest_to_common_barrier;
    copy_dest_to_common_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    copy_dest_to_common_barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    copy_dest_to_common_barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    copy_dest_to_common_barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
    copy_dest_to_common_barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    copy_dest_to_common_barrier.Transition.pResource = srv_buffer.Get();

    ComPtr<ID3D12CommandAllocator> upload_command_allocator;
    ThrowIfFailed(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                 IID_PPV_ARGS(&upload_command_allocator)));

    ComPtr<ID3D12GraphicsCommandList> upload_command_list;
    ThrowIfFailed(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                            upload_command_allocator.Get(), nullptr,
                                            IID_PPV_ARGS(&upload_command_list)));

    HANDLE fence_event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    ComPtr<ID3D12Fence> fence;
    UINT64 signal_fence_value = 1;
    ThrowIfFailed(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));

    upload_command_list->CopyBufferRegion(srv_buffer.Get(), 0, srv_upload_buffer.Get(), 0,
                                          input_output_buffer_size);
    upload_command_list->ResourceBarrier(1, &copy_dest_to_common_barrier);

    ThrowIfFailed(upload_command_list->Close());

    ID3D12CommandList* upload_command_lists[] = {upload_command_list.Get()};
    command_queue->ExecuteCommandLists(1, upload_command_lists);

    ThrowIfFailed(command_queue->Signal(fence.Get(), signal_fence_value));

    if (fence->GetCompletedValue() < signal_fence_value) {
        ThrowIfFailed(fence->SetEventOnCompletion(signal_fence_value, fence_event));
        WaitForSingleObject(fence_event, INFINITE);
    }

    // Create srv on the input buffer
    ComPtr<ID3D12DescriptorHeap> srv_uav_heap;
    uint32_t num_descriptors = dispatch_times_per_frame + 1;

    D3D12_DESCRIPTOR_HEAP_DESC srv_uav_heap_desc = {};
    srv_uav_heap_desc.NumDescriptors = num_descriptors;
    srv_uav_heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    srv_uav_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    ThrowIfFailed(device->CreateDescriptorHeap(&srv_uav_heap_desc, IID_PPV_ARGS(&srv_uav_heap)));
    uint32_t srv_uav_descriptor_size =
        device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    // Create shader resource views (SRV) of the constant buffers for the
    // compute shader to read from.
    D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
    srv_desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv_desc.Buffer.NumElements = dispatch_thread_num;
    srv_desc.Buffer.StructureByteStride = 0;
    srv_desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;

    CD3DX12_CPU_DESCRIPTOR_HANDLE srv_handle(srv_uav_heap->GetCPUDescriptorHandleForHeapStart(), 0,
                                             srv_uav_descriptor_size);
    device->CreateShaderResourceView(srv_buffer.Get(), &srv_desc, srv_handle);

    CD3DX12_CPU_DESCRIPTOR_HANDLE uav_handle(srv_uav_heap->GetCPUDescriptorHandleForHeapStart(), 1,
                                             srv_uav_descriptor_size);

    // Create output buffers and uavs.
    const D3D12_HEAP_PROPERTIES uav_resource_heap =
        CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    const D3D12_RESOURCE_DESC uav_buffer_desc = CD3DX12_RESOURCE_DESC::Buffer(
        input_output_buffer_size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    uint32_t uav_buffer_num = dispatch_times_per_frame;
    std::vector<ComPtr<ID3D12Resource>> uav_buffer(uav_buffer_num);

    for (uint32_t i = 0; i < uav_buffer_num; ++i) {
        ThrowIfFailed(device->CreateCommittedResource(
            &uav_resource_heap, D3D12_HEAP_FLAG_NONE, &uav_buffer_desc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&uav_buffer[i])));

        D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc = {};
        uav_desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        uav_desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uav_desc.Buffer.FirstElement = 0;
        uav_desc.Buffer.NumElements = dispatch_thread_num;
        uav_desc.Buffer.StructureByteStride = 0;
        uav_desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;

        device->CreateUnorderedAccessView(uav_buffer[i].Get(), nullptr, &uav_desc, uav_handle);
        uav_handle.Offset(1, srv_uav_descriptor_size);
    }

    // Prepare uploading constants.
    std::vector<std::vector<float>> upload_constants(dispatch_times_per_frame);

    for (uint32_t i = 0; i < dispatch_times_per_frame; ++i) {
        upload_constants[i].resize(constant_upload_elements_num);
        for (uint32_t j = 0; j < constant_upload_elements_num; ++j) {
            upload_constants[i][j] = distribution(engine);
        }
    }

    // Prepare constant buffer and upload buffers. To support using uniform buffer to upload
    // constants.
    D3D12_RESOURCE_BARRIER copy_dest_to_constant_barrier;
    copy_dest_to_constant_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    copy_dest_to_constant_barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    copy_dest_to_constant_barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    copy_dest_to_constant_barrier.Transition.StateAfter =
        D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
    copy_dest_to_constant_barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

    const D3D12_HEAP_PROPERTIES cbv_resource_heap =
        CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);

    align = constant_upload_bytes / buffer_alignment_bytes;
    if (constant_upload_bytes % buffer_alignment_bytes > 0) {
        align += 1;
    }

    size_t constant_buffer_size = align * buffer_alignment_bytes;
    uint32_t constant_buffer_num = 1;

    D3D12_RESOURCE_DESC cbv_upload_resource_desc;
    D3D12_RESOURCE_DESC cbv_resource_desc;

    std::vector<ComPtr<ID3D12Resource>> cbv_upload_buffer(constant_buffer_num);
    std::vector<ComPtr<ID3D12Resource>> constant_buffer(constant_buffer_num);

    if (upload_mode == use_uniform_buffer) {
        if (use_batch_mode) {
            constant_buffer_num = dispatch_times_per_frame;
            cbv_upload_buffer.resize(constant_buffer_num);
            constant_buffer.resize(constant_buffer_num);
        }

        cbv_upload_resource_desc = CD3DX12_RESOURCE_DESC::Buffer(constant_buffer_size);

        for (uint32_t i = 0; i < constant_buffer_num; ++i) {
            ThrowIfFailed(device->CreateCommittedResource(
                &upload_heap, D3D12_HEAP_FLAG_NONE, &cbv_upload_resource_desc,
                D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&cbv_upload_buffer[i])));
        }
        cbv_resource_desc = CD3DX12_RESOURCE_DESC::Buffer(constant_buffer_size);

        for (uint32_t i = 0; i < constant_buffer_num; ++i) {
            ThrowIfFailed(device->CreateCommittedResource(
                &cbv_resource_heap, D3D12_HEAP_FLAG_NONE, &cbv_resource_desc,
                D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&constant_buffer[i])));
        }
    }

    if (use_dbo) {
        uint32_t dbo_per_instance_contents_num = align * buffer_alignment_bytes / bytes_per_element;
        uint32_t dbo_contents_num = dbo_per_instance_contents_num * dispatch_times_per_frame;
        size_t dbo_buffer_size = align * buffer_alignment_bytes * dispatch_times_per_frame;

        std::vector<float> dbo_contents(dbo_contents_num);
        uint32_t dbo_contents_index = 0;
        for (uint32_t i = 0; i < dispatch_times_per_frame; ++i) {
            dbo_contents_index = i * dbo_per_instance_contents_num;
            for (uint32_t j = 0; j < constant_upload_elements_num; ++j) {
                dbo_contents[dbo_contents_index + j] = upload_constants[i][j];
            }
        }

        cbv_upload_resource_desc = CD3DX12_RESOURCE_DESC::Buffer(dbo_buffer_size);

        ThrowIfFailed(device->CreateCommittedResource(
            &upload_heap, D3D12_HEAP_FLAG_NONE, &cbv_upload_resource_desc,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&cbv_upload_buffer[0])));

        cbv_resource_desc = CD3DX12_RESOURCE_DESC::Buffer(dbo_buffer_size);

        ThrowIfFailed(device->CreateCommittedResource(
            &cbv_resource_heap, D3D12_HEAP_FLAG_NONE, &cbv_resource_desc,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&constant_buffer[0])));

        float* p_upload_data_begin;
        CD3DX12_RANGE read_range(0, 0);  // We do not intend to read from this resource on the CPU.
        ThrowIfFailed(cbv_upload_buffer[0]->Map(0, &read_range,
                                                reinterpret_cast<void**>(&p_upload_data_begin)));
        memcpy(p_upload_data_begin, dbo_contents.data(), dbo_buffer_size);
        cbv_upload_buffer[0]->Unmap(0, nullptr);

        ThrowIfFailed(upload_command_allocator->Reset());
        ThrowIfFailed(upload_command_list->Reset(upload_command_allocator.Get(), nullptr));
        upload_command_list->CopyBufferRegion(constant_buffer[0].Get(), 0,
                                              cbv_upload_buffer[0].Get(), 0, dbo_buffer_size);
        copy_dest_to_constant_barrier.Transition.pResource = constant_buffer[0].Get();
        upload_command_list->ResourceBarrier(1, &copy_dest_to_constant_barrier);
        ThrowIfFailed(upload_command_list->Close());
        ID3D12CommandList* upload_dbo_command_lists[] = {upload_command_list.Get()};
        command_queue->ExecuteCommandLists(1, upload_dbo_command_lists);

        ++signal_fence_value;
        ThrowIfFailed(command_queue->Signal(fence.Get(), signal_fence_value));

        if (fence->GetCompletedValue() < signal_fence_value) {
            ThrowIfFailed(fence->SetEventOnCompletion(signal_fence_value, fence_event));
            WaitForSingleObject(fence_event, INFINITE);
        }
    }

    uint32_t submit_num = use_batch_mode ? 1 : dispatch_times_per_frame;
    uint32_t dispatch_num_per_submit = use_batch_mode ? dispatch_times_per_frame : 1;

    // Create compute command allocator and compute command lists. For no batching mode,
    // we record single dispatch in one command list and submit it immediately. Reuse the
    // command list for each submit introduce fence waiting, which impact the peformance
    // measurement. So creating allocator and command lists based on counts of submit. In
    // this way, we avoid reusing the command list and no need to wait for fence before next
    // submission.
    std::vector<ComPtr<ID3D12CommandAllocator>> compute_command_allocators(submit_num);

    for (uint32_t i = 0; i < submit_num; ++i) {
        ThrowIfFailed(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                     IID_PPV_ARGS(&compute_command_allocators[i])));
    }

    std::vector<ComPtr<ID3D12GraphicsCommandList>> compute_command_lists(submit_num);

    for (uint32_t i = 0; i < submit_num; ++i) {
        ThrowIfFailed(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                compute_command_allocators[i].Get(), nullptr,
                                                IID_PPV_ARGS(&compute_command_lists[i])));
        ThrowIfFailed(compute_command_lists[i]->Close());
    }

    // Record start time
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    double total_time_nanoseconds = 0.0;

    for (uint32_t frame_count = 0; frame_count < frame_num + 1; ++frame_count) {
        start = std::chrono::high_resolution_clock::now();
        uint32_t submit_dispatch_num = 0;
        for (uint32_t submit_count = 0; submit_count < submit_num;
             ++submit_count, submit_dispatch_num += dispatch_num_per_submit) {
            // Command list allocators can only be reset when the associated
            // command lists have finished execution on the GPU; apps should use
            // fences to determine GPU execution progress.
            ThrowIfFailed(compute_command_allocators[submit_count]->Reset());

            // However, when ExecuteCommandList() is called on a particular command
            // list, that command list can then be reset at any time and must be before
            // re-recording.
            ThrowIfFailed(compute_command_lists[submit_count]->Reset(
                compute_command_allocators[submit_count].Get(), nullptr));

            ID3D12DescriptorHeap* pp_heaps[] = {srv_uav_heap.Get()};
            compute_command_lists[submit_count]->SetDescriptorHeaps(_countof(pp_heaps), pp_heaps);
            compute_command_lists[submit_count]->SetComputeRootSignature(
                compute_root_signature.Get());

            ++signal_fence_value;

            for (uint32_t dispatch_count_in_submit = 0;
                 dispatch_count_in_submit < dispatch_num_per_submit; ++dispatch_count_in_submit) {
                uint32_t dispatch_count = submit_dispatch_num + dispatch_count_in_submit;
                compute_command_lists[submit_count]->SetPipelineState(
                    compute_state[dispatch_count_in_submit].Get());

                if (upload_mode == use_root_constants) {
                    compute_command_lists[submit_count]->SetComputeRoot32BitConstants(
                        0, constant_upload_elements_num,
                        reinterpret_cast<void*>(upload_constants[dispatch_count].data()), 0);
                }

                if (upload_mode == use_uniform_buffer) {
                    float* p_upload_data_begin;
                    CD3DX12_RANGE read_range(
                        0, 0);  // We do not intend to read from this resource on the CPU.
                    ThrowIfFailed(cbv_upload_buffer[dispatch_count_in_submit]->Map(
                        0, &read_range, reinterpret_cast<void**>(&p_upload_data_begin)));
                    memcpy(p_upload_data_begin, upload_constants[dispatch_count].data(),
                           sizeof(float) * constant_upload_elements_num);
                    cbv_upload_buffer[dispatch_count_in_submit]->Unmap(0, nullptr);

                    compute_command_lists[submit_count]->CopyBufferRegion(
                        constant_buffer[dispatch_count_in_submit].Get(), 0,
                        cbv_upload_buffer[dispatch_count_in_submit].Get(), 0, constant_buffer_size);
                    copy_dest_to_constant_barrier.Transition.pResource =
                        constant_buffer[dispatch_count_in_submit].Get();
                    compute_command_lists[submit_count]->ResourceBarrier(
                        1, &copy_dest_to_constant_barrier);

                    compute_command_lists[submit_count]->SetComputeRootConstantBufferView(
                        0, constant_buffer[dispatch_count_in_submit]->GetGPUVirtualAddress());
                }

                if (upload_mode == use_dbo) {
                    D3D12_GPU_VIRTUAL_ADDRESS buffer_location =
                        constant_buffer[0]->GetGPUVirtualAddress() +
                        dispatch_count * constant_buffer_size;
                    compute_command_lists[submit_count]->SetComputeRootConstantBufferView(
                        0, buffer_location);
                }

                compute_command_lists[submit_count]->SetComputeRootDescriptorTable(
                    1, srv_uav_heap->GetGPUDescriptorHandleForHeapStart());
                compute_command_lists[submit_count]->SetComputeRootDescriptorTable(
                    2, CD3DX12_GPU_DESCRIPTOR_HANDLE(
                           srv_uav_heap->GetGPUDescriptorHandleForHeapStart(), dispatch_count + 1,
                           srv_uav_descriptor_size));

                compute_command_lists[submit_count]->Dispatch(dispatch_block_num, 1, 1);
            }

            ThrowIfFailed(compute_command_lists[submit_count]->Close());

            ID3D12CommandList* submit_command_lists[] = {compute_command_lists[submit_count].Get()};
            command_queue->ExecuteCommandLists(1, submit_command_lists);

            ThrowIfFailed(command_queue->Signal(fence.Get(), signal_fence_value));
        }

        if (fence->GetCompletedValue() < signal_fence_value) {
            ThrowIfFailed(fence->SetEventOnCompletion(signal_fence_value, fence_event));
            WaitForSingleObject(fence_event, INFINITE);
        }
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::nano> elapsed_nanoseconds = end - start;
        if (frame_count > 0) {
            total_time_nanoseconds += elapsed_nanoseconds.count();
        }
        if (frame_count == 0) {
            std::cout << "(Frame 0 is not take into average time calculation) ";
        }

        std::cout << "Frame " << std::to_string(frame_count)
                  << " time (ns) : " << elapsed_nanoseconds.count() << " ns;" << std::endl;
    }

    double average_nanoseconds_per_frame = total_time_nanoseconds / frame_num;

    if (debug_mode) {
        std::cout << std::endl;
        std::cout << "NOTE:Debug mode on, time is not accurate." << std::endl;
    }

    std::cout << "Average frame time (ns) of " << std::to_string(frame_num)
              << " frames : " << average_nanoseconds_per_frame << " ns; " << std::endl;
    std::cout << std::endl;

    if (debug_mode) {
        uint32_t download_uav_buffer_index = 0;
        uint32_t step = (uav_buffer_num - 1) / 2;
        step = step > 0 ? step : 1;
        while (download_uav_buffer_index < uav_buffer_num) {
            DownloadResultBufferContents(
                device, command_queue, uav_buffer[download_uav_buffer_index],
                input_output_buffer_size, download_uav_buffer_index, dispatch_thread_num);
            download_uav_buffer_index += step;
            std::cout << std::endl;
        }
    }

    return 1;
}
