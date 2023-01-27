// Copyright (c) 2022, Tri Dao.

// Splitting the different head dimensions to different files to speed up compilation.

#include "fmha_fwd_launch_template.h"
#include "cuda_fp16.h"

void run_fmha_fwd_hdim40(Launch_params<FMHA_fprop_params> &launch_params) {
    FP16_SWITCH(launch_params.params.is_bf16, ({
        if (launch_params.params.seqlen_k == 128) {
            using Kernel_traits = FMHA_kernel_traits<128, 40, 16, 1, 4, 0x08u, elem_type>;
            run_fmha_fwd_loop<Kernel_traits>(launch_params);
        } else if (launch_params.params.seqlen_k >= 256) {
            using Kernel_traits = FMHA_kernel_traits<256, 40, 16, 1, 4, 0x08u, elem_type>;
            run_fmha_fwd_loop<Kernel_traits>(launch_params);
        }
    }));
    // cudaDeviceSynchronize();
    // printf("@@@ print output in fmha_fwd\n");
    // print_out<<<1,1>>>(reinterpret_cast<half*>(launch_params.params.o_ptr),
    //                   launch_params.params.b*launch_params.params.h*launch_params.params.seqlen_q*launch_params.params.d,
    //                   launch_params.params.d);
    // printf("@@@ print output in fmha_fwd\n");
    // cudaDeviceSynchronize();

}
