huggingface-cli download --resume-download Qwen/Qwen2-VL-7B-Instruct --local-dir /mnt/petrelfs/gaopeng/lyt/Qwen2-VL-Finetune/ckt/  --local-dir-use-symlinks False 




srun -p Gvlab-S1-32  --gres=gpu:0 --cpus-per-task=8 --pty bash
#!/bin/sh
srun  bash /mnt/petrelfs/luyiting/Lumina-mGPT/lumina_mgpt/exps/7B_b2.sh
#sbatch -p Gvlab-S1-32 -n2 --ntasks-per-node=1 --quotatype=spot --cpus-per-task=8 --job-name=7B_2 --gres=gpu:2 /mnt/petrelfs/luyiting/Lumina-mGPT/lumina_mgpt/exps/7B_b2.sh 
#srun -p Gvlab-S1-32 -n2 --ntasks-per-node=1 --quotatype=spot --cpus-per-task=8 --job-name=7B_2 --gres=gpu:2 bash /mnt/petrelfs/luyiting/Lumina-mGPT/lumina_mgpt/exps/7B_b2.sh
#sinfo -t idle   查看空闲节点
#scancel 15802854    Kill任务
#squeue -u luyiting   查看提交任务状态
#squeue -u gaopeng   查看提交任务状态
#squeue -p Gvlab-S1-32
#PD：PENDING，作业已提交但还未开始运行。
#R：RUNNING，作业正在运行。
#CG：COMPLETING，作业即将完成。
#CD：COMPLETED，作业已完成。
#F：FAILED，作业失败。
#TO：TIMEOUT，作业超时。
#SH-IDC1-10-140-0-32:192140
#squeue -j 15918795 手动登陆节点查看运行状态
#手动登录该节点： ssh node123
srun -p Gvlab-S1-32 --quotatype=spot  --gres=gpu:0 --cpus-per-task=8 --pty bash
#srun -p Gvlab-S1-32 --quotatype=spot --gres=gpu:1 --cpus-per-task=8 --pty bash
#srun -p Gvlab-S1-32 --quotatype=spot --gres=gpu:2 --cpus-per-task=8 --pty bash
#srun -p Gvlab-S1-32 -x SH-IDC1-10-140-1-131,H-IDC1-10-140-1-132,H-IDC1-10-140-1-133--quotatype=spot --gres=gpu:0 --cpus-per-task=8 --pty bash
srun -p Gvlab-S1-32  -x SH-IDC1-10-140-1-131,SH-IDC1-10-140-1-54,SH-IDC1-10-140-0-134,SH-IDC1-10-140-0-205,SH-IDC1-10-140-0-206 --quotatype=spot  --gres=gpu:0 --cpus-per-task=8 --pty bash
#srun -p Gvlab-S1-32 -x SH-IDC1-10-140-0-131,SH-IDC1-10-140-0-132,SH-IDC1-10-140-0-133 --quotatype=spot --gres=gpu:0 --cpus-per-task=8 --pty bash
#cinfo  -p Gvlab-S1-32
#cinfo  -p Gveval-S1 
#cinfo  -p lumina
cinfo  -p Omnilab
#sbatch -p Gvlab-S1-32 -n1 --ntasks-per-node=1 --quotatype spot --job-name=7B_8 --cpus-per-task=12 --gres=gpu:8  /mnt/petrelfs/luyiting/Lumina-mGPT/scripts/run8_full.sh  
#sbatch -p Gvlab-S1-32 -x SH-IDC1-10-140-0-243 -n4 --ntasks-per-node=1 --quotatype=spot --cpus-per-task=8 --job-name=7B_8 --gres=gpu:2 /mnt/petrelfs/luyiting/Lumina-mGPT/scripts/run8_full.sh   
#sbatch -p Gvlab-S1-32 -n4 --ntasks-per-node=2 --quotatype=spot --cpus-per-task=8 --job-name=7B_8 --gres=gpu:8  /mnt/petrelfs/luyiting/Lumina-mGPT/scripts/run8_full.sh  
#sbatch -p Gvlab-S1-32 -n4 --ntasks-per-node=2 --quotatype=spot --cpus-per-task=8 --job-name=7B_8 --gres=gpu:2  /mnt/petrelfs/luyiting/Lumina-mGPT/scripts/run8_full.sh  
#srun -p Gvlab-S1-32 -n4 --ntasks-per-node=1 --quotatype=spot --cpus-per-task=8 --job-name=7B_8 --gres=gpu:2  bash /mnt/petrelfs/luyiting/Lumina-mGPT/lumina_mgpt/exps/7B_full.sh
#aws s3 ls endpoint-url=http://p-ceph-norm-outside.pjlab.org.cn s3://ldy/xllmx/data/images/ocrvga/images/1570761 列出下面的文件

#-n4表
rclone copy --progress --transfers 100 --checkers 100  lc2://luyiting/Images/llava-critic-113k/ My-aliyun-oss1:pjlab-lingjun-lumina-gpt/Datasets/luyiting/llava-critic-113k/
##清除节点显存


swatch -n SH-IDC1-10-140-0-150 memory_release
swatch -n  SH-IDC1-10-140-0-154  memory_release
swatch -n  SH-IDC1-10-140-0-208 memory_release
swatch -n   SH-IDC1-10-140-1-67 memory_release
swatch -n    SH-IDC1-10-140-1-46 memory_release
swatch -n  SH-IDC1-10-140-0-140 memory_release
swatch -n 节点 memory_release 
swatch -n SH-IDC1-10-140-0-146 memory_release 
swatch -n SH-IDC1-10-140-0-146  check_fast
swatch -n SH-IDC1-10-140-0-146 check_fast 

swatch -n SH-IDC1-10-140-0-146 clean_process 
swatch -n SH-IDC1-10-140-0-146 nv 
swatch -n SH-IDC1-10-140-0-146 list_program 
swatch -n  SH-IDC1-10-140-1-83 nv
swatch -n   SH-IDC1-10-140-1-119  nv
swatch -n  SH-IDC1-10-140-0-160  list_program
##查看用户名的python进程
ps -u gaopeng -f | grep python

swatch -n   SH-IDC1-10-140-0-131 nv


swatch -n  SH-IDC1-10-140-0-203 memory_release

swatch -n  SH-IDC1-10-140-1-143     nv


 swatch -n  SH-IDC1-10-140-1-123 nv


 swatch -n SH-IDC1-10-140-0-236 clean_process 

 swatch -n SH-IDC1-10-140-0-232 nv
  swatch -n  SH-IDC1-10-140-1-127 nv

  swatch -n  SH-IDC1-10-140-0-255 nv

   swatch -n  SH-IDC1-10-140-0-181 nv

      swatch -n SH-IDC1-10-140-1-1 clean_process
deepspeed0.14.0


srun -N 0 -w SH-IDC1-10-140-0-145 -p Omnilab  bash



swatch -n SH-IDC1-10-140-0-237 nv

swatch -n SH-IDC1-10-140-0-208 nv


swatch -n SH-IDC1-10-140-0-181 nv
swatch -n SH-IDC1-10-140-0-181  clean_process



# rclone ls lc2:s2//audio_data/raw_data/Emilia/EN
#


rclone copy --progress --transfers 100 --checkers 100 My-aliyun-oss1:pjlab-lingjun-lumina-gpt/Datasets/luyiting/MAmmoTH-VL-Instruct-12M-new/ lc2://luyiting/Images/MAmmoTH-VL-Instruct-12M-new/



export LD_LIBRARY_PATH=//mnt/petrelfs/share/test-cuda/cuda-12.1/lib64:$LD_LIBRARY_PATH



export CUDA_HOME=/mnt/petrelfs/gaopeng/ldy/cuda-12.1
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}


export CUDA_HOME=/mnt/petrelfs/share/cuda-12.1
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

export LD_LIBRARY_PATH=/mnt/petrelfs/gaopeng/.triton/cache/ef3905381b0298986d1632a4ec1db8f5/:$LD_LIBRARY_PATH

export KERNEL_PATH='/mnt/petrelfs/gaopeng/.triton/autotune/Fp16Matmul_4d_kernel.pickle'

export KERNEL_PATH='/mnt/petrelfs/gaopeng/.triton/autotune/Fp16Matmul_2d_kernel.pickle'


export CUDA_HOME=/mnt/hwfile/alpha_vl/linziyi/cuda-12.1
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

ls /mnt/hwfile/alpha_vl/linziyi/cuda-12.1/lib64/ | grep cudnn

ls /mnt/petrelfs/share/cuda-12.1/lib64/ | grep cudnn

ls /mnt/petrelfs/share/test-cuda/cuda-12.1/ | grep cudnn

ls /mnt/petrelfs/gaopeng/ldy/cuda-12.1/lib64/ | grep cudnn


export CUDA_HOME=/mnt/petrelfs/share/test-cuda/cuda-12.1/
#export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH#
export PATH=/mnt/petrelfs/share/test-cuda/cuda-12.1/bin:$PATH


cat /mnt/petrelfs/share/test-cuda/cuda-12.1/include/cudnn_version.h | grep CUDNN_MAJOR -A 2

cat /mnt/petrelfs/gaopeng/ldy/cuda-12.1/include/cudnn_version.h | grep CUDNN_MAJOR -A 2


echo $LD_LIBRARY_PATH
/mnt/hwfile/alpha_vl/linziyi/cuda-12.1/lib64:/mnt/petrelfs/share/ffmpeg-4.2.1/lib:/usr/lib64/:/mnt/petrelfs/gaopeng/.triton/cache/ef3905381b0298986d1632a4ec1db8f5/:/mnt/hwfile/alpha_vl/linziyi/cuda-12.1/lib64:/mnt/petrelfs/share/gcc/gmp-4.3.2/lib/:/mnt/petrelfs/share/gcc/mpfr-2.4.2/lib:/mnt/petrelfs/share/gcc/mpc-0.8.1/lib:/mnt/petrelfs/share/gcc/gcc-9.4.0/lib64::/mnt/petrelfs/share/ffmpeg-4.2.1/lib:/usr/lib64/:/mnt/petrelfs/gaopeng/.triton/cache/ef3905381b0298986d1632a4ec1db8f5/:/mnt/hwfile/alpha_vl/linziyi/cuda-12.1/lib64:/mnt/petrelfs/share/gcc/gmp-4.3.2/lib/:/mnt/petrelfs/share/gcc/mpfr-2.4.2/lib:/mnt/petrelfs/share/gcc/mpc-0.8.1/lib:/mnt/petrelfs/share/gcc/gcc-9.4.0/lib64::/mnt/petrelfs/share/ffmpeg-4.2.1/lib:/usr/lib64/:/mnt/petrelfs/gaopeng/.triton/cache/ef3905381b0298986d1632a4ec1db8f5/:/mnt/hwfile/alpha_vl/linziyi/cuda-12.1/lib64:/mnt/petrelfs/share/gcc/gmp-4.3.2/lib/:/mnt/petrelfs/share/gcc/mpfr-2.4.2/lib:/mnt/petrelfs/share/gcc/mpc-0.8.1/lib:/mnt/petrelfs/share/gcc/gcc-9.4.0/lib64::/mnt/petrelfs/gaopeng/.triton/cache/ef3905381b0298986d1632a4ec1db8f5/:/mnt/petrelfs/share/cuda-12.1/lib64:/mnt/petrelfs/share/ffmpeg-4.2.1/lib:/usr/lib64/:/mnt/petrelfs/gaopeng/.triton/cache/ef3905381b0298986d1632a4ec1db8f5/:/mnt/petrelfs/gaopeng/ldy/cuda-12.1/lib64:/mnt/petrelfs/share/gcc/gmp-4.3.2/lib/:/mnt/petrelfs/share/gcc/mpfr-2.4.2/lib:/mnt/petrelfs/share/gcc/mpc-0.8.1/lib:/mnt/petrelfs/share/gcc/gcc-9.4.0/lib64:

export LD_LIBRARY_PATH=/mnt/petrelfs/gaopeng/ldy/cuda-12.1/lib64:/mnt/petrelfs/share/ffmpeg-4.2.1/lib:/usr/lib64:$LD_LIBRARY_PATH
