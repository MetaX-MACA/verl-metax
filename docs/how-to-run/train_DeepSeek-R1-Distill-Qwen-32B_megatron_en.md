
### docker start
docker run -d -it --net=host --uts=host --ipc=host --privileged=true --group-add video --shm-size 100gb --ulimit memlock=-1 --security-opt seccomp=unconfined --security-opt apparmor=unconfined --device=/dev/dri --device=/dev/mxcd --device=/dev/infiniband -v /pde_ai/datasets/:/external/datasets/ -v /pde_ai/models/:/external/models/ --name verl_train_multinode image_name bash

### Multi-Node Setup

Before starting multi-node training, you need to first configure a resource pool using Ray. You may need to modify the network interface name in env.sh as required.

```
On each node, execute the following:
cd /workspace/verl-metax

Environment variables must be set before starting Ray
source examples/env/megatron.sh

On the master node, execute:
ray start --head --port=1009

On other worker nodes, execute:
ray start --address='master_ip:1009'

After setup, you can check the status of the resource pool with:
ray status


To start a 2-node training job, execute on the master node (or any one of the nodes):
bash examples/grpo/deepseek_r1_qwen_32b_megatron_2nodes.sh


If you need to stop Ray, execute on each node:

ray stop

```

If sleep mode is required, configure it on every node before starting Ray;
see the [shared environment guide](../../examples/env/README.md#vllm-metax-0230-sleep-mode).

### Version Info

Packages' version should be aligned.
Aligned versions：Megatron-LM-0.18.0 + vllm-0.23.0 + transformers-5.8.1
