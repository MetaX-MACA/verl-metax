### docker start

docker run -d -it --net=host --uts=host --ipc=host --privileged=true --group-add video --shm-size 100gb --ulimit memlock=-1 --security-opt seccomp=unconfined --security-opt apparmor=unconfined --device=/dev/dri --device=/dev/mxcd --device=/dev/infiniband -v /pde_ai/datasets/:/external/datasets/ -v /pde_ai/models/:/external/models/ --name verl_train_multinode image_name bash

### multi node

多机需要先使用ray配置资源池，env.sh里面的网卡名称按需修改

```
在每个节点执行：
cd /workspace/verl-metax

环境变量需要在ray之前配置
source examples/env/megatron.sh

在master节点执行：
ray start --head --port=1009

在其它slave节点执行：
ray start --address='masterip:1009'

执行完后，可以查看资源池的状态
ray status


启动2节点训练，只需在master节点执行（或者其中一个节点）：
bash examples/grpo/deepseek_r1_qwen_32b_megatron_2nodes.sh


如果需要关闭ray,在每个节点执行
ray stop

```

如需启用 sleep mode，请在每个节点启动 Ray 前按照
[共享环境说明](../../examples/env/README.md#vllm-metax-0230-sleep-mode)配置。

### 版本对齐

版本不对齐会引起报错, 运行前应进行版本检查
已验证版本信息：Megatron-LM-0.18.0 + vllm-0.23.0 + transformers-5.8.1
