
from clearml import Task

# note: you need "clearml==0.17.6rc1"

#Task.add_requirements('requirements.txt')
Task.add_requirements("./diffusers")
Task.add_requirements("triton")
Task.add_requirements("ftfy")
Task.add_requirements("safetensors")
Task.add_requirements("accelerate")
Task.add_requirements("transformers")
Task.add_requirements("bitsandbytes", "0.35.0")
Task.add_requirements("natsort")
Task.add_requirements("torchvision")
Task.add_requirements("xformers", "0.0.17.dev447")
Task.add_requirements("matplotlib")
Task.add_requirements("torch", "1.13.1+cu116")




task = Task.init(
    project_name='Text Classification',
    task_name='dreambooth',
    tags="just a test",
    auto_connect_arg_parser=True,
)

task.execute_remotely(queue_name="<=12GB", clone=False, exit_process=True)
import diffusers