
from clearml import Task

# note: you need "clearml==0.17.6rc1"
Task.add_requirements("./diffusers")
Task.add_requirements('requirements.txt')
Task.add_requirements("safetensors")


task = Task.init(
    project_name='Text Classification',
    task_name='dreambooth',
    tags="just a test",
    auto_connect_arg_parser=True,
)

task.execute_remotely(queue_name="<=12GB", clone=False, exit_process=True)
import diffusers