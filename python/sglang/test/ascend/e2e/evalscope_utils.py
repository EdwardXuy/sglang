from evalscope import TaskConfig, run_task

GENERATION_CONFIG_DEFAULT = {
    "do_sample": True,
    "max_tokens": 1024,
    "seed": 3407,
    "top_p": 0.8,
    "top_k": 20,
    "temperature": 0.7,
    "n": 1,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
    "time_out": 3600,
    "stream": True,
    "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
}

DATASET_DIR_DEFAULT = "/tmp/ddataset/"

def run_evalscope_accuracy_test(
    model,
    eval_type="openai_ai",
    api_url=None,
    api_key="EMPTY",
    generation_config=None,
    datasets=None,
    dataset_dir=None,
    eval_batch_size=128,
    limit=10,
):
    if generation_config is None:
        generation_config = GENERATION_CONFIG_DEFAULT
    if dataset_dir is None:
        dataset_dir = DATASET_DIR_DEFAULT
    task_config = TaskConfig(
        model=model,
        eval_type=eval_type,
        api_url=api_url,
        api_key=api_key,
        generation_config=generation_config,
        datasets=datasets,
        dataset_dir=dataset_dir,
        eval_batch_size=eval_batch_size,
        limit=limit,
    )

    run_task(task_config)
