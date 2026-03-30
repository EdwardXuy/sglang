from evalscope import TaskConfig, run_task


def run_evalscope_accuracy_test(
    model,
    eval_type="openai_ai",
    api_url=None,
    api_key="EMPTY",
    datasets="gsm8k",
    eval_batch_size=128,
    limit=10,
):
    task_config = TaskConfig(
        model=model,
        eval_type=eval_type,
        api_url=api_url,
        api_key=api_key,
        datasets=datasets,
        eval_batch_size=eval_batch_size,
        limit=limit,
    )

    run_task(task_config)
