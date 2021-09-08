
def get_data_loader_distributed(params, world_rank):
    if params.data_loader_config.startswith("dali"):
        if params.data_loader_config == "dali-inmem":
            from .data_loader_dali_cupy import get_data_loader_distributed
        elif params.data_loader_config == "dali-lomem":
            from .data_loader_dali import get_data_loader_distributed
        else:
            raise NotImplementedError(f"Error, data loader config {params.data_loader_config} not supported!")
    else:
        from .data_loader import get_data_loader_distributed

    return get_data_loader_distributed(params, world_rank)
