from pyro import poutine
from pyro.infer.util import is_validation_enabled
from pyro.poutine.util import prune_subsample_sites
from pyro.util import check_model_guide_match, check_site_shape

def get_importance_trace(
    graph_type, max_plate_nesting, model, guide, args, kwargs, detach=False
):
    """
    Returns a single trace from the guide, which can optionally be detached,
    and the model that is run against it.
    """
    # Dispatch between callables vs GuideMessengers.
    unwrapped_guide = poutine.unwrap(guide)
    if isinstance(unwrapped_guide, poutine.messenger.Messenger):
        if detach:
            raise NotImplementedError("GuideMessenger does not support detach")
        guide(*args, **kwargs)
        model_trace, guide_trace = unwrapped_guide.get_traces()
    else:
        guide_trace = poutine.trace(guide, graph_type=graph_type).get_trace(
            *args, **kwargs
        )
        if detach:
            guide_trace.detach_()
        model_trace = poutine.trace(
            poutine.replay(model, trace=guide_trace), graph_type=graph_type
        ).get_trace(*args, **kwargs)

    if is_validation_enabled():
        check_model_guide_match(model_trace, guide_trace, max_plate_nesting)

    guide_trace = prune_subsample_sites(guide_trace)
    model_trace = prune_subsample_sites(model_trace)

    model_trace.compute_log_prob()
    guide_trace.compute_score_parts()
    if is_validation_enabled():
        for site in model_trace.nodes.values():
            if site["type"] == "sample":
                check_site_shape(site, max_plate_nesting)
        for site in guide_trace.nodes.values():
            if site["type"] == "sample":
                check_site_shape(site, max_plate_nesting)

    return model_trace, guide_trace