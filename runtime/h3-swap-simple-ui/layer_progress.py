"""Observe block dispatch without touching tensors or synchronizing the GPU."""
def callback(index, report, previous=None):
    def run(args, extra):
        report(index + 1)
        if previous is not None:
            return previous(args, extra)
        return extra['original_block'](args)
    return run
