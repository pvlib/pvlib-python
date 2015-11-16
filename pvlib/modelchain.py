"""
Stub documentation for the module.
"""

class ModelChain(object):
    """
    Basic class. Consider an abstract base class.
    """
    def __init__(self, *args, **kwargs):
        pass
    
    def run_model(self):
        pass


class MoreSpecificModelChain(ModelChain):
    """
    Something more specific.
    """
    def __init__(self, *args, **kwargs):
        super(MoreSpecificModelChain, self).__init__(**kwargs)
    
    def run_model(self):
        # overrides the parent ModelChain method
        pass