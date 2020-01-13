class EasyDict(dict):
    '''
    Dictionary object wrapper to make it accessible using dot(.)s.
    https://github.com/makinacorpus/easydict
    ---
    ex.
    dic1 = {'k1':v1, 'k2':{'k3':v2}}
    dic2 = EasyDict(dic1)
    # To get v1, v2:
    dic1['k1'] # => v1
    dic1['k2']['k3'] # => v2
    # This can be accessed by:
    dic2.k1 # => v1
    dic2.k2.k3 # => v2
    '''

    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith('__') and k.endswith('__')) and k not in ('update', 'pop'):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x)
                     if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(EasyDict, self).__setattr__(name, value)
        super(EasyDict, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k in d:
            setattr(self, k, d[k])

    def pop(self, k, d=None):
        delattr(self, k)
        return super(EasyDict, self).pop(k, d)
