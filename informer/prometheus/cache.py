from informer.prometheus.driver import PrometheusDriver


class PrometheusCacheBase(object):

    def __init__(self, **kwargs):
        """
        initialize
        :param data_list: data names and their labels
        :param kwargs:
            "url": prometheus url
            "data_list": data list
        """
        self.data_list: dict[str, dict] = kwargs["data_list"]
        self.driver = PrometheusDriver(url=kwargs["url"])

    def startup(self):
        pass



