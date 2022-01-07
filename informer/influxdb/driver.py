from influxdb_client import InfluxDBClient
from influxdb_client.client import query_api


class InfluxDBDriver(object):

    def __init__(self,
                 url: str,
                 token: str):
        """
        influxDB driver
        :param url: url. example: http://localhost:8086
        :param token: token
        """
        self.client = InfluxDBClient(
            url=url,
            token=token
        )

    def _get_query_api(self,
                       query_options: query_api.QueryOptions = query_api.QueryOptions()):
        return self.client.query_api(query_options=query_options)





