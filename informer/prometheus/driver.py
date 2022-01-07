import logging

import numpy
import requests
import json
from urllib import parse


class QueryResult(object):

    def __init__(self, _result: str):
        result_dict = json.loads(_result)
        self.status = result_dict["status"]
        self.data = None
        if self.status == "success":
            self.data = result_dict["data"]

        self.result_type = None
        self.result = None
        self.result_metric_num = None

        if self.data is not None:
            self.result_type = self.data["resultType"]
            self.result = self.data["result"]

        if self.result is not None:
            self.result_metric_num = len(self.result)

    def get_metric_num(self):
        return self.result_metric_num

    def find_by_labels(self, labels: dict, hard: bool):
        if self.result is None:
            return None
        if labels is None:
            return None
        for res in self.result:  # find the first match element
            metric = res["metric"]
            if len(metric) != len(labels):  # key num not match
                if hard:
                    continue
            else:
                find = True
                for key in labels:
                    if key not in metric or str(labels[key]) != metric[key]:
                        find = False
                        break
                if find:
                    return res
        return None

    def get_ndarray_data(self, hard: bool, labels: dict = None):
        """
        if labels is none then return the first result
        if labels is not none, then return the match result
        if none result match the given labels then return none
        if no result available then return none
        :param hard: hard
        :param labels: labels
        :return: numpy.ndarray
        """
        if self.result is None:
            return None

        if labels is None:
            res = self.result[0]
        else:
            res = self.find_by_labels(labels, hard)
        if res is None:
            return None
        if self.result_type == "matrix":
            value = res["values"]
            return numpy.array(value)
        elif self.result_type == "vector":
            ret = [res["value"]]
            return numpy.array(ret)
        else:
            logging.warning("do not support other result type")
            return None

    def __str__(self) -> str:
        string = "==============\n"
        string += "status: " + self.status + "\n"
        string += ("result type: " + self.result_type + "\n") if self.result_type is not None else ""
        string += ("metric num:" + str(self.result_metric_num) + "\n") if self.result_metric_num is not None else ""
        string += ("result: " + str(self.get_ndarray_data()) + "\n") if self.result is not None else ""
        string += "==============\n"
        return string


class PrometheusDriver(object):
    def __init__(self,
                 url: str):
        """
        prometheus driver
        :param url: url. <http://localhost:9090>
        """
        self.url = url

        self.base_headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }

    def query_range(self,
                    data_name: str,
                    labels: dict,
                    start: float,
                    end: float,
                    step: float) -> QueryResult:
        """
        query range. detail: https://prometheus.io/docs/prometheus/latest/querying/api/
        :param data_name: data name
        :param labels: labels
        :param start: start (unix timestamp)
        :param end: end (unix timestamp)
        :param step: step (second)
        :return:
        """
        query = PrometheusDriver._format_data_name_and_label(data_name, labels)
        return self._query_range(query, start, end, step)

    def query_duration(self,
                       data_name: str,
                       labels: dict = None,
                       duration: str = None) -> QueryResult:
        """
        query for data using `data_name{label_name="label"}[3m]` format
        :param data_name: data_name
        :param labels: labels
        :param duration: duration string. example: 3m
        :return:
        """
        query = PrometheusDriver._format_data_name_and_label(data_name, labels)
        query += "[" + duration + "]" if duration is not None else ""
        return self._query(query)

    @classmethod
    def _format_data_name_and_label(cls,
                                    data_name: str,
                                    labels: dict = None):
        query = data_name
        if labels is not None:
            query += "{"
            first: bool = True
            for key in labels:
                if first:
                    first = False
                else:
                    query += ","
                query += key + "=\"" + labels[key] + "\""
            query += "}"
        return query

    def _query_range(self,
                     query: str,
                     start: float,
                     end: float,
                     step: float):
        url = self.url + '/api/v1/query_range'
        data = {
            "query": query,
            "start": start,
            "end": end,
            "step": step
        }
        form_data = parse.urlencode(data)

        ret = requests.post(url, headers=self.base_headers, data=form_data)
        return QueryResult(ret.text)

    def query_raw(self,
                  query: str) -> QueryResult:
        return self._query(query)

    def _query(self,
               query: str) -> QueryResult:
        url = self.url + '/api/v1/query'

        data = {
            "query": query
        }
        form_data = parse.urlencode(data)

        ret = requests.post(url, headers=self.base_headers, data=form_data)
        return QueryResult(ret.text)


def test_main(driver: PrometheusDriver,
              data_name: str,
              labels: dict = None,
              duration: str = None):
    print("query ==================")
    data = driver.query_duration(data_name, labels, duration)
    data2 = driver.query_range(data_name, labels, 1641367709.623, 1641367884.623, 5)
    print(data)
    print(data2)
    print("query ==================")
