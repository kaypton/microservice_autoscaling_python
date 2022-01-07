from informer.cache.node.nodeCache import PrometheusNodeCpu
from informer.cache.node.nodeCache import PrometheusNodeMemory
from informer.cache.container.cadvisorCache import PrometheusCadvisorContainer

caches = {
    "prometheus.node.cpu": PrometheusNodeCpu,
    "prometheus.node.memory": PrometheusNodeMemory,
    "prometheus.container.cadvisor": PrometheusCadvisorContainer
}



