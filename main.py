import test.gymPendulumV1.gymPendulumV1Main as test_main
import informer.prometheus.driver as p_driver


if __name__ == "__main__":
    test_main.main()
    p_driver.test_main(p_driver.PrometheusDriver("http://222.201.144.237:9090"),
                       "system_cpu_system_usage",
                       None,
                       "3m")

