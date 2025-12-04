# kafka_consumer_lag_monitor.py
from kafka import KafkaAdminClient, KafkaConsumer
from kafka import TopicPartition
from time import sleep
from prometheus_client import start_http_server, Gauge
import argparse
import os

# optional Prometheus metrics
lag_gauge = Gauge('kafka_consumer_partition_lag', 'Lag for consumer group partition', ['group', 'topic', 'partition'])

def get_end_offsets(bootstrap_servers, topic):
    admin = KafkaAdminClient(bootstrap_servers=bootstrap_servers)
    consumer = KafkaConsumer(bootstrap_servers=bootstrap_servers)
    partitions = consumer.partitions_for_topic(topic)
    res = {}
    if not partitions:
        return res
    for p in partitions:
        tp = TopicPartition(topic, p)
        consumer.assign([tp])
        consumer.seek_to_end(tp)
        res[p] = consumer.position(tp)
    consumer.close()
    admin.close()
    return res

def get_consumer_group_offsets(bootstrap_servers, group_id, topic):
    consumer = KafkaConsumer(group_id=group_id, bootstrap_servers=bootstrap_servers)
    partitions = consumer.partitions_for_topic(topic)
    res = {}
    for p in partitions:
        tp = TopicPartition(topic, p)
        # position for group is obtained by creating an assigned consumer and using committed()
        offs = consumer.committed(tp)
        res[p] = offs if offs is not None else 0
    consumer.close()
    return res

def monitor(broker, group, topic, port=None, interval=5):
    if port:
        start_http_server(port)
    while True:
        try:
            end = get_end_offsets(broker, topic)
            committed = get_consumer_group_offsets(broker, group, topic)
            for p in end:
                e = end.get(p, 0)
                c = committed.get(p, 0)
                lag = e - c if e is not None and c is not None else None
                print(f"Partition {p}: end={e} committed={c} lag={lag}")
                if port:
                    lag_gauge.labels(group=group, topic=topic, partition=str(p)).set(lag if lag is not None else 0)
            print("------")
        except Exception as exc:
            print("Error checking offsets:", exc)
        sleep(interval)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--broker", default="localhost:9092")
    parser.add_argument("--group", default="consumer-group")
    parser.add_argument("--topic", default="events")
    parser.add_argument("--prom-port", type=int, default=None)
    args = parser.parse_args()
    monitor(args.broker, args.group, args.topic, port=args.prom_port)
