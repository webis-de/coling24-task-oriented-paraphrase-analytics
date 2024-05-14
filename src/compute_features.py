import importlib
import logging
import multiprocessing
import os.path
import queue
import sys
import time
import multiprocessing as mp
import click
import collections

import config
import paraphrase.parser.parser


class AtomicCounter:
    """An atomic, thread-safe incrementing counter.
    """

    def __init__(self, initial=0):
        """Initialize a new atomic counter to given initial value (default 0)."""
        self.value = initial
        self._lock = mp.Lock()

    def increment(self, num=1):
        """Atomically increment the counter by num (default 1) and return the
        new value.
        """
        with self._lock:
            self.value += num
            return self.value


class ParseThread(mp.Process):
    def __init__(self, thread_id, buffer_parsed, parser, counter_parsed):
        super().__init__()
        self.thread_id = thread_id
        self.buffer_parsed = buffer_parsed
        self.parser = parser
        self.counter_parsed = counter_parsed

    def run(self):
        for paraphrase_set in self.parser.get_next():
            self.buffer_parsed.put(paraphrase_set)
            self.counter_parsed.increment()


class ComputeThread(mp.Process):
    def __init__(self, thread_id, in_buffer: mp.Queue, out_buffer: mp.Queue, counter_scored):
        super().__init__()
        self.thread_id = thread_id
        self.in_buffer = in_buffer
        self.out_buffer = out_buffer

        self.counter_scored = counter_scored
        self.metric_instances = configure_metrics()

    def run(self):
        global metric_conf, EXIT_FLAG
        while self.in_buffer.empty():
            pass

        while not self.in_buffer.empty():
            if self.in_buffer.empty():
                return

            paraphrase_set = self.in_buffer.get()
            compute_features(paraphrase_set, self.metric_instances)

            self.out_buffer.put(paraphrase_set)
            self.counter_scored.increment()


class StatusThread(mp.Process):
    def __init__(self, thread_id, buffer_parsed: mp.Queue, buffer_scored: mp.Queue, counter_parsed, counter_scored):
        super().__init__(daemon=True)
        self.thread_id = thread_id
        self.buffer_parsed = buffer_parsed
        self.buffer_scored = buffer_scored

        self.print_time = time.time()
        self.parsed_last = 0
        self.scored_last = 0

        self.buffer_parsed_last = 0
        self.buffer_scored_last = 0

        self.counter_parsed = counter_parsed
        self.counter_scored = counter_scored
        self.total_time = 0
        logging.basicConfig(encoding="utf-8", level=logging.DEBUG)
        handler = logging.StreamHandler(sys.stderr)
        self.logger = logging.getLogger()
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)

    def run(self):
        global EXIT_FLAG

        while not EXIT_FLAG:
            time_since_print = time.time() - self.print_time
            if time_since_print >= 1.:
                self.total_time += time_since_print
                print(
                    f"\rParsed: {self.counter_parsed.value} ({self.counter_parsed.value / self.total_time:.2f}/s) | "
                    f"Scored: {self.counter_scored.value} ({self.counter_scored.value / self.total_time:.2f}/s) | "
                    f"Buffer parsed: {self.buffer_parsed.qsize()} ({(self.buffer_parsed.qsize() - self.buffer_parsed_last) / time_since_print:.2f}/s) | "
                    f"Buffer scored: {self.buffer_scored.qsize()} ({self.buffer_scored.qsize() - self.buffer_scored_last / time_since_print:.2f}/s)",
                    file=sys.stderr)

                # sys.stdout.flush()
                sys.stderr.flush()
                self.print_time = time.time()
                self.parsed_last = self.counter_parsed.value
                self.scored_last = self.counter_scored.value
                self.buffer_parsed_last = self.buffer_parsed.qsize()
                self.buffer_scored_last = self.buffer_scored.qsize()


def compute_features(paraphrase_set, metric_instances):
    for metric in metric_instances:
        metric_instances[metric].compute(paraphrase_set)


def compute_features_process(in_buffer: multiprocessing.Queue, out_buffer: multiprocessing.Queue, counter: mp.Value):
    metric_instances = configure_metrics()
    try:
        while (paraphrase_set := in_buffer.get()) is not None:
            for metric in metric_instances:
                metric_instances[metric].compute(paraphrase_set)

            out_buffer.put(paraphrase_set)
            counter.value += 1
    except EOFError:
        return


def configure_metrics():
    metric_instances = {}
    for metric in metric_conf["metrics"]:
        metric_instance = get_class(metric["class"])(metric)
        metric_instances[metric["name"]] = metric_instance

    return metric_instances


def parse_process(in_buffer: multiprocessing.Queue, out_buffer: multiprocessing.Queue, counter: mp.Value):
    while (parser := in_buffer.get()) is not None:
        parser_instance = parser[0](parser[1])
        for paraphrase_set in parser_instance.get_next():
            out_buffer.put(paraphrase_set)
            counter.value += 1

    out_buffer.put(None)


def print_status_process(total_parsed, total_scored, buffer_parsed, buffer_scored):
    print_time = time.time()
    total_time = 0

    buffer_parsed_last = 0
    buffer_scored_last = 0

    while True:
        time_since_print = time.time() - print_time
        if time_since_print >= 1.:
            total_time += time_since_print
            print(
                f"\rParsed: {total_parsed.value} ({total_parsed.value / total_time:.2f}/s) | "
                f"Scored: {total_scored.value} ({total_scored.value / total_time:.2f}/s) | "
                f"Buffer parsed: {buffer_parsed.value} ({(buffer_parsed.value - buffer_parsed_last) / time_since_print:.2f}/s) | "
                f"Buffer scored: {buffer_scored.value} ({buffer_scored.value - buffer_scored_last / time_since_print:.2f}/s)",
                file=sys.stderr)

            sys.stderr.flush()
            buffer_parsed_last = buffer_parsed.value
            buffer_scored_last = buffer_scored.value
            print_time = time.time()


def get_class(dotted_path):
    comp = dotted_path.split(".")
    pkg_name = ".".join(comp[0: -1])
    class_name = comp[-1]

    if pkg_name not in sys.modules:
        mod = importlib.import_module(pkg_name)
    else:
        mod = sys.modules[pkg_name]

    return getattr(mod, class_name)


dataset_conf = config.load_config(config.DATASET_CONF_PATH)
metric_conf = config.load_config(config.METRIC_CONF_PATH)
EXIT_FLAG = False


@click.command()
@click.option("-d", "--dataset", "datasets",
              type=click.Choice([d["name"] for d in dataset_conf["datasets"]], case_sensitive=False),
              multiple=True, default=[])
@click.option("-t", "--task", "tasks",
              type=click.Choice([d["task"] for d in dataset_conf["datasets"]], case_sensitive=False),
              multiple=True, default=[])
@click.option("-m", "--multi-threaded", "multi_threaded",
              is_flag=True, default=False)
@click.option("-n", "--num-threads", "num_threads",
              type=int, default=4)
def main(datasets, tasks, multi_threaded, num_threads):
    if len(datasets) > 0 and len(tasks) > 0:
        raise click.ClickException("ERROR: \"--dataset\" and \"--task\" are mutually exclusive options")

    logging.basicConfig(encoding="utf-8", level=logging.DEBUG)

    counter_scored = multiprocessing.Value("d", 0)
    if multi_threaded:
        manager = mp.Manager()
        buffer_parser = manager.Queue()
        buffer_parsed = manager.Queue()
        buffer_scored = manager.Queue()

        counter_parsed = multiprocessing.Value("d", 0)
        counter_scored = multiprocessing.Value("d", 0)

        counter_buffer_parsed = multiprocessing.Value("d", 0)
        counter_buffer_scored = multiprocessing.Value("d", 0)

        status_process = mp.Process(target=print_status_process,
                                    args=(
                                        counter_parsed, counter_scored, counter_buffer_parsed, counter_buffer_scored))

        status_process.start()

        parser_process = mp.Process(target=parse_process, args=(buffer_parser, buffer_parsed, counter_parsed))
        parser_process.start()

        compute_processes = []
        for _ in range(num_threads):
            process = mp.Process(target=compute_features_process, args=(buffer_parsed, buffer_scored, counter_scored))
            process.start()
            compute_processes.append(process)

    for dataset in dataset_conf["datasets"]:
        if len(datasets) > 0 and dataset["name"] not in datasets:
            continue

        if len(tasks) > 0 and dataset["task"] not in tasks:
            continue

        out_path = os.path.join(os.path.dirname(__file__), "../out/" + dataset["name"] + ".jsonl")
        if type(dataset["files"][0]) == str:
            files = [os.path.join(dataset["path"], file_name) for file_name in dataset["files"]]
        else:
            files = []
            for file_set in dataset["files"]:
                for i in range(len(file_set)):
                    file_set[i] = os.path.join(dataset["path"], file_set[i])

                files.append(file_set)

        parser_class = get_class(dataset["parser"])

        with open(out_path, "w+") as out_file:
            if multi_threaded:
                buffer_parser.put((parser_class, files))

                try:
                    while True:
                        paraphrase_set = buffer_scored.get(timeout=10)
                        counter_buffer_parsed.value = buffer_parsed.qsize()
                        counter_buffer_scored.value = buffer_scored.qsize()
                        paraphrase_set.meta["task"] = dataset["task"]
                        out_file.write(paraphrase_set.to_json() + "\n")
                except queue.Empty:
                    continue

            else:
                parser_instance = parser_class(files)
                metric_instances = configure_metrics()
                print_time = time.time()
                total_time = 0

                for paraphrase_set in parser_instance.get_next():
                    compute_features(paraphrase_set, metric_instances)

                    counter_scored.value += 1
                    paraphrase_set.meta["task"] = dataset["task"]
                    out_file.write(paraphrase_set.to_json() + "\n")

                    time_since_print = time.time() - print_time
                    if time_since_print >= 1.:
                        total_time += time_since_print
                        logging.info(
                            f"\rScored: {counter_scored.value} ({counter_scored.value / total_time:.2f}/s)")
                        print_time = time.time()

    if multi_threaded:
        buffer_parser.put(None)

        parser_process.join()

        for process in compute_processes:
            process.join()

        status_process.kill()


if __name__ == '__main__':
    mp.set_start_method("spawn")

    main()
