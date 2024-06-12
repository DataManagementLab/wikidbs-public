import time
import hydra
from omegaconf import DictConfig
import logging
import random
import json
import math
from pathlib import Path

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../../conf", config_name="prepare.yaml")
def prepare_jsonl_parts(cfg: DictConfig):

    before = time.time()
    with open(cfg.wikidata_jsonl, "r") as profiling_file:
        all_lines = profiling_file.readlines()
    log.info(f"Read profiling file with {len(all_lines)} lines in {time.time() - before:.2f} seconds.")  # 25.55s for profiling_info_final.jsonl

    before = time.time()
    random.seed(170846269)
    random.shuffle(all_lines)
    log.info(f"Shuffled lines in {time.time() - before:.2f} seconds.")  # 55.03s for profiling_info_final.jsonl

    topic_dicts_to_process = []
    predicate_counter = {}

    before = time.time()
    for line in all_lines: 
        topic_dict = json.loads(line)
        if topic_dict["predicate_label"] not in predicate_counter.keys():
            predicate_counter[topic_dict["predicate_label"]] = 0

        topic_dict["idx"] = predicate_counter[topic_dict["predicate_label"]]
        topic_dicts_to_process.append(topic_dict)
        predicate_counter[topic_dict["predicate_label"]] += 1
    topic_dicts_to_process.sort(key=lambda x: x["idx"])
    topic_dicts_to_process = topic_dicts_to_process[:cfg.total_topics_limit]
    log.info(f"Planning to crawl {len(topic_dicts_to_process)} databases")
    log.info(f"Chose lines to process in {time.time() - before:.2f} seconds.")

    # split topics_dicts_to_process in cfg.num_separate_files files
    file_group_size = math.ceil(len(topic_dicts_to_process) / cfg.num_separate_files)
    log.info(f"{file_group_size} topics in each file")

    file_groups = []
    for left in range(0, len(topic_dicts_to_process), file_group_size):
        file_groups.append(topic_dicts_to_process[left:left + file_group_size])

    # write every group back into a seperate jsonl file
    file_name_base = f"total_{len(topic_dicts_to_process)}"

    for i, group in enumerate(file_groups):
        group_file_name = Path(cfg.output_folder) / f"{file_name_base}_part_{i}.jsonl"
    
        with open(group_file_name, "w", encoding="utf8") as output_file:
            for topic_dict in group:
                output_file.write(json.dumps(topic_dict) + '\n')

if __name__ == "__main__":
    prepare_jsonl_parts()