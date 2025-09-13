#######################################################################
# Name: runner.py
#
# - Wrapped by RLRunner to define Runner as a Ray object.
# - Each runner contains a worker class that interacts with the environment.
#######################################################################

from .parameter import *

import os
import torch
import ray
import numpy as np
from .model import PolicyNet, QNet
from .worker import Worker


class Runner(object):
    def __init__(self, meta_agent_id):
        self.meta_agent_id = meta_agent_id
        self.device = torch.device('cuda') if USE_GPU else torch.device('cpu')
        self.local_network = PolicyNet(INPUT_DIM, EMBEDDING_DIM)
        self.local_q_net = QNet(INPUT_DIM, EMBEDDING_DIM)
        self.local_network.to(self.device)
        self.local_q_net.to(self.device)

    def get_weights(self):
        return self.local_network.state_dict()

    def set_policy_net_weights(self, weights):
        self.local_network.load_state_dict(weights)

    def set_q_net_weights(self, weights1):
        self.local_q_net.load_state_dict(weights1)

    def do_job(self, episode_number):
        save_img = True if episode_number % SAVE_IMG_GAP == 0 else False
        n_agent = np.random.randint(NUM_ROBOTS_MIN, NUM_ROBOTS_MAX+1, 1)[0]     
        worker = Worker(self.meta_agent_id, n_agent, self.local_network, self.local_q_net, episode_number, device=self.device, save_image=save_img, greedy=False)
        worker.work(episode_number)

        job_results = worker.episode_buffer
        perf_metrics = worker.perf_metrics
        return job_results, perf_metrics

    def job(self, weights_set, episode_number):
        print("\n", GREEN, "starting episode {} on metaAgent {}".format(episode_number, self.meta_agent_id), NC)
        # set the local weights to the global weight values from the master network
        self.set_policy_net_weights(weights_set[0])
        self.set_q_net_weights(weights_set[1])

        job_results, metrics = self.do_job(episode_number)

        info = {
            "id": self.meta_agent_id,
            "episode_number": episode_number,
        }

        return job_results, metrics, info


### Wraps around Runner class to define class as a Ray object ### 
@ray.remote(num_cpus=1, num_gpus=NUM_GPU/NUM_META_AGENT)
class RLRunner(Runner):
    def __init__(self, meta_agent_id):        
        if GPU_RAY_MAPPING is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_RAY_MAPPING[meta_agent_id])
        super().__init__(meta_agent_id)


if __name__=='__main__':
    ray.init()
    runner = RLRunner.remote(0)
    job_id = runner.do_job.remote(1)
    out = ray.get(job_id)
    print(out[1])
