import copy

class IterativeDistillManager:
    def __init__(self):
        self.sparsity_to_record = [0.1, 0.2, 0.3, 0.4]
        self.recorded_zs = []
        self.cur_to_record = 0
        self.target = 0.5
        self.reach_target = False
        self.iter_steps = -1

    def update(self, sparsity, l0_module, remain_steps):
        if self.reach_target:
            return
        if self.cur_to_record == len(self.sparsity_to_record):
            if sparsity >= self.target:
                self.reach_target = True
                self.iter_steps = remain_steps
            return
        target = self.sparsity_to_record[self.cur_to_record]
        if sparsity >= target:
            print('record zs at sparsity {}.'.format(target))
            zs = l0_module.forward(training=False)
            self.recorded_zs.append(
                {key:copy.deepcopy(zs[key].detach()) for key in zs.keys()}
            )
            self.cur_to_record += 1
            
    def check_use_iterative_distill(self, remain_steps):
        if not self.reach_target:
            return None
        assert remain_steps <= self.iter_steps
        steps_interval = self.iter_steps // (len(self.recorded_zs) + 1)
        phase = (remain_steps // steps_interval) - 1
        phase = min(phase, len(self.recorded_zs) - 1)
        if phase < 0:
            return None
        return self.recorded_zs[phase]
        

        