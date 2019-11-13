# CREDIT: https://github.com/lessw2020/Ranger-Mish-ImageWoof-5/blob/master/train.py
from fastai.callbacks import *

def fit_with_annealing(learn, num_epoch, lr, annealing_start=0.7):
    n = len(learn.data.train_dl)
    anneal_start = int(n*num_epoch*annealing_start)
    phase0 = TrainingPhase(anneal_start).schedule_hp('lr', lr)
    phase1 = TrainingPhase(n*num_epoch - anneal_start).schedule_hp('lr', lr, anneal=annealing_cos)
    phases = [phase0, phase1]
    sched = GeneralScheduler(learn, phases)
    learn.callbacks.append(sched)
    learn.fit(num_epoch)