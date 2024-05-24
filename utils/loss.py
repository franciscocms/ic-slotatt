import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np

def save_loss(LOSS_PATH, loss_dict):
  with open(LOSS_PATH + "/loss_dict.pkl", 'wb') as f:
    pkl.dump(loss_dict, f)

def save_loss_plot(loss_list, LOSS_PATH, resume_step, nsteps):
  step = int((resume_step + nsteps)/10)
  plt.figure()
  for i, loss in enumerate(loss_list):
      plt.plot(range(len(loss_list[loss])), loss_list[loss], label = loss)
  plt.xlabel('steps')
  plt.xticks(np.arange(0, len(loss_list[loss]), step))
  plt.ylabel('loss')
  plt.legend()
  plt.savefig(LOSS_PATH + '/loss_plot.png', bbox_inches = 'tight')
  plt.close()