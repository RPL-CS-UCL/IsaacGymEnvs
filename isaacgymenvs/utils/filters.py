############# Filetr Class ####################

import collections
from absl import logging
import numpy as np
from scipy.signal import butter
import torch

ACTION_FILTER_ORDER = 2
ACTION_FILTER_LOW_CUT = 0.0
ACTION_FILTER_HIGH_CUT = 4.0

class TensorQueue:

  def __init__(self, maxlen, num_envs, num_joints, device, dtype=torch.float):
    self.maxlen = maxlen
    self.device = device
    self.dtype = dtype
    self.num_envs = num_envs
    self.num_joints = num_joints

    # Queue shape: (maxlen, num_envs, num_joints)
    self.queue = torch.zeros((num_envs,num_joints,maxlen), device=device, dtype=dtype, requires_grad=False)

  def reset_idx(self, idx):
    self.queue[idx,:,:] = 0

  def reset(self):
    self.queue = torch.zeros((self.num_envs,self.num_joints, self.maxlen), device=self.device, dtype=self.dtype, requires_grad=False)

  def appendleft(self, x):
    if len(x.shape) == 2:
      x = x.unsqueeze(-1)
    self.queue = torch.cat((x, self.queue[:,:,:-1]), dim=-1)

  def __str__(self):
    return str(self.queue)



class ActionFilter(object):
  """Implements a generic lowpass or bandpass action filter."""

  def __init__(self, a, b, order, num_joints, ftype='lowpass', num_envs=1):
    """Initializes filter.
    Either one per joint or same for all joints.
    Args:
      a: filter output history coefficients
      b: filter input coefficients
      order: filter order
      num_joints: robot DOF
      ftype: filter type. 'lowpass' or 'bandpass'
    """
    self.num_joints = num_joints
    self.num_envs = num_envs

    if isinstance(a, list):
      self.a = a
      self.b = b
    else:
      self.a = [a]
      self.b = [b]

    # Either a set of parameters per joint must be specified as a list
    # Or one filter is applied to every joint
    if not ((len(self.a) == len(self.b) == num_joints) or (
        len(self.a) == len(self.b) == 1)):
      raise ValueError('Incorrect number of filter values specified')

    # Normalize by a[0]
    for i in range(len(self.a)):
      self.b[i] /= self.a[i][0]
      self.a[i] /= self.a[i][0]

    # Convert single filter to same format as filter per joint
    if len(self.a) == 1:
      self.a *= num_joints
      self.b *= num_joints
    self.a = np.stack(self.a)
    self.b = np.stack(self.b)

    self.a = torch.tensor(self.a,dtype=torch.float,device=self.device,requires_grad=False)
    self.b = torch.tensor(self.b,dtype=torch.float,device=self.device,requires_grad=False)

    if ftype == 'bandpass':
      assert len(self.b[0]) == len(self.a[0]) == 2 * order + 1
      self.hist_len = 2 * order
    elif ftype == 'lowpass':
      assert len(self.b[0]) == len(self.a[0]) == order + 1
      self.hist_len = order
    else:
      raise ValueError('%s filter type not supported' % (ftype))

    logging.info('Filter shapes: a: %s, b: %s', self.a.shape, self.b.shape)
    logging.info('Filter type:%s', ftype)

    self.yhist = TensorQueue(maxlen=self.hist_len, num_envs=self.num_envs, num_joints=self.num_joints, device=self.device)
    self.xhist = TensorQueue(maxlen=self.hist_len, num_envs=self.num_envs, num_joints=self.num_joints, device=self.device)
    self.reset()

  def reset(self):
    """Resets the history buffers to 0."""
    self.xhist.reset()
    self.yhist.reset()

  def filter(self, x):
    """Returns filtered x."""
    xs = self.xhist.queue.clone()
    ys = self.yhist.queue.clone()
    term1 = torch.multiply(x,self.b[:,0])

    term2 = torch.multiply(xs,self.b[:,1:])
    term3 = torch.sum(term2,dim=-1)

    term4 = torch.multiply(ys,self.a[:,1:])
    term5 = torch.sum(term4,dim=-1)

    y = term1 + term3 - term5

    self.xhist.appendleft(x.clone())
    self.yhist.appendleft(y.clone())

    return y

  def init_history(self, x):
    """Initializes the history buffers to x."""
    for _ in range(self.hist_len):
      self.xhist.appendleft(x)
      self.yhist.appendleft(x)


class ActionFilterButter(ActionFilter):
  """Butterworth filter."""

  def __init__(self,
               lowcut=None,
               highcut=None,
               sampling_rate=None,
               order=ACTION_FILTER_ORDER,
               num_joints=None,
               device="cpu",
               num_envs = 1):
    """Initializes a butterworth filter.

    Either one per joint or same for all joints.

    Args:
      lowcut: list of strings defining the low cutoff frequencies.
        The list must contain either 1 element (same filter for all joints)
        or num_joints elements
        0 for lowpass, > 0 for bandpass. Either all values must be 0
        or all > 0
      highcut: list of strings defining the high cutoff frequencies.
        The list must contain either 1 element (same filter for all joints)
        or num_joints elements
        All must be > 0
      sampling_rate: frequency of samples in Hz
      order: filter order
      num_joints: robot DOF
      device: device to use for the filter
      num_envs: number of environments

    """
    self.lowcut = ([float(x) for x in lowcut]
                   if lowcut is not None else [ACTION_FILTER_LOW_CUT])
    self.highcut = ([float(x) for x in highcut]
                    if highcut is not None else [ACTION_FILTER_HIGH_CUT])

    self.device = device

    if len(self.lowcut) != len(self.highcut):
      raise ValueError('Number of lowcut and highcut filter values should '
                       'be the same')

    if sampling_rate is None:
      raise ValueError('sampling_rate should be provided.')

    if num_joints is None:
      raise ValueError('num_joints should be provided.')

    if np.any(self.lowcut):
      if not np.all(self.lowcut):
        raise ValueError('All the filters must be of the same type: '
                         'lowpass or bandpass')
      self.ftype = 'bandpass'
    else:
      self.ftype = 'lowpass'

    a_coeffs = []
    b_coeffs = []
    for i, (l, h) in enumerate(zip(self.lowcut, self.highcut)):
      if h <= 0.0:
        raise ValueError('Highcut must be > 0')

      b, a = self.butter_filter(l, h, sampling_rate, order)
      logging.info(
          'Butterworth filter: joint: %d, lowcut: %f, highcut: %f, '
          'sampling rate: %d, order: %d, num joints: %d', i, l, h,
          sampling_rate, order, num_joints)
      b_coeffs.append(b)
      a_coeffs.append(a)

    super(ActionFilterButter, self).__init__(
        a_coeffs, b_coeffs, order, num_joints, self.ftype, num_envs)

  def butter_filter(self, lowcut, highcut, fs, order=5):
    """Returns the coefficients of a butterworth filter.

    If lowcut = 0, the function returns the coefficients of a low pass filter.
    Otherwise, the coefficients of a band pass filter are returned.
    Highcut should be > 0

    Args:
      lowcut: low cutoff frequency
      highcut: high cutoff frequency
      fs: sampling rate
      order: filter order
    Return:
      b, a: parameters of a butterworth filter
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if low:
      b, a = butter(order, [low, high], btype='band')
    else:
      b, a = butter(order, [high], btype='low')
    return b, a

  def reset_idx(self, idx):
    """Resets the history buffers to 0 for the idx-th environment."""
    self.xhist.reset_idx(idx)
    self.yhist.reset_idx(idx)

if __name__ == '__main__':

    # Filter requirements.
    lower = [0.0]
    upper = [0.95]
    order = 2
    num_joints = 4
    ftype = 'lowpass'
    num_envs = 3
    sampling_rate = 100
    

    # QUEUE TESTS
    queue = TensorQueue(2, num_envs, num_joints, device="cpu")
    a = torch.ones((num_envs,num_joints))
    b = torch.ones((num_envs,num_joints)) * 2
    c = torch.ones((num_envs,num_joints)) * 3
    queue.appendleft(a)
    queue.appendleft(b)
    queue.appendleft(c)
    queue.reset_idx(1)
    queue.reset()

    # FILTER TESTS
    action_filter_butter = ActionFilterButter(lower,upper,sampling_rate,order,num_joints,"cpu",num_envs)
    init_hist = torch.ones((num_envs,num_joints)) * 5
    action_filter_butter.init_history(init_hist)
    action = torch.rand((num_envs,num_joints))
    action_filter_butter.filter(action)



