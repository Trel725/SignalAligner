import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.fftpack
import tensorflow as tf


def run_mean(x):
    N = 6
    return np.abs(np.convolve(x, np.ones(N) / N, mode='same'))


class TimeSeqAligner(object):
    """Align a sequence of time-domain signals with drifting phase

    Arguments:
        data: matrix of signals to align (in rows)
        lr: learning rate (step size of gradient descent algorithm)
        int_len: FFT length. Signals are padded to that length when fft is performed.
                 Its recommended to have it few times longer then the signal so that
                 binning is not a problem
        optimizer: one of keras optimizers to perform gradient descent
        metric: metric to optimize. A callable (tf.function taking power spectrum as input
                and returning single value) or one of the
                - "snr" for signal-to-noise ratio (sum of peaks over sum of non-peaks)
                - "ssum" for sum of peaks (peaks are everything above self.thresh, set it to change behaviour)
                - "smax" for maximum of the signal (optimize height of the larges peak)
        adaptive_lr: whether to dynamically adjust lr during optimization. If metric is not improved for
                     more then alr_hysteresis steps, it it multiplied by (1-alr_delta)
        window: region of interest for signal to optimize. Tuple of ints in the range 0, int_len/2.
                E.g. we want to optimize signal in the range 20-30 Hz, int_len is 1024, nyquist freqency
                is 100 Hz. Then specify window as approx. (100, 150)
        cumulative: whether shifts are cumulative, i.e. GD optimizes differences between shifts. Leads to correlations
                    and problematic convergence, thus not recommended to use
        alr_hysteresis, alr_delta: parameters of adaptive learning rate
    """

    def __init__(self, data, lr=0.1,
                 int_len=1024,
                 optimizer="sgd",
                 metric="snr",
                 adaptive_lr=True,
                 window=(None, None),
                 cumulative=False,
                 alr_hysteresis=10,
                 alr_delta=0.01):
        super(TimeSeqAligner, self).__init__()
        self.data = data
        self.int_len = int_len
        self.cumulative = cumulative
        self.y_tf = tf.convert_to_tensor(data - data.mean(1)[:, None], dtype=tf.complex128)
        self.t_relative = tf.cast(tf.linspace(0, 1, self.int_len), tf.complex128)
        self.filter_indices = tf.range(self.int_len)
        self.lr = lr
        self.opt = tf.keras.optimizers.get(optimizer)
        self.opt.lr.assign(lr)
        self.reset()
        self.hysteresis = alr_hysteresis
        self.deltalr = alr_delta
        self.adaptive_lr = adaptive_lr
        self.window = window

        metrics = dict(snr=self._snr, ssum=self._ssum, smax=self._smax)
        if metric in metrics:
            self.metric = metrics[metric]
        elif callable(metric):
            self.metric = metric
        else:
            raise ValueError(f"Unknown metric! Must callable or one from {list(metrics.keys())}")

    def reset(self):
        '''resets optimization variables'''
        shifts = tf.zeros(self.data.shape[0], dtype=tf.float64)[:, None]
        self.shifts = tf.Variable(shifts)
        self.best_fit = tf.cast(0.0, tf.float64)
        self.best_solution = tf.identity(self.shifts)
        self.thresh = 3 * np.std(self.alignment(self.shifts))

    def fft(self, x):
        '''performs (possible padded) FFT'''
        K = 1
        if x.shape[0] > self.int_len * K:
            x = x[:self.int_len * K]
        res = scipy.fftpack.fft(np.pad(x - x.mean(), (0, self.int_len * K - len(x))))
        return res[:self.int_len * K // 2]

    def plot_raw_spectrogram(self):
        '''plots the (running mean-filtered) spectrogram'''
        plt.subplot(2, 1, 1)
        plt.imshow(np.apply_along_axis(run_mean, 0,
                                       np.apply_along_axis(
                                           lambda x: self.fft(x), 1, self.data)
                                       ),
                   aspect=1)
        plt.subplot(2, 1, 2)
        plt.plot(np.abs(self.fft(self.data.mean(0)))**2)

    @tf.function
    def generate_filter(self, d, *args):
        """ generates filter to perfrom circular subpixel shift.
        See publication below for details
        # Closed Form Variable Fractional Time Delay Using FFT
        """
        N = self.int_len  # t_relative.shape[0]
        filt = tf.ones(N, dtype=tf.complex128)
        filt = tf.where((self.filter_indices > 0) & (self.filter_indices < N // 2),
                        tf.exp(-1j * 2 * np.pi * d * self.t_relative),
                        filt)
        filt = tf.where(self.filter_indices > N // 2,
                        tf.exp(1j * 2 * np.pi * d * (1.0 - self.t_relative)),
                        filt)
        filt = tf.where(self.filter_indices == N,
                        tf.cast(tf.cos(np.pi * d), tf.complex128),
                        filt)
        return filt

    @tf.function
    def shift_times(self, shifts):
        """performs actual subpixel shifting of the signals by *shifts* pixels.
        Returns spectrum of shifted signal"""
        if self.cumulative:
            sh = tf.cumsum(shifts)
        else:
            sh = shifts
        shift_matrix = tf.map_fn(self.generate_filter, tf.cast(sh, tf.complex128))
        padded = tf.pad(self.y_tf, [[0, 0], [0, self.int_len - self.y_tf.shape[1]]])
        specs = tf.signal.fft(padded)
        shifted = specs * shift_matrix
        return shifted

    @tf.function
    def alignment(self, shifts):
        ''' returns power spectrum of aligned signal '''
        specs = self.shift_times(shifts)
        return tf.math.abs(tf.math.reduce_mean(specs, axis=0))**2

    @tf.function
    def _snr(self, x):
        # signal-to-noise ratio (sum of peaks over sum of non-peaks)
        sig = x
        signal_roi = sig[self.window[0]:self.window[1]]
        peaks = signal_roi > self.thresh
        power_signal = tf.math.reduce_sum(signal_roi[peaks])
        power_noise = tf.math.reduce_sum(signal_roi[~peaks])
        if self.window[0] is not None:
            power_noise += tf.math.reduce_sum(sig[:self.window[0]])
        if self.window[1] is not None:
            power_noise += tf.math.reduce_sum(sig[:self.window[1]])
        return power_signal / power_noise

    @tf.function
    def _ssum(self, x):
        # sum of peaks
        sig = x[self.window[0]:self.window[1]]
        peaks = sig > self.thresh
        return tf.math.reduce_sum(sig[peaks])

    @tf.function
    def _smax(self, x):
        # maximum of the signal (optimize height of the larges peak)
        sig = x[self.window[0]:self.window[1]]
        return tf.math.reduce_max(sig)

    @tf.function
    def get_grads(self, shifts):
        """ calculate gradients of metric by shifts """
        with tf.GradientTape() as g:
            g.watch(shifts)
            fit = self.metric(self.alignment(shifts))
            grad = g.gradient(fit, shifts)
        return grad, fit

    def optimize(self, n_iter=5000, lr=None):
        """ main optimization routine """
        bad_counter = 0
        if lr is not None:
            self.opt.lr.assign(lr)
        for i in range(n_iter):
            grad, fit = self.get_grads(self.shifts)
            if fit > self.best_fit:
                self.best_solution = tf.identity(self.shifts)
                self.best_fit = fit
                bad_counter = 0

            if fit < self.best_fit:
                self.shifts.assign(self.best_solution)
                bad_counter += 1

            if bad_counter > self.hysteresis and self.adaptive_lr:
                newlr = self.opt.lr * (1 - self.deltalr)
                self.opt.lr.assign(newlr)
                bad_counter = 0

            self.opt.apply_gradients([(-grad, self.shifts)])
            if i % 100 == 0:
                print(f"Fit: {fit.numpy():.3f}, lr: {self.opt.lr.numpy():.3f}")

    def get_aligned_sequence(self):
        """ return resulted aligned signal (in time space)"""
        return np.fft.ifft(self.shift_times(self.best_solution).numpy()).real[:, :self.data.shape[1]]

    def plot_result_freq(self):
        """ plot spectrum and spectrogram of resulted aligned signal """
        plt.subplot(2, 1, 1)
        plt.plot(self.alignment(self.best_solution).numpy())
        plt.plot(self.alignment(tf.zeros_like(self.best_solution, dtype=tf.float64)).numpy())
        plt.subplot(2, 1, 2)
        plt.imshow(np.apply_along_axis(run_mean, 0,
                                       np.apply_along_axis(
                                           lambda x: x, 1, self.shift_times(self.best_solution).numpy())
                                       )[:, :self.int_len // 2],
                   aspect=1)

    def plot_result_time(self):
        """ plot original and aligned signal in time space"""
        plt.subplot(2, 1, 1)
        corr = self.get_aligned_sequence().mean(0)
        plt.plot(corr)
        plt.title("Corrected")
        plt.subplot(2, 1, 2)
        plt.plot(self.data.mean(0) - self.data.mean())
        plt.title("Original")
        plt.tight_layout()

    def plot_threshold(self):
        """ plot threshold, used in peak detection"""
        plt.plot(self.alignment(self.shifts).numpy())
        plt.plot(self.alignment(tf.zeros(self.data.shape[0], dtype=tf.float64)[:, None]).numpy())
        plt.hlines(self.thresh, 0, self.int_len)
